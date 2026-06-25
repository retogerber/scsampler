from __future__ import annotations

from math import sqrt
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix, issparse

from .backends import cp, to_numpy

try:
    from miniball import get_bounding_ball
except ImportError:  # pragma: no cover - optional dependency
    get_bounding_ball = None


MAX_SCALING_SAMPLE_SIZE = 256
MAX_PENALTY_VALUE = 1e30


def uclab(X: Any, n: int, alpha: int, rng: np.random.Generator, drop_start: float = 1, drop_rate: float = 0) -> np.ndarray:
    n_obs = X.shape[0]
    if n < 1 or n > n_obs:
        raise ValueError("`n` must be between 1 and the number of observations.")

    xp = cp if cp is not None and isinstance(X, cp.ndarray) else np
    scaled_X = _scale_matrix(X, rng)
    row_norms = _row_norms(scaled_X, xp)
    penalties = xp.zeros(n_obs, dtype=xp.float64)
    blocked = np.zeros(n_obs, dtype=bool)
    sample_index = np.empty(n, dtype=np.int64)

    next_index = int(rng.integers(n_obs))
    drop_step = int(drop_start * n)

    for i in range(n):
        if blocked[next_index]:
            next_index = _pick_next_index(penalties, blocked)
        sample_index[i] = next_index
        blocked[next_index] = True
        penalties = _set_value(penalties, np.array([next_index]), xp.inf, xp)
        penalties = _update_penalties(scaled_X, row_norms, penalties, next_index, alpha, xp)
        penalties = _set_value(penalties, np.array([next_index]), xp.inf, xp)

        if drop_rate and i + 1 == drop_step:
            penalties, blocked = _drop_candidates(penalties, blocked, n_remaining=n - (i + 1), drop_rate=drop_rate, xp=xp)
        if i + 1 < n:
            next_index = _pick_next_index(penalties, blocked)

    return sample_index


def uclab_from_graph(graph: csr_matrix, n: int, alpha: int, rng: np.random.Generator) -> np.ndarray:
    n_obs = graph.shape[0]
    if n < 1 or n > n_obs:
        raise ValueError("`n` must be between 1 and the number of observations.")

    graph = graph.tocsr()
    penalties = np.zeros(n_obs, dtype=np.float64)
    blocked = np.zeros(n_obs, dtype=bool)
    sample_index = np.empty(n, dtype=np.int64)

    next_index = int(rng.integers(n_obs))
    for i in range(n):
        if blocked[next_index]:
            next_index = _pick_next_index(penalties, blocked)
        sample_index[i] = next_index
        blocked[next_index] = True
        penalties[next_index] = np.inf

        start, stop = graph.indptr[next_index], graph.indptr[next_index + 1]
        neighbors = graph.indices[start:stop]
        distances = graph.data[start:stop]
        if distances.size:
            penalties[neighbors] += _distance_penalty(distances, alpha, np)
        penalties[next_index] = np.inf

        if i + 1 < n:
            next_index = _pick_next_index(penalties, blocked)

    return sample_index


def uclab_split(
    X: Any,
    n: int,
    alpha: int,
    rng: np.random.Generator,
    drop_start: float = 1,
    drop_rate: float = 0,
    split: int = 4,
) -> np.ndarray:
    if split <= 1 or n <= 1:
        return uclab(X, n, alpha, rng=rng, drop_start=drop_start, drop_rate=drop_rate)

    shuffled_index = rng.permutation(X.shape[0])
    counts = [n // split] * split
    for i in range(n % split):
        counts[i] += 1

    selected_parts = []
    offset = 0
    for part_index, chunk_index in enumerate(np.array_split(shuffled_index, split)):
        count = counts[part_index]
        if count == 0 or chunk_index.size == 0:
            continue
        chunk = X[chunk_index]
        local = uclab(chunk, count, alpha, rng=rng, drop_start=drop_start, drop_rate=drop_rate)
        selected_parts.append(chunk_index[local])
        offset += len(chunk_index)

    return np.concatenate(selected_parts)


def _scale_matrix(X: Any, rng: np.random.Generator) -> Any:
    n_obs = X.shape[0]
    sample_size = min(n_obs, MAX_SCALING_SAMPLE_SIZE)
    sample_index = rng.choice(n_obs, size=sample_size, replace=False)
    sample = to_numpy(_take_rows(X, sample_index))
    radius = 1.0
    try:
        if get_bounding_ball is None:
            raise RuntimeError("miniball is unavailable")
        _, r2 = get_bounding_ball(sample)
        radius = sqrt(max(r2, 1e-12))
    except Exception:
        centered = sample - sample.mean(axis=0, keepdims=True)
        radius = max(np.linalg.norm(centered, axis=1).max(initial=1.0), 1.0)
    radius = max(radius, 1.0)
    return X / radius


def _row_norms(X: Any, xp: Any) -> Any:
    if issparse(X):
        return np.asarray(X.multiply(X).sum(axis=1)).ravel()
    return xp.sum(X * X, axis=1)


def _update_penalties(X: Any, row_norms: Any, penalties: Any, sample_index: int, alpha: int, xp: Any) -> Any:
    if issparse(X):
        point = np.asarray(X[sample_index].toarray()).ravel()
        distances_sq = row_norms + row_norms[sample_index] - 2 * np.asarray(X @ point).ravel()
        distances_sq = np.maximum(distances_sq, 0.0)
        penalties += _distance_penalty(np.sqrt(distances_sq), alpha, np)
        return penalties

    point = X[sample_index]
    distances_sq = row_norms + row_norms[sample_index] - 2 * (X @ point)
    distances_sq = xp.maximum(distances_sq, 0.0)
    penalties += _distance_penalty(xp.sqrt(distances_sq), alpha, xp)
    return penalties


def _distance_penalty(distances: Any, alpha: int, xp: Any) -> Any:
    safe = xp.maximum(distances, 1e-12)
    log_penalty = -float(alpha) * xp.log(safe)
    return xp.exp(xp.minimum(log_penalty, np.log(MAX_PENALTY_VALUE)))


def _pick_next_index(penalties: Any, blocked: np.ndarray) -> int:
    if blocked.all():
        raise ValueError("No observations remain available for selection.")
    candidate = int(np.argmin(to_numpy(penalties)))
    if not blocked[candidate]:
        return candidate
    available = np.flatnonzero(~blocked)
    return int(available[0])


def _drop_candidates(penalties: Any, blocked: np.ndarray, n_remaining: int, drop_rate: float, xp: Any) -> tuple[Any, np.ndarray]:
    available = np.flatnonzero(~blocked)
    if available.size <= n_remaining:
        return penalties, blocked
    drop_number = min(int(available.size * drop_rate), available.size - n_remaining)
    if drop_number <= 0:
        return penalties, blocked
    penalty_values = to_numpy(penalties)[available]
    drop_local = np.argsort(penalty_values)[-drop_number:]
    drop_index = available[drop_local]
    blocked[drop_index] = True
    penalties = _set_value(penalties, drop_index, xp.inf, xp)
    return penalties, blocked


def _set_value(values: Any, indices: np.ndarray, fill_value: float, xp: Any) -> Any:
    if xp is np:
        values[indices] = fill_value
        return values
    values[xp.asarray(indices)] = fill_value
    return values


def _take_rows(X: Any, index: np.ndarray) -> Any:
    return X[index]