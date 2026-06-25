from __future__ import annotations

from typing import Any, Optional

import numpy as np
from scipy.sparse import csr_matrix, issparse

try:
    import cupy as cp
except ImportError:  # pragma: no cover - optional dependency
    cp = None

try:
    import rapids_singlecell as rsc
except ImportError:  # pragma: no cover - optional dependency
    rsc = None


def resolve_backend(requested: str, X: Any) -> str:
    if requested not in {"auto", "cpu", "gpu"}:
        raise ValueError("`backend` must be one of {'auto', 'cpu', 'gpu'}.")
    if requested == "cpu":
        return "cpu"
    if requested == "gpu":
        if cp is None:
            raise ImportError("GPU execution requires the optional `cupy` dependency.")
        return "gpu"
    if cp is not None and isinstance(X, cp.ndarray):
        return "gpu"
    return "cpu"


def to_numpy(X: Any) -> np.ndarray:
    if cp is not None and isinstance(X, cp.ndarray):
        return cp.asnumpy(X)
    return np.asarray(X)


def to_backend_array(X: Any, backend: str) -> Any:
    if backend == "gpu":
        if cp is None:
            raise ImportError("GPU execution requires the optional `cupy` dependency.")
        if issparse(X):
            return cp.asarray(X.toarray())
        return cp.asarray(X)
    return X


def build_neighbor_graph(
    X: Any,
    n_neighbors: int,
    backend: str,
    metric: str = "euclidean",
    random_state: int = 0,
    algorithm: str = "auto",
    algorithm_kwds: Optional[dict] = None,
    block_size: Optional[int] = None,
) -> csr_matrix:
    if n_neighbors < 1:
        raise ValueError("`n_neighbors` must be at least 1.")
    if metric != "euclidean":
        raise ValueError("Only the 'euclidean' metric is currently supported.")
    if backend == "gpu" and rsc is not None and cp is not None:
        return _build_rapids_neighbor_graph(
            X,
            n_neighbors=n_neighbors,
            random_state=random_state,
            algorithm=algorithm,
            algorithm_kwds=algorithm_kwds,
        )
    return _build_exact_neighbor_graph(X, n_neighbors=n_neighbors, backend=backend, block_size=block_size)


def _build_rapids_neighbor_graph(
    X: Any,
    n_neighbors: int,
    random_state: int,
    algorithm: str,
    algorithm_kwds: Optional[dict],
) -> csr_matrix:
    from anndata import AnnData

    gpu_X = to_backend_array(X, "gpu")
    adata = AnnData(X=gpu_X)
    kwargs = dict(algorithm_kwds or {})
    if algorithm == "auto":
        algorithm = "cagra"
    rsc.pp.neighbors(
        adata,
        n_neighbors=n_neighbors,
        use_rep="X",
        metric="euclidean",
        random_state=random_state,
        algorithm=algorithm,
        algorithm_kwds=kwargs or None,
    )
    distances = adata.obsp["distances"]
    if hasattr(distances, "get"):
        distances = distances.get()
    return distances.tocsr()


def _build_exact_neighbor_graph(
    X: Any,
    n_neighbors: int,
    backend: str,
    block_size: Optional[int],
) -> csr_matrix:
    device_X = to_backend_array(X, backend)
    xp = cp if backend == "gpu" else np
    n_obs = device_X.shape[0]
    if n_obs < 2:
        raise ValueError("Neighbor-graph sampling requires at least two observations.")
    n_neighbors = min(n_neighbors, n_obs - 1)
    if block_size is None:
        block_size = max(256, min(2048, 1_048_576 // max(1, device_X.shape[1])))

    if issparse(X) and backend == "cpu":
        row_norms = np.asarray(X.multiply(X).sum(axis=1)).ravel()
    else:
        row_norms = xp.sum(device_X * device_X, axis=1)

    indices = np.empty((n_obs, n_neighbors), dtype=np.int64)
    distances = np.empty((n_obs, n_neighbors), dtype=np.float64)

    for start in range(0, n_obs, block_size):
        stop = min(start + block_size, n_obs)
        if issparse(X) and backend == "cpu":
            dist_sq = _cpu_sparse_block_distances(X, row_norms, start, stop)
            block_xp = np
        else:
            block = device_X[start:stop]
            block_norms = row_norms[start:stop]
            dist_sq = block_norms[:, None] + row_norms[None, :] - 2 * (block @ device_X.T)
            dist_sq = xp.maximum(dist_sq, 0.0)
            block_xp = xp

        local_rows = block_xp.arange(stop - start)
        global_rows = block_xp.arange(start, stop)
        dist_sq[local_rows, global_rows] = block_xp.inf

        idx = block_xp.argpartition(dist_sq, kth=n_neighbors - 1, axis=1)[:, :n_neighbors]
        part = block_xp.take_along_axis(dist_sq, idx, axis=1)
        order = block_xp.argsort(part, axis=1)
        idx = block_xp.take_along_axis(idx, order, axis=1)
        dist = block_xp.sqrt(block_xp.take_along_axis(dist_sq, idx, axis=1))

        indices[start:stop] = to_numpy(idx)
        distances[start:stop] = to_numpy(dist)

    indptr = np.arange(0, n_obs * n_neighbors + 1, n_neighbors, dtype=np.int64)
    graph = csr_matrix((distances.reshape(-1), indices.reshape(-1), indptr), shape=(n_obs, n_obs))
    return graph.maximum(graph.T).tocsr()


def _cpu_sparse_block_distances(X: Any, row_norms: np.ndarray, start: int, stop: int) -> np.ndarray:
    block = X[start:stop]
    cross = block @ X.T
    if hasattr(cross, "toarray"):
        cross = cross.toarray()
    dist_sq = row_norms[start:stop, None] + row_norms[None, :] - 2 * np.asarray(cross)
    return np.maximum(dist_sq, 0.0)
