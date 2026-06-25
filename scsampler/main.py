from __future__ import annotations

from typing import Any, Optional, Union

import numpy as np
from anndata import AnnData
from pyarrow import ChunkedArray
from scipy.sparse import coo_matrix, issparse, spmatrix
from sklearn.decomposition import TruncatedSVD

from .backends import build_neighbor_graph, resolve_backend, to_backend_array
from .uclab import uclab, uclab_from_graph, uclab_split


NEIGHBOR_GRAPH_THRESHOLD = 4096
MIN_NEIGHBOR_GRAPH_SIZE = 32

def scsampler(
    data: Union[AnnData, np.ndarray, spmatrix, ChunkedArray],
    fraction: Optional[float] = None,
    n_obs: Optional[int] = None,
    random_state: int = 0,
    alpha: int = 50,
    copy: bool = False,
    obsm: Optional[str] = "X_pca",
    dr_num: Optional[int] = None,
    obs_index: Any = 0,
    var_index: Any = 0,
    random_split: Optional[int] = None,
    backend: str = "auto",
    selection_method: str = "auto",
    n_neighbors: Optional[int] = None,
    neighbor_algorithm: str = "auto",
    neighbor_algorithm_kwds: Optional[dict] = None,
    metric: str = "euclidean",
    block_size: Optional[int] = None,
) -> Optional[AnnData]:
    """Subsample a matrix or AnnData object with CPU/GPU-aware diversity sampling."""
    X = _extract_matrix(data, obsm=obsm, obs_index=obs_index, var_index=var_index)
    old_n_obs = X.shape[0]
    if old_n_obs == 0:
        raise ValueError("`data` must contain at least one observation.")

    if dr_num is not None:
        svd = TruncatedSVD(n_components=dr_num, random_state=random_state)
        X = svd.fit_transform(X)

    new_n_obs = _resolve_n_obs(old_n_obs, fraction=fraction, n_obs=n_obs)
    if new_n_obs == old_n_obs:
        obs_indices = np.arange(old_n_obs, dtype=np.int64)
    else:
        rng = np.random.default_rng(random_state)
        resolved_backend = resolve_backend(backend, X)
        method = _resolve_selection_method(selection_method, resolved_backend, old_n_obs)
        if method == "neighbors":
            if isinstance(data, AnnData) and "distances" in data.obsp:
                graph = data.obsp["distances"]
                if not issparse(graph) or graph.format != "csr":
                    graph = graph.tocsr()
            else:
                neighbor_count = _resolve_neighbor_count(old_n_obs, n_neighbors)
                graph = build_neighbor_graph(
                    X,
                    n_neighbors=neighbor_count,
                    backend=resolved_backend,
                    metric=metric,
                    random_state=random_state,
                    algorithm=neighbor_algorithm,
                    algorithm_kwds=neighbor_algorithm_kwds,
                    block_size=block_size,
                )
            obs_indices = uclab_from_graph(graph, new_n_obs, alpha=alpha, rng=rng)
        else:
            split = 1 if random_split is None else random_split
            backend_X = to_backend_array(X, resolved_backend)
            obs_indices = uclab_split(backend_X, new_n_obs, alpha=alpha, rng=rng, drop_start=1, drop_rate=0, split=split)

    if isinstance(data, AnnData):
        if copy:
            return data[obs_indices].copy()
        data._inplace_subset_obs(obs_indices)
        return None
    if copy:
        return X[obs_indices].copy(), obs_indices
    return obs_indices


def _extract_matrix(data: Union[AnnData, np.ndarray, spmatrix, ChunkedArray], obsm: Optional[str], obs_index: Any, var_index: Any) -> Any:
    if isinstance(data, AnnData):
        if obsm is not None and obsm in data.obsm:
            return data.obsm[obsm]
        return data.X
    if isinstance(data, ChunkedArray):
        values = np.asarray(data.to_numpy())
        obs = np.asarray(obs_index)
        var = np.asarray(var_index)
        if obs.ndim == 0 or var.ndim == 0 or obs.size != values.size or var.size != values.size:
            raise ValueError("ChunkedArray input requires `obs_index` and `var_index` arrays matching the flattened data size.")
        return coo_matrix((values, (obs, var))).tocsr()
    return data


def _resolve_n_obs(old_n_obs: int, fraction: Optional[float], n_obs: Optional[int]) -> int:
    if n_obs is not None:
        new_n_obs = int(n_obs)
    elif fraction is not None:
        if fraction < 0 or fraction > 1:
            raise ValueError(f"`fraction` needs to be within [0, 1], not {fraction}")
        new_n_obs = int(fraction * old_n_obs)
    else:
        raise ValueError("Either pass `n_obs` or `fraction`.")
    if new_n_obs < 1:
        raise ValueError("The requested sample size must be at least 1 observation.")
    if new_n_obs > old_n_obs:
        raise ValueError("The requested sample size cannot exceed the number of observations.")
    return new_n_obs


def _resolve_selection_method(selection_method: str, backend: str, n_obs: int) -> str:
    if selection_method not in {"auto", "exact", "neighbors"}:
        raise ValueError("`selection_method` must be one of {'auto', 'exact', 'neighbors'}.")
    if selection_method != "auto":
        return selection_method
    if backend == "gpu" or n_obs >= NEIGHBOR_GRAPH_THRESHOLD:
        return "neighbors"
    return "exact"


def _resolve_neighbor_count(n_obs: int, n_neighbors: Optional[int]) -> int:
    if n_obs < 2:
        raise ValueError("Neighbor-graph sampling requires at least two observations.")
    if n_neighbors is None:
        n_neighbors = min(max(MIN_NEIGHBOR_GRAPH_SIZE, int(np.ceil(np.sqrt(n_obs)))), n_obs - 1)
    if n_neighbors < 1:
        raise ValueError("`n_neighbors` must be at least 1.")
    return min(int(n_neighbors), n_obs - 1)