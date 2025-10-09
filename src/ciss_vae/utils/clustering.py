"""run_cissvae takes in the dataset as an input and (optionally) clusters on missingness before running ciss_vae full model."""
from __future__ import annotations
from typing import Optional, Sequence, Tuple, Union
import pandas as pd
import numpy as np

## helper function for getting leiden clusters from snn graph

import numpy as np
def _leiden_from_snn(
    X: np.ndarray,
    *,
    metric: str = "euclidean",
    k: int = 50,
    resolution: float = 0.01,
    objective: str = "CPM",
    mutual: bool = False,
    seed: int | None = None,
):
    try:
        from sklearn.neighbors import NearestNeighbors
        from scipy.sparse import csr_matrix
        import igraph as ig
        import leidenalg as la
    except ImportError as e:
        raise ImportError(
            "Leiden SNN requires scikit-learn, scipy, python-igraph, leidenalg."
        ) from e

    metric = metric.lower()
    algo = "auto" if metric in {"euclidean"} else "brute"

    # kNN connectivity (binary) graph
    nn = NearestNeighbors(n_neighbors=k, metric=metric, algorithm=algo)
    nn.fit(X)
    A = nn.kneighbors_graph(n_neighbors=k, mode="connectivity").tocsr()
    AT = A.T.tocsr()

    n = A.shape[0]
    src = []
    dst = []
    wts = []

    # For each i, compute shared-neighbor counts with its k neighbors
    for i in range(n):
        start, end = A.indptr[i], A.indptr[i + 1]
        neigh = A.indices[start:end]
        if neigh.size == 0:
            continue

        # overlap counts: (# shared neighbors) for each j in neigh
        # A[neigh] @ A[i, :].T gives overlap counts as a sparse matrix
        # Convert to dense array for easier handling
        overlaps = A[neigh].dot(A[i, :].T).toarray().flatten()  # Fixed: .A1 -> .toarray().flatten()

        deg_i = neigh.size
        deg_js = A.indptr[neigh + 1] - A.indptr[neigh]
        unions = deg_i + deg_js - overlaps

        # Jaccard weight in [0,1]
        with np.errstate(divide="ignore", invalid="ignore"):
            w = overlaps / np.maximum(unions, 1)
        
        # Optional: keep only mutual neighbors
        if mutual:
            incoming = AT.indices[AT.indptr[i] : AT.indptr[i + 1]]
            mask = np.isin(neigh, incoming, assume_unique=False)
            neigh = neigh[mask]
            w = w[mask]

        # keep positive weights
        pos = w > 0
        neigh = neigh[pos]
        w = w[pos]
        if neigh.size == 0:
            continue

        src.append(np.full(neigh.size, i, dtype=np.int32))
        dst.append(neigh.astype(np.int32))
        wts.append(w.astype(float))

    if not src:
        raise ValueError("SNN graph ended up empty; try mutual=False, increasing k, or using cosine.")

    src = np.concatenate(src)
    dst = np.concatenate(dst)
    wts = np.concatenate(wts)

    # Build symmetric weighted adjacency: take max(i->j, j->i)
    from scipy.sparse import coo_matrix
    W = coo_matrix((wts, (src, dst)), shape=(n, n)).tocsr()
    W = W.maximum(W.T)

    # Build igraph
    coo = W.tocoo()
    mask = coo.row < coo.col
    edges = list(zip(coo.row[mask].tolist(), coo.col[mask].tolist()))
    weights = coo.data[mask].astype(float)

    g = ig.Graph(n=n, edges=edges, directed=False)
    g.es["weight"] = list(map(float, weights))

    if seed is not None:
        la.Optimiser().set_rng_seed(int(seed))

    obj = objective.lower()
    if obj == "cpm":
        part = la.find_partition(
            g, la.CPMVertexPartition, weights="weight", resolution_parameter=resolution
        )
    elif obj in {"rb", "rbconfig", "rbconfiguration"}:
        part = la.find_partition(
            g, la.RBConfigurationVertexPartition, weights="weight", resolution_parameter=resolution
        )
    else:
        part = la.find_partition(g, la.ModularityVertexPartition, weights="weight")

    labels = np.asarray(part.membership, dtype=int)
    return labels
##  helper function for getting leiden clusters from knn graph

def _leiden_from_knn(
    X: np.ndarray,
    *,
    metric: str,
    k: int = 15,
    resolution: float = 0.5,
    objective: str = "CPM",  # {"CPM","RB","Modularity"}
    seed: Optional[int] = None,
    weight_scheme: str = "auto",  # "auto", "heat", "inv", "1-minus"
):
    """
    Build a kNN graph on X and run Leiden. Returns integer labels.
    """
    try:
        from sklearn.neighbors import NearestNeighbors
        import igraph as ig
        import leidenalg as la
    except ImportError as e:
        raise ImportError(
            "Leiden path requires: scikit-learn, python-igraph, and leidenalg.\n"
            "Install with: pip install python-igraph leidenalg scikit-learn"
        ) from e

    metric = metric.lower()
    # Build kNN graph with distances
    # Use brute for metrics like 'jaccard'/'cosine' to be safe and consistent
    algo = "auto" if metric in {"euclidean"} else "brute"
    nn = NearestNeighbors(n_neighbors=k, metric=metric, algorithm=algo)
    nn.fit(X)
    A = nn.kneighbors_graph(n_neighbors=k, mode="distance")  # CSR sparse, stores distances

    # Convert distances to similarity weights for Leiden
    d = A.data
    if weight_scheme == "auto":
        if metric in {"jaccard", "cosine"}:
            w = 1.0 - d
            if metric == "cosine":
                # cosine similarity can be negative if vectors aren't normalized
                w = np.clip(w, 0.0, None)
        elif metric == "euclidean":
            # Heat kernel based on median neighbor distance
            sigma = np.median(d) + 1e-12
            w = np.exp(-(d / sigma) ** 2)
        else:
            # Fallback: inverse distance
            w = 1.0 / (d + 1e-12)
    elif weight_scheme == "1-minus":
        w = 1.0 - d
    elif weight_scheme == "heat":
        sigma = np.median(d) + 1e-12
        w = np.exp(-(d / sigma) ** 2)
    elif weight_scheme == "inv":
        w = 1.0 / (d + 1e-12)
    else:
        raise ValueError("Unknown weight_scheme.")

    # Replace distances with weights
    from scipy.sparse import csr_matrix
    W = csr_matrix((w, A.indices, A.indptr), shape=A.shape)
    # Symmetrize by taking the maximum weight in either direction
    W = W.maximum(W.T)

    # Build igraph
    coo = W.tocoo()
    mask = coo.row < coo.col  # undirected: take upper triangle once
    edges = list(zip(coo.row[mask].tolist(), coo.col[mask].tolist()))
    weights = coo.data[mask].astype(float)

    import igraph as ig
    import leidenalg as la

    g = ig.Graph(n=W.shape[0], edges=edges, directed=False)
    g.es["weight"] = list(map(float, weights))

    if seed is not None:
        la.Optimiser().set_rng_seed(int(seed))

    obj = objective.lower()
    if obj == "cpm":
        part = la.find_partition(
            g, la.CPMVertexPartition, weights="weight", resolution_parameter=resolution
        )
    elif obj in {"rb", "rbconfig", "rbconfiguration"}:
        part = la.find_partition(
            g, la.RBConfigurationVertexPartition, weights="weight", resolution_parameter=resolution
        )
    else:
        # Modularity (no resolution parameter in standard form)
        part = la.find_partition(g, la.ModularityVertexPartition, weights="weight")

    labels = np.asarray(part.membership, dtype=int)
    return labels

# -------------------
# Func 1: Cluster on missingness
# -------------------
## leiden if no k specified

def cluster_on_missing(
    data, 
    cols_ignore=None, 
    n_clusters=None, 
    # if n_clusters = None use Leiden
    k_neighbors: int = 15,
    use_snn: bool = True,
    leiden_resolution: float = 0.5,
    leiden_objective: str = "CPM",
    seed: int = 42
):
    """
    Cluster samples based on their missingness patterns using KMeans or Leiden.

    When ``n_clusters`` is ``None``, performs Leiden clustering on a graph constructed
    from the binary missingness mask of the dataset. If ``use_snn=True``, builds a shared-nearest-neighbor (SNN)
    graph using Jaccard similarity; otherwise, constructs a standard kNN graph with Jaccard weights.
    Returns both the cluster labels and an optional silhouette score.

    :param data: Input dataset with potential missing values, shape ``(n_samples, n_features)``.
        Non-numeric columns should be excluded or specified in ``cols_ignore``.
    :type data: pandas.DataFrame

    :param cols_ignore: Column names to exclude from the missingness pattern clustering.
        Typically includes identifiers or static metadata columns. Defaults to ``None``.
    :type cols_ignore: list[str] or None, optional

    :param n_clusters: Number of clusters for KMeans. If ``None``, uses Leiden clustering
        on the binary missingness mask instead. Defaults to ``None``.
    :type n_clusters: int or None, optional

    :param k_neighbors: Number of nearest neighbors used when constructing the kNN/SNN
        graph for Leiden clustering. Defaults to ``15``.
    :type k_neighbors: int, optional

    :param use_snn: If ``True``, constructs a shared-nearest-neighbor (SNN) graph using
        mutual neighbor overlap weighted by Jaccard similarity. If ``False``, uses standard
        kNN graph weighting by Jaccard distance. Defaults to ``True``.
    :type use_snn: bool, optional

    :param leiden_resolution: Resolution parameter for Leiden clustering; higher values yield
        more clusters. Defaults to ``0.5``.
    :type leiden_resolution: float, optional

    :param leiden_objective: Objective function for Leiden optimization.
        One of ``{"CPM", "RB", "Modularity"}``. Defaults to ``"CPM"``.
    :type leiden_objective: str, optional

    :param seed: Random seed for reproducibility in KMeans and Leiden algorithms.
        Defaults to ``42``.
    :type seed: int, optional

    :returns: Tuple ``(labels, silhouette)``:
        - **labels** (:class:`numpy.ndarray`): Cluster assignments of length ``n_samples``.
        - **silhouette** (:class:`float` or ``None``): Silhouette score computed using Jaccard distance
          on the binary missingness mask; ``None`` if undefined.
    :rtype: tuple[numpy.ndarray, float or None]

    **Example**::

        >>> labels, silh = cluster_on_missing(data, n_clusters=None, use_snn=True)
        >>> np.unique(labels)
        array([0, 1, 2])
        >>> print(f"Silhouette: {silh:.3f}")
        Silhouette: 0.408
    """

    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
    except ImportError as e:
        raise ImportError(
            "This function requires scikit-learn. Install with: pip install scikit-learn"
        ) from e

    # Boolean missingness mask: True=missing, False=observed
    mask_matrix = (
        data.drop(columns=cols_ignore).isna().to_numpy(dtype=bool)
        if cols_ignore is not None
        else data.isna().to_numpy(dtype=bool)
    )

    # ---------------------------
    # Branch: Leiden or KMeans
    # ---------------------------
    if n_clusters is None:
        # Route to your existing helpers (keeps code unified)
        if use_snn:
            labels = _leiden_from_snn(
                mask_matrix,                 # boolean array OK (sklearn jaccard supports boolean)
                metric="jaccard",
                k=k_neighbors,
                resolution=leiden_resolution,
                objective=leiden_objective,
                mutual=False,
                seed=seed,
            )
        else:
            labels = _leiden_from_knn(
                mask_matrix,
                metric="jaccard",
                k=k_neighbors,
                resolution=leiden_resolution,
                objective=leiden_objective,
                seed=seed,
                weight_scheme="1-minus",     # similarity = 1 - jaccard distance
            )
        X_for_sil = mask_matrix
        sil_metric = "jaccard"

    else:
        # KMeans on the binary mask
        n_init = "auto"
        try:
            _ = KMeans(n_clusters=2, n_init=n_init)
        except TypeError:
            n_init = 10  # scikit-learn < 1.4
        km = KMeans(n_clusters=n_clusters, n_init=n_init, random_state=seed)
        labels = km.fit_predict(mask_matrix.astype(float))
        X_for_sil = mask_matrix
        sil_metric = "jaccard"

    # Silhouette if ≥2 clusters and all cluster sizes ≥2
    unique, counts = np.unique(labels, return_counts=True)
    silhouette = None
    if len(unique) > 1 and np.all(counts >= 2):
        silhouette = silhouette_score(X_for_sil, labels, metric=sil_metric)

    return labels, silhouette

def cluster_on_missing_prop(
    prop_matrix: Union[pd.DataFrame, np.ndarray],
    *,
    n_clusters: Optional[int] = None,
    seed: Optional[int] = None,
    # Leiden params (used when n_clusters is None)
    k_neighbors: int = 15,
    use_snn: bool = True,
    snn_mutual: bool = True,
    leiden_resolution: float = 0.5,
    leiden_objective: str = "CPM",
    metric: str = "euclidean",          # use "euclidean" or "cosine" for proportions
    scale_features: bool = False,
) -> Tuple[np.ndarray, Optional[float]]:
    """
    Cluster samples based on their per-feature missingness proportions using KMeans or Leiden.

    When ``n_clusters`` is ``None``, performs Leiden clustering on a graph constructed
    from the missingness proportion matrix. If ``use_snn=True``, builds a shared-nearest-neighbor (SNN)
    graph with Jaccard-based or metric-based similarity; otherwise uses a standard kNN graph.
    Returns both the cluster labels and an optional silhouette score.

    :param prop_matrix: Matrix of missingness proportions, shape ``(n_samples, n_features)``.
        Each entry represents the fraction of missing values for a feature within each sample.
        Values must lie in ``[0, 1]``.
    :type prop_matrix: pandas.DataFrame or numpy.ndarray

    :param n_clusters: Number of clusters for KMeans. If ``None``, uses Leiden clustering instead.
        Defaults to ``None``.
    :type n_clusters: int or None, optional

    :param seed: Random seed for KMeans initialization and Leiden reproducibility.
        Defaults to ``None``.
    :type seed: int or None, optional

    :param k_neighbors: Number of nearest neighbors for kNN/SNN graph construction used by Leiden.
        Defaults to ``15``.
    :type k_neighbors: int, optional

    :param use_snn: If ``True``, constructs a shared-nearest-neighbor (SNN) graph using
        mutual or weighted neighbor overlap. If ``False``, uses standard kNN.
        Defaults to ``True``.
    :type use_snn: bool, optional

    :param snn_mutual: If ``True``, retains only mutual nearest neighbors when building the SNN graph.
        Defaults to ``True``.
    :type snn_mutual: bool, optional

    :param leiden_resolution: Resolution parameter controlling cluster granularity in Leiden.
        Higher values produce more clusters. Defaults to ``0.5``.
    :type leiden_resolution: float, optional

    :param leiden_objective: Objective function for Leiden optimization.
        One of ``{"CPM", "RB", "Modularity"}``. Defaults to ``"CPM"``.
    :type leiden_objective: str, optional

    :param metric: Distance metric used for kNN graph construction and silhouette calculation.
        Recommended options are ``"euclidean"`` or ``"cosine"``.
        Defaults to ``"euclidean"``.
    :type metric: str, optional

    :param scale_features: Whether to standardize features (zero mean, unit variance) prior to clustering.
        Recommended when feature scales differ widely. Defaults to ``False``.
    :type scale_features: bool, optional

    :returns: Tuple ``(labels, silhouette)``:
        - **labels** (:class:`numpy.ndarray`): Cluster assignments of length ``n_samples``.
        - **silhouette** (:class:`float` or ``None``): Silhouette score based on the same metric;
          ``None`` if undefined (e.g., single cluster).
    :rtype: tuple[numpy.ndarray, float or None]

    **Example**::

        >>> labels, silh = cluster_on_missing_prop(prop_matrix, n_clusters=None, use_snn=True)
        >>> np.unique(labels)
        array([0, 1, 2, 3])
        >>> print(f"Silhouette: {silh:.3f}")
        Silhouette: 0.421
    """

    # Optional deps kept inside
    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        from sklearn.preprocessing import StandardScaler
    except ImportError as e:
        raise ImportError(
            "Optional dependencies required: scikit-learn (and leidenalg via _leiden_from_knn).\n"
            "Install with: pip install ciss_vae[clustering]"
        ) from e

    # Convert to array (handle DataFrame and custom classes with to_numpy)
    if isinstance(prop_matrix, pd.DataFrame):
        X = prop_matrix.to_numpy(dtype=float, copy=True)
    elif hasattr(prop_matrix, "to_numpy"):
        X = prop_matrix.to_numpy(dtype=float, copy=True)  # supports MissingnessMatrix
    else:
        X = np.asarray(prop_matrix, dtype=float).copy()

    if X.ndim != 2:
        raise ValueError("prop_matrix must be 2D (n_samples × n_features).")
    n_samples, n_features = X.shape

    # Sanity checks: finite and within [0,1]
    if not np.isfinite(X).all():
        raise ValueError("prop_matrix contains non-finite values (NaN/Inf).")
    if (X < 0).any() or (X > 1).any():
        X = np.clip(X, 0.0, 1.0)

    # Optionally scale features
    X_rows = X
    if scale_features:
        X_rows = StandardScaler().fit_transform(X_rows)

    metric = metric.lower()
    if metric not in {"euclidean", "cosine"}:
        raise ValueError("metric must be 'euclidean' or 'cosine' for proportion data.")

    # Clustering
    if n_clusters is None:
        # Leiden on kNN graph derived from the chosen metric
        if use_snn:
            labels = _leiden_from_snn(
                X_rows,
                metric=metric,
                k=k_neighbors,
                resolution=leiden_resolution,
                objective=leiden_objective,
                mutual=snn_mutual,
                seed=seed,
            )
        else:
            labels = _leiden_from_knn(
                X_rows,
                metric=metric,
                k=k_neighbors,
                resolution=leiden_resolution,
                objective=leiden_objective,
                seed=seed,
                weight_scheme="auto",
            )
        X_for_sil = X_rows
        sil_metric = metric
    else:
        n_init = "auto"
        try:
            _ = KMeans(n_clusters=2, n_init=n_init)
        except TypeError:
            n_init = 10
        km = KMeans(n_clusters=n_clusters, n_init=n_init, random_state=seed)
        labels = km.fit_predict(X_rows)
        X_for_sil = X_rows
        sil_metric = metric

    # Silhouette if valid
    unique, counts = np.unique(labels, return_counts=True)
    silhouette = None
    if len(unique) > 1 and np.all(counts >= 2):
        silhouette = silhouette_score(X_for_sil, labels, metric=sil_metric)

    return labels, silhouette