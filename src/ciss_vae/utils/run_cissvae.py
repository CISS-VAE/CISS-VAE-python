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
    cols_ignore = None, 
    n_clusters = None, 
    ## if n_clusters = None use leiden. 
    k_neighbors: int = 15,
    use_snn: bool = True,
    leiden_resolution: float = 0.5,
    leiden_objective: str = "CPM",
    seed = 42):
    """Cluster samples based on their missingness patterns using KMeans or Leiden.
    
    When n_clusters is None, uses Leiden on a kNN graph built from Jaccard distances
    over the binary missingness mask. Otherwise uses KMeans(n_clusters).
    
    Returns (labels, silhouette). Silhouette uses Jaccard for the mask.
    
    :param data: Input dataset containing missing values
    :type data: pandas.DataFrame
    :param cols_ignore: Column names to exclude from clustering analysis, defaults to None
    :type cols_ignore: list[str], optional
    :param n_clusters: Number of clusters for KMeans; if None, uses Leiden Clustering, defaults to None
    :type n_clusters: int, optional
    :param seed: Random seed for reproducible clustering, defaults to None
    :type seed: int, optional
    :param k_neighbors: Number of nearest neighbors for leiden knn
    :type k_neighbors: int, optional
    :param leiden_resolution: Resolution for leiden clustering. Default 0.5
    :type leiden_resolution: float, optional
    :param leiden_objective: One of {"CPM","RB","Modularity"}. Default CPM
    :type prop: str, {"CPM","RB","Modularity"}
    :return: Tuple of (cluster_labels, silhouette_score)
    :rtype: tuple[numpy.ndarray, float or None]
    :raises ImportError: If scikit-learn or other dependencies are not installed
    """
    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
    except ImportError as e:
        raise ImportError(
            "This function requires scikit-learn. Install with: pip install scikit-learn"
        ) from e

    # Step 1: Binary mask (1=missing, 0=observed) as boolean
    if cols_ignore is not None:
        mask_matrix = data.drop(columns=cols_ignore).isna().to_numpy(dtype=bool)
    else:
        mask_matrix = data.isna().to_numpy(dtype=bool)

    if n_clusters is None:
        # Leiden on Jaccard
        labels = _leiden_from_knn(
            mask_matrix,
            metric="jaccard",
            k=k_neighbors,
            resolution=leiden_resolution,
            objective=leiden_objective,
            seed=seed,
            weight_scheme="1-minus",  # Jaccard similarity = 1 - distance
        )
        X_for_sil = mask_matrix
        sil_metric = "jaccard"
    else:
        # KMeans on the binary mask
        n_init = "auto"
        try:
            _ = KMeans(n_clusters=2, n_init=n_init)
        except TypeError:
            n_init = 10
        km = KMeans(n_clusters=n_clusters, n_init=n_init, random_state=seed)
        labels = km.fit_predict(mask_matrix.astype(float))
        X_for_sil = mask_matrix
        sil_metric = "jaccard"

    # Silhouette if ≥2 clusters and all cluster sizes ≥2
    unique, counts = np.unique(labels, return_counts=True)
    silhouette = None
    if len(unique) > 1 and np.all(counts >= 2):
        from sklearn.metrics import silhouette_score
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
    """Cluster samples based on their per-feature missingness proportions.

    If n_clusters is None -> Leiden on a kNN graph.
    Else -> KMeans(n_clusters).
    Returns (labels, silhouette). Silhouette uses the same metric.

    Parameters
    ----------
    prop_matrix : (n_samples, n_features)
        Rows = samples, columns = features; entries in [0, 1].
    n_clusters : int or None
        If None -> Leiden; else KMeans(n_clusters).
    seed : int or None
        Random seed for KMeans (and Leiden RNG).
    k_neighbors : int
        k for the kNN graph used by Leiden.
    leiden_resolution : float
        Resolution parameter for Leiden (CPM or RB objectives).
    leiden_objective : {"CPM","RB","Modularity"}
        Leiden objective; CPM recommended.
    metric : {"euclidean","cosine"}
        Distance metric used to build the graph and for silhouette.
    scale_features : bool
        Standardize columns before clustering.
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
# --------------------
# Func 2: Make dataset & run VAE
# --------------------

def run_cissvae(data, val_proportion = 0.1, replacement_value = 0.0, columns_ignore = None, print_dataset = True, do_not_impute_matrix = None,## dataset params
clusters = None, n_clusters = None,     k_neighbors: int = 15,
    leiden_resolution: float = 0.5,
    leiden_objective: str = "CPM", seed = 42, missingness_proportion_matrix = None, scale_features = False,## clustering params
hidden_dims = [150, 120, 60], latent_dim = 15, layer_order_enc = ["unshared", "unshared", "unshared"],
layer_order_dec=["shared", "shared",  "shared"], latent_shared=False, output_shared=False, batch_size = 4000,
return_model = True,## model params
epochs = 500, initial_lr = 0.01, decay_factor = 0.999, beta= 0.001, device = None, ## initial training params
max_loops = 100, patience = 2, epochs_per_loop = None, initial_lr_refit = None, decay_factor_refit = None, beta_refit = None, ## refit params
verbose = False,
return_clusters = True,
return_silhouettes = False,
return_history = False, 
return_dataset = False,
):
    """End-to-end pipeline for Clustering-Informed Shared-Structure Variational Autoencoder (CISS-VAE).
    
    Complete workflow that handles data preparation, optional clustering on missingness patterns,
    initial VAE training, and iterative imputation-refitting loops until convergence. Returns
    the final reconstructed dataset and optionally the trained model, clustering metrics, and training history.
    
    :param data: Input data matrix with potential missing values
    :type data: pandas.DataFrame or numpy.ndarray or torch.Tensor
    :param val_proportion: Fraction of non-missing entries per cluster to mask for validation, defaults to 0.1
    :type val_proportion: float, optional
    :param replacement_value: Value used to fill masked validation entries during training, defaults to 0.0
    :type replacement_value: float, optional
    :param columns_ignore: Column names/indices to exclude from validation masking, defaults to None
    :type columns_ignore: list[int or str], optional
    :param print_dataset: Whether to print dataset summary information, defaults to True
    :type print_dataset: bool, optional
    :param clusters: Precomputed cluster labels per sample; if None, performs clustering, defaults to None
    :type clusters: array-like, optional
    :param n_clusters: Number of clusters for KMeans; if None with clusters=None, uses Leiden Clustering, defaults to None
    :type n_clusters: int, optional
    :param k_neighbors: Number of nearest neighbors for leiden knn
    :type k_neighbors: int, optional
    :param leiden_resolution: Resolution for leiden clustering. Default 0.5
    :type leiden_resolution: float, optional
    :param leiden_objective: One of {"CPM","RB","Modularity"}. Default CPM
    :type leiden_objective: str, {"CPM","RB","Modularity"}
    :param seed: Random seed for reproducibility, defaults to 42
    :type seed: int, optional
    :param missingness_proportion_matrix: Missingness proportion matrix for biomarker clustering, defaults to None
    :type missingness_proportion_matrix: pandas.DataFrame or numpy.ndarray, optional
    :param scale_features: Whether to scale features in proportion matrix clustering, defaults to False
    :type scale_features: bool, optional
    :param hidden_dims: Sizes of hidden layers in encoder/decoder, defaults to [150, 120, 60]
    :type hidden_dims: list[int], optional
    :param latent_dim: Dimensionality of VAE latent space, defaults to 15
    :type latent_dim: int, optional
    :param layer_order_enc: Shared/unshared specification for encoder layers, defaults to ["unshared", "unshared", "unshared"]
    :type layer_order_enc: list[str], optional
    :param layer_order_dec: Shared/unshared specification for decoder layers, defaults to ["shared", "shared", "shared"]
    :type layer_order_dec: list[str], optional
    :param latent_shared: Whether latent layer weights are shared across clusters, defaults to False
    :type latent_shared: bool, optional
    :param output_shared: Whether final output layer weights are shared across clusters, defaults to False
    :type output_shared: bool, optional
    :param batch_size: Number of samples per training batch, defaults to 4000
    :type batch_size: int, optional
    :param return_model: Whether to return the trained VAE model, defaults to True
    :type return_model: bool, optional
    :param epochs: Number of epochs for initial training phase, defaults to 500
    :type epochs: int, optional
    :param initial_lr: Initial learning rate for optimizer, defaults to 0.01
    :type initial_lr: float, optional
    :param decay_factor: Multiplicative factor for learning rate decay per epoch, defaults to 0.999
    :type decay_factor: float, optional
    :param beta: Weight of KL-divergence term in VAE loss, defaults to 0.001
    :type beta: float, optional
    :param device: Computation device ("cpu" or "cuda"); auto-selects if None, defaults to None
    :type device: str or torch.device, optional
    :param max_loops: Maximum number of imputation-refitting loops, defaults to 100
    :type max_loops: int, optional
    :param patience: Number of loops without improvement before early stopping, defaults to 2
    :type patience: int, optional
    :param epochs_per_loop: Epochs per refit loop; uses epochs if None, defaults to None
    :type epochs_per_loop: int, optional
    :param initial_lr_refit: Learning rate for refit loops; uses initial_lr if None, defaults to None
    :type initial_lr_refit: float, optional
    :param decay_factor_refit: LR decay for refit; uses decay_factor if None, defaults to None
    :type decay_factor_refit: float, optional
    :param beta_refit: KL weight for refit; uses beta if None, defaults to None
    :type beta_refit: float, optional
    :param verbose: Whether to print progress messages during training, defaults to False
    :type verbose: bool, optional
    :param return_clusters: Whether to return list of clusters, defaults to True
    :type return_clusters: bool, optional
    :param return_silhouettes: Whether to return clustering silhouette score, defaults to False
    :type return_silhouettes: bool, optional
    :param return_dataset: Whether to return full Clusterdataset object, defaults to False
    :type return_dataset: bool, optional
    :param return_history: Whether to return concatenated training history, defaults to False
    :type return_history: bool, optional
    :return: Returns imputed_dataset always; optionally returns model, clusters, silhouette score, and/or training history based on flags
    :rtype: pandas.DataFrame or tuple containing combinations of (pandas.DataFrame, CISSVAE, float, pandas.DataFrame)
    """
    
    import torch
    from torch.utils.data import DataLoader
    from ciss_vae.classes.vae import CISSVAE
    from ciss_vae.classes.cluster_dataset import ClusterDataset
    from ciss_vae.training.train_initial import train_vae_initial
    from ciss_vae.training.train_refit import impute_and_refit_loop

    # ------------
    # Set params
    # ------------
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if epochs_per_loop is None:
        epochs_per_loop = epochs
    
    if decay_factor_refit is None:
        decay_factor_refit = decay_factor

    if beta_refit is None: 
        beta_refit = beta

    silh = None


    # ------------
    # Cluster if needed
    # ------------
    if clusters is None:
        if missingness_proportion_matrix is None:
            clusters, silh = cluster_on_missing(
                data, 
                cols_ignore = columns_ignore, 
                n_clusters = n_clusters, 
                seed = seed, 
                k_neighbors = k_neighbors,
                leiden_resolution = leiden_resolution,
                leiden_objective = leiden_objective)
            
            if(verbose):
                nclusfound = len(np.unique(clusters))
                print(f"There were {nclusfound} clusters, with an average silhouette score of {silh}")

        else:
            clusters, silh = cluster_on_missing_prop(
                prop_matrix = missingness_proportion_matrix, 
                n_clusters = n_clusters, 
                seed = seed, 
                k_neighbors = k_neighbors,
                leiden_resolution = leiden_resolution,
                leiden_objective = leiden_objective, 
                scale_features = scale_features)

            if(verbose):
                nclusfound = len(np.unique(clusters))
                print(f"There were {nclusfound} clusters, with an average silhouette score of {silh}")

    # --------------------------
    # MAJOR FIX: Ensure that cluster labeling 
    # --------------------------
    unique_clusters = np.unique(clusters) 
    cluster_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_clusters)}
    clusters = np.array([cluster_mapping[label] for label in clusters])

    dataset = ClusterDataset(data = data, 
                            cluster_labels = clusters, 
                            val_proportion = val_proportion,
                            replacement_value = replacement_value, 
                            columns_ignore = columns_ignore,
                            do_not_impute = do_not_impute_matrix)

    if print_dataset:
        print("Cluster dataset:\n", dataset)
    
    train_loader = DataLoader(dataset, batch_size = batch_size, shuffle = True)

    vae = CISSVAE(
        input_dim=dataset.shape[1],
        hidden_dims = hidden_dims,
        latent_dim = latent_dim,
        layer_order_enc = layer_order_enc,
        layer_order_dec = layer_order_dec,
        latent_shared = latent_shared,
        output_shared = output_shared,
        num_clusters = dataset.n_clusters,
        debug = False
    )

    vae, initial_history_df = train_vae_initial(
        model=vae,
        train_loader=train_loader,
        epochs=epochs,
        initial_lr=initial_lr,
        decay_factor=decay_factor,
        beta=beta,
        device=device,
        verbose=verbose,
        return_history = return_history
    )

    imputed_dataset, vae, _, refit_history_df = impute_and_refit_loop(
        model=vae,
        train_loader=train_loader,
        max_loops=max_loops,
        patience=patience,
        epochs_per_loop=epochs_per_loop,
        initial_lr=initial_lr_refit, ## should start from last learning rate
        decay_factor=decay_factor_refit,
        beta=beta_refit,
        device=device,
        verbose=verbose,
        batch_size=batch_size,
    )
    
    # ----------------
    # Construct history dataframe
    # ----------------
    if return_history:
        if initial_history_df is not None and refit_history_df is not None:
            # Concatenate initial and refit histories
            combined_history_df = pd.concat([initial_history_df, refit_history_df], ignore_index=True)
        elif initial_history_df is not None:
            combined_history_df = initial_history_df
        elif refit_history_df is not None:
            combined_history_df = refit_history_df
        else:
            combined_history_df = None
        
    # -------------------
    # Return statements
    # -------------------

    # Build return tuple dynamically
    return_items = [imputed_dataset]

    if return_model:
        return_items.append(vae)

    if return_dataset:
        return_items.append(dataset)

    if return_clusters:
        return_items.append(clusters)

    if return_silhouettes:
        return_items.append(silh)

    if return_history:
        return_items.append(combined_history_df)

    # Return as tuple if multiple items, single item otherwise
    if len(return_items) == 1:
        return return_items[0]
    else:
        return tuple(return_items)

    


