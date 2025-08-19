"""run_cissvae takes in the dataset as an input and (optionally) clusters on missingness before running ciss_vae full model."""
from __future__ import annotations
from typing import Optional, Sequence, Tuple, Union
import pandas as pd
import numpy as np

#from ciss_vae.utils.helpers import plot_vae_architecture


# -------------------
# Func 1: Cluster on missingness
# -------------------
## hdbscan if no k specified

def cluster_on_missing(data, cols_ignore = None, 
 n_clusters = None, seed = None, min_cluster_size = None, 
 cluster_selection_epsilon = 0.25, prop = False):
    """Cluster samples based on their missingness patterns using KMeans or HDBSCAN.
    
    Groups samples with similar patterns of missing values to identify subpopulations
    that may benefit from cluster-specific imputation strategies. Uses Jaccard distance
    on binary missingness indicators.
    
    :param data: Input dataset containing missing values
    :type data: pandas.DataFrame
    :param cols_ignore: Column names to exclude from clustering analysis, defaults to None
    :type cols_ignore: list[str], optional
    :param n_clusters: Number of clusters for KMeans; if None, uses HDBSCAN, defaults to None
    :type n_clusters: int, optional
    :param seed: Random seed for reproducible clustering, defaults to None
    :type seed: int, optional
    :param min_cluster_size: Minimum cluster size for HDBSCAN; if None, uses data.shape[0]//25, defaults to None
    :type min_cluster_size: int, optional
    :param cluster_selection_epsilon: HDBSCAN cluster selection threshold, defaults to 0.25
    :type cluster_selection_epsilon: float, optional
    :param prop: Legacy parameter, not used, defaults to False
    :type prop: bool, optional
    :return: Tuple of (cluster_labels, silhouette_score)
    :rtype: tuple[numpy.ndarray, float or None]
    :raises ImportError: If scikit-learn or hdbscan dependencies are not installed
    """

    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics import pairwise_distances, silhouette_score
        from sklearn.preprocessing import StandardScaler
        import hdbscan
    except ImportError as e:
        raise ImportError(
            "This function requires optional dependencies (scikit-learn and hdbscan). "
            "Install them with: pip install ciss_vae[clustering]"
        ) from e

    # -----------------
    # Step 1: Get mask matrix (1=missing, 0=observed)
    # -----------------
    if cols_ignore is not None:
        mask_matrix =  data.drop(columns=cols_ignore).isna().astype(int)
    else:
        mask_matrix = data.isna().astype(int)

    if min_cluster_size is None:
        min_cluster_size = data.shape[0] // 25 

    ## Get number of samples
    n_samples = mask_matrix.shape[0]
    

    if n_clusters is None:
        method = "hdbscan"

        ## Create mask matrix (1 = missing, 0 = observed), drop ignored columns
        mask_matrix = mask_matrix.astype(bool).values

        ## Jaccard requires boolean/binary NumPy array
        dists = pairwise_distances(mask_matrix, metric="jaccard")

        ## Run HDBSCAN with precomputed distance matrix
        clusterer = hdbscan.HDBSCAN(metric='precomputed', min_cluster_size=min_cluster_size,
        cluster_selection_epsilon = cluster_selection_epsilon)
        clusters = clusterer.fit_predict(dists)

        ## Get silhouette 
        sil_metric = "precomputed"
        x_for_sil = dists

    else: 
        method = "kmeans"
        clusters = KMeans(n_clusters = n_clusters, random_state=seed).fit(mask_matrix).labels_
        sil_metric = "jaccard"
        x_for_sil = mask_matrix

    # Compute silhouette score if possible
    unique_labels = set(clusters)
    if len(unique_labels) > 1 and all(list(clusters).count(l) > 1 for l in unique_labels):
        silhouette = silhouette_score(x_for_sil, clusters, metric=sil_metric)
    else:
        silhouette = None  # cannot compute silhouette with a single cluster or singleton clusters


    return clusters, silhouette

# --------------------
# Func 1b: Cluster on proportion of missingness by feature
# --------------------
def cluster_on_missing_prop(
    prop_matrix: Union[pd.DataFrame, np.ndarray],
    *,
    n_clusters: Optional[int] = None,
    seed: Optional[int] = None,
    min_cluster_size: Optional[int] = None,
    cluster_selection_epsilon: float = 0.25,
    metric: str = "euclidean",
    scale_features: bool = False,
) -> Tuple[np.ndarray, Optional[float]]:
    """Cluster data based on their per-sample missingness proportions.
    
    Groups data with similar patterns of missingness across samples, treating
    each feature as a point in the space of per-sample missingness rates. Useful
    for identifying features that are missing together systematically.
    
    :param prop_matrix: Matrix where rows are samples, columns are features, entries are missingness proportions [0,1]
    :type prop_matrix: pandas.DataFrame or numpy.ndarray
    :param n_clusters: Number of clusters for KMeans; if None, uses HDBSCAN, defaults to None
    :type n_clusters: int, optional
    :param seed: Random seed for KMeans reproducibility, defaults to None
    :type seed: int, optional
    :param min_cluster_size: HDBSCAN minimum cluster size; if None, uses max(2, n_features//25), defaults to None
    :type min_cluster_size: int, optional
    :param cluster_selection_epsilon: HDBSCAN cluster selection threshold, defaults to 0.25
    :type cluster_selection_epsilon: float, optional
    :param metric: Distance metric ("euclidean" or "cosine"), defaults to "euclidean"
    :type metric: str, optional
    :param scale_features: Whether to standardize feature vectors before clustering, defaults to False
    :type scale_features: bool, optional
    :return: Tuple of (feature_cluster_labels, silhouette_score)
    :rtype: tuple[numpy.ndarray, float or None]
    :raises ImportError: If scikit-learn or hdbscan dependencies are not installed
    :raises ValueError: If prop_matrix is not 2D or contains invalid values
    """


    # --- Imports kept inside to keep optional deps optional ---
    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        from sklearn.preprocessing import StandardScaler
        import hdbscan  # type: ignore
    except ImportError as e:
        raise ImportError(
            "Optional dependencies required: scikit-learn and hdbscan.\n"
            "Install with: pip install ciss_vae[clustering]"
        ) from e

    # --- Convert to array; capture column names to preserve alignment ---
    if isinstance(prop_matrix, pd.DataFrame):
        col_names: Sequence[str] = list(prop_matrix.columns)
        X = prop_matrix.to_numpy(copy=True)
    else:
        X = np.asarray(prop_matrix, dtype=float).copy()
        col_names = [f"col_{j}" for j in range(X.shape[1])]

    if X.ndim != 2:
        raise ValueError("prop_matrix must be 2D (n_samples × n_biomarkers).")

    n_samples, n_features = X.shape  # features = biomarkers

    # --- Sanity checks; clip to [0,1] if slightly out of bounds ---
    if not np.isfinite(X).all():
        raise ValueError("prop_matrix contains non-finite values (NaN/Inf).")
    if (X < 0).any() or (X > 1).any():
        X = np.clip(X, 0.0, 1.0)

    # --- Each column (biomarker) is a point in R^(n_samples) ---
    X_cols = X.T  # shape: (n_biomarkers, n_samples)

    if scale_features:
        X_cols = StandardScaler().fit_transform(X_cols)

    if min_cluster_size is None:
        min_cluster_size = max(2, n_features // 25)

    metric = metric.lower()
    if metric not in {"euclidean", "cosine"}:
        raise ValueError("metric must be 'euclidean' or 'cosine'.")

    # --- Clustering ---
    if n_clusters is None:
        clusterer = hdbscan.HDBSCAN(
            metric=metric,
            min_cluster_size=min_cluster_size,
            cluster_selection_epsilon=cluster_selection_epsilon,
        )
        labels = clusterer.fit_predict(X_cols)
        X_for_sil = X_cols
        sil_metric = metric
    else:
        # sklearn>=1.4 supports n_init="auto"; fall back if needed
        n_init = "auto"
        try:
            _ = KMeans(n_clusters=2, n_init=n_init)
        except TypeError:
            n_init = 10
        km = KMeans(n_clusters=n_clusters, n_init=n_init, random_state=seed)
        labels = km.fit_predict(X_cols)
        X_for_sil = X_cols
        sil_metric = metric

    # --- Silhouette (only if ≥2 clusters and no singletons) ---
    unique, counts = np.unique(labels, return_counts=True)
    if len(unique) > 1 and np.all(counts >= 2):
        silhouette = silhouette_score(X_for_sil, labels, metric=sil_metric)
    else:
        silhouette = None

    return labels, silhouette

# --------------------
# Func 2: Make dataset & run VAE
# --------------------

def run_cissvae(data, val_proportion = 0.1, replacement_value = 0.0, columns_ignore = None, print_dataset = True, ## dataset params
clusters = None, n_clusters = None, cluster_selection_epsilon = 0.25, seed = 42, missingness_proportion_matrix = None, scale_features = False,## clustering params
hidden_dims = [150, 120, 60], latent_dim = 15, layer_order_enc = ["unshared", "unshared", "unshared"],
layer_order_dec=["shared", "shared",  "shared"], latent_shared=False, output_shared=False, batch_size = 4000,
return_model = True,## model params
epochs = 500, initial_lr = 0.01, decay_factor = 0.999, beta= 0.001, device = None, ## initial training params
max_loops = 100, patience = 2, epochs_per_loop = None, initial_lr_refit = None, decay_factor_refit = None, beta_refit = None, ## refit params
verbose = False,
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
    :param n_clusters: Number of clusters for KMeans; if None with clusters=None, uses HDBSCAN, defaults to None
    :type n_clusters: int, optional
    :param cluster_selection_epsilon: HDBSCAN cluster selection threshold, defaults to 0.25
    :type cluster_selection_epsilon: float, optional
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
    :param return_silhouettes: Whether to return clustering silhouette score, defaults to False
    :type return_silhouettes: bool, optional
    :param return_history: Whether to return concatenated training history, defaults to False
    :type return_history: bool, optional
    :return: Returns imputed_dataset always; optionally returns model, silhouette score, and/or training history based on flags
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
            clusters, silh = cluster_on_missing(data, cols_ignore = columns_ignore, n_clusters = n_clusters, seed = seed, cluster_selection_epsilon = cluster_selection_epsilon)
        else:
            clusters, silh = cluster_on_missing_prop(prop_matrix = missingness_proportion_matrix, n_clusters = n_clusters, seed = seed, cluster_selection_epsilon=cluster_selection_epsilon, scale_features = scale_features)

    dataset = ClusterDataset(data = data, 
                            cluster_labels = clusters, 
                            val_proportion = val_proportion,
                            replacement_value = replacement_value, 
                            columns_ignore = columns_ignore)

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

    if return_silhouettes:
        return_items.append(silh)

    if return_history:
        return_items.append(combined_history_df)

    # Return as tuple if multiple items, single item otherwise
    if len(return_items) == 1:
        return return_items[0]
    else:
        return tuple(return_items)

    


