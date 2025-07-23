"""run_cissvae takes in the dataset as an input and (optionally) clusters on missingness before running ciss_vae full model."""

import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from ciss_vae.classes.vae import CISSVAE
from ciss_vae.classes.cluster_dataset import ClusterDataset
from ciss_vae.training.train_initial import train_vae_initial
from ciss_vae.training.train_refit import impute_and_refit_loop
#from ciss_vae.utils.helpers import plot_vae_architecture


# -------------------
# Func 1: Cluster on missingness
# -------------------
## hdbscan if no k specified

def cluster_on_missing(data, cols_ignore = None, 
 n_clusters = None, seed = None, min_cluster_size = None, 
 cluster_selection_epsilon = 0.25):
    """
    Given pandas dataframe with missing data, clusters on missingness pattern and returns cluster labels.
        Parameters:
            - data : (pd.DataFrame) : 
                The original dataset
            - cols_ignore : (list[str]) default=None : 
                List of columns to ignore when clustering.
            - n_clusters : (int) default=None: 
                Set n_clusters to perform KMeans clustering with n_clusters clusters. If none, will use hdbscan for clustering.
            - seed : (int) default=None: 
                Set seed. 
            - min_cluster_size : (int) default=None: 
                Set min_cluster_size for hdbscan. 
            - cluster_selection_epsilon : (float) default=0.25: 
                Set cluster_selection_epsilon for hdbscan. 

        Returns:
            - clusters : cluster labels
            - silhouette : silhouette score
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
# Func 2: Make dataset & run VAE
# --------------------

def run_cissvae(data, val_percent = 0.1, replacement_value = 0.0, columns_ignore = None, print_dataset = True, ## dataset params
clusters = None, n_clusters = None, cluster_selection_epsilon = 0.25, seed = 42, ## clustering params
hidden_dims = [150, 120, 60], latent_dim = 15, layer_order_enc = ["unshared", "unshared", "unshared"],
layer_order_dec=["shared", "shared",  "shared"], latent_shared=False, output_shared=False, batch_size = 4000,
return_model = True,## model params
epochs = 500, initial_lr = 0.01, decay_factor = 0.999, beta= 0.001, device = None, ## initial training params
max_loops = 100, patience = 2, epochs_per_loop = None, initial_lr_refit = None, decay_factor_refit = None, beta_refit = None, ## refit params
verbose = False,
return_silhouettes = False
):
    """
    End-to-end pipeline to train a Clustering-Informed Shared-Structure Variational Autoencoder (CISS-VAE).

    This function handles data preparation, optional clustering, initial VAE training,
    and iterative refitting loops until convergence, returning the final reconstructed data and (optionally) the trained model.

    Parameters
    ----------
    data : pd.DataFrame | np.ndarray | torch.Tensor
        Input data matrix (samples × features), may contain missing values.
    val_percent : float, default=0.1
        Fraction of non-missing entries per cluster to mask for validation.
    replacement_value : float, default=0.0
        Value used to fill in masked entries (e.g., zero imputation).
    columns_ignore : list[int|str] or None, default=None
        Column names or indices to exclude from masking for valiation.
    print_dataset : bool, default=True
        If True, prints dataset summary.

    clusters : array-like or None, default=None
        Precomputed cluster labels per sample. If None, clustering will be performed.
    n_clusters : int or None, default=None
        Number of clusters to form with KMeans clustering if `clusters` is None. If None and clusters is None, will perform hdbscan clustering.
    cluster_selection_epsilon : float, default=0.25
        cluster_selection_epsilon for hdbscan clustering. 
    seed : int, default=42
        Random seed for reproducibility.

    hidden_dims : list of int, default=[150, 120, 60]
        Sizes of hidden layers in encoder/decoder (excluding latent layer).
    latent_dim : int, default=15
        Dimensionality of the VAE latent space.
    layer_order_enc : list of {"shared","unshared"}, default=["unshared","unshared","unshared"]
        Specify whether each encoder layer is shared across clusters or unique per cluster.
    layer_order_dec : list of {"shared","unshared"}, default=["shared","shared","shared"]
        Specify whether each decoder layer is shared or unique per cluster.
    latent_shared : bool, default=False
        If True, latent layer weights are shared across clusters.
    output_shared : bool, default=False
        If True, final output layer weights are shared across clusters.
    batch_size : int, default=4000
        Number of samples per training batch.
    return_model : bool, default=True
        If True, returns the trained VAE model; otherwise returns only reconstructed data.

    epochs : int, default=500
        Number of epochs for initial training.
    initial_lr : float, default=0.01
        Initial learning rate for the optimizer.
    decay_factor : float, default=0.999
        Multiplicative factor to decay the learning rate each epoch.
    beta : float, default=0.001
        Weight of the KL-divergence term in the VAE loss.
    device : str or torch.device or None, default=None
        Device for computation ("cpu" or "cuda"). If None, auto-selects.

    max_loops : int, default=100
        Maximum number of refitting loops after initial training.
    patience : int, default=2
        Number of loops with no improvement before early stopping.
    epochs_per_loop : int or None, default=None
        Number of epochs per refit loop; if None, uses `epochs`.
    initial_lr_refit : float or None, default=None
        Learning rate for refit loops; if None, uses `initial_lr`.
    decay_factor_refit : float or None, default=None
        LR decay for refit; if None, uses `decay_factor`.
    beta_refit : float or None, default=None
        KL weight for refit; if None, uses `beta`.

    verbose : bool, default=False
        If True, prints progress messages during training and refitting.
    return_silhouettes : bool, default=False
        If True, returns silhouettes from clustering
    """

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
        clusters, silh = cluster_on_missing(data, cols_ignore = columns_ignore, n_clusters = n_clusters, seed = seed, cluster_selection_epsilon = cluster_selection_epsilon)

    dataset = ClusterDataset(data = data, 
                            cluster_labels = clusters, 
                            val_percent = val_percent,
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

    vae = train_vae_initial(
        model=vae,
        train_loader=train_loader,
        epochs=epochs,
        initial_lr=initial_lr,
        decay_factor=decay_factor,
        beta=beta,
        device=device,
        verbose=verbose
    )

    imputed_dataset, vae, _, _ = impute_and_refit_loop(
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
        batch_size=batch_size
    )

    if return_model: 
        if return_silhouettes:
            return imputed_dataset, vae, silh
        else:
            return imputed_dataset, vae
        
    else:
        if return_silhouettes:
            return imputed_dataset, silh
        else:
            return imputed_dataset

    


