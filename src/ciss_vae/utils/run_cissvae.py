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
    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics import pairwise_distances
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
    

    if n_clusters is None:
        method = "hdbscan"

        # Create mask matrix (1 = missing, 0 = observed), drop ignored columns
        mask_matrix = mask_matrix.astype(bool).values

        # Jaccard requires boolean/binary NumPy array
        dists = pairwise_distances(mask_matrix, metric="jaccard")

        # Run HDBSCAN with precomputed distance matrix
        clusterer = hdbscan.HDBSCAN(metric='precomputed', min_cluster_size=min_cluster_size,
        cluster_selection_epsilon = cluster_selection_epsilon)
        clusters = clusterer.fit_predict(dists)
    else: 
        method = "kmeans"
        clusters = KMeans(n_clusters = n_clusters, random_state=seed).fit(mask_matrix).labels_


    return clusters

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
verbose = False
):
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


    # ------------
    # Cluster if needed
    # ------------
    if clusters is None:
        clusters = cluster_on_missing(data, cols_ignore = columns_ignore, n_clusters = n_clusters, seed = seed, cluster_selection_epsilon = cluster_selection_epsilon)

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
        return imputed_dataset, vae
    else:
        return imputed_dataset

