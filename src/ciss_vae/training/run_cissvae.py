"""run_cissvae takes in the dataset as an input and (optionally) clusters on missingness before running ciss_vae full model."""
from __future__ import annotations
from typing import Optional, Sequence, Tuple, Union
import pandas as pd
import numpy as np
# --------------------
# Func 2: Make dataset & run VAE
# --------------------

def run_cissvae(
    data, 
    val_proportion = 0.1, 
    replacement_value = 0.0, 
    columns_ignore = None, 
    print_dataset = True, 
    imputable_matrix = None,
    binary_feature_mask = None,
    ## dataset params
    clusters = None, 
    n_clusters = None,     
    k_neighbors: int = 15,
    leiden_resolution: float = 0.5,
    leiden_objective: str = "CPM", 
    seed = 42, 
    missingness_proportion_matrix = None, 
    scale_features = False,
    ## clustering params
    hidden_dims = [150, 120, 60], 
    latent_dim = 15, 
    layer_order_enc = ["unshared", "unshared", "unshared"],
    layer_order_dec=["shared", "shared",  "shared"], 
    latent_shared=False, 
    output_shared=False, 
    batch_size = 4000,
    return_model = True,
    ## model params
    epochs = 500, 
    initial_lr = 0.01, 
    decay_factor = 0.999, 
    weight_decay = 0.001,
    beta= 0.001, 
    device = None, 
    ## initial training params
    max_loops = 100,
    patience = 2, 
    epochs_per_loop = None, 
    initial_lr_refit = None, 
    decay_factor_refit = None, 
    beta_refit = None, 
    ## refit params
    verbose = False,
    return_clusters = False,
    return_silhouettes = False,
    return_history = False, 
    return_dataset = False,
    debug = False,
    ):
    """
    End-to-end pipeline for Clustering-Informed Shared-Structure Variational Autoencoder (CISS-VAE).

    This workflow prepares data (validation masking, optional feature/biomarker clustering inputs),
    optionally infers sample clusters, trains the VAE, and performs iterative
    impute–refit loops with early stopping. Returns the final imputed dataset and,
    optionally, the trained model and auxiliary artifacts.

    :param data: Input matrix with potential missing values, shape ``(n_samples, n_features)``.
    :type data: pandas.DataFrame or numpy.ndarray or torch.Tensor

    :param val_proportion: Per-cluster fraction of **non-missing** entries to mask for validation.
        May be a single float (global), a per-cluster sequence, or mapping.
        Defaults to ``0.1``.
    :type val_proportion: float or collections.abc.Sequence or collections.abc.Mapping or pandas.Series, optional

    :param replacement_value: Value used to fill masked validation entries in the *training* tensor.
        Does not affect the separate validation target tensor. Defaults to ``0.0``.
    :type replacement_value: float, optional

    :param columns_ignore: Columns to exclude from validation masking (names if ``data`` is a DataFrame,
        otherwise integer indices). Defaults to ``None``.
    :type columns_ignore: list[str or int] or None, optional

    :param print_dataset: If ``True``, prints dataset summary/statistics during setup. Defaults to ``True``.
    :type print_dataset: bool, optional

    :param imputable_matrix: Optional binary mask with the same shape as ``data`` indicating which entries
        are eligible for imputation. Use ``1`` to **allow** imputation and ``0`` to **exclude** from imputation.
        Defaults to ``None``.
    :type imputable_matrix: pandas.DataFrame or numpy.ndarray or torch.Tensor or None, optional

    :param clusters: Precomputed cluster labels for samples (length ``n_samples``). If ``None``,
        clustering may be performed depending on ``n_clusters`` and Leiden settings. Defaults to ``None``.
    :type clusters: array-like or None, optional

    :param n_clusters: If provided, performs KMeans with ``n_clusters``. If ``None`` and
        ``clusters`` is also ``None``, Leiden-based clustering is used. Defaults to ``None``.
    :type n_clusters: int or None, optional

    :param k_neighbors: Number of nearest neighbors for the Leiden KNN graph construction. Defaults to ``15``.
    :type k_neighbors: int, optional

    :param leiden_resolution: Resolution parameter for Leiden clustering. Defaults to ``0.5``.
    :type leiden_resolution: float, optional

    :param leiden_objective: Objective function for Leiden clustering. One of ``{"CPM", "RB", "Modularity"}``.
        Defaults to ``"CPM"``.
    :type leiden_objective: str, optional

    :param seed: Random seed for reproducibility. Defaults to ``42``.
    :type seed: int, optional

    :param missingness_proportion_matrix: Optional matrix for biomarker/feature clustering where each entry
        is the per-sample proportion of missingness for each feature. If provided, can guide clustering on
        missingness patterns. Defaults to ``None``.
    :type missingness_proportion_matrix: pandas.DataFrame or numpy.ndarray or None, optional

    :param scale_features: If ``True``, standardizes features for proportion-matrix-based clustering.
        Defaults to ``False``.
    :type scale_features: bool, optional

    :param hidden_dims: Encoder/decoder hidden layer sizes (mirrored architecture). Defaults to ``[150, 120, 60]``.
    :type hidden_dims: list[int], optional

    :param latent_dim: Dimensionality of the latent space. Defaults to ``15``.
    :type latent_dim: int, optional

    :param layer_order_enc: Per-layer specification for encoder blocks; values are
        ``"shared"`` or ``"unshared"``. Length should match ``hidden_dims``. Defaults to
        ``["unshared", "unshared", "unshared"]``.
    :type layer_order_enc: list[str], optional

    :param layer_order_dec: Per-layer specification for decoder blocks; values are
        ``"shared"`` or ``"unshared"``. Length should match ``hidden_dims``. Defaults to
        ``["shared", "shared", "shared"]``.
    :type layer_order_dec: list[str], optional

    :param latent_shared: If ``True``, shares latent layer parameters across clusters. Defaults to ``False``.
    :type latent_shared: bool, optional

    :param output_shared: If ``True``, shares final output layer across clusters. Defaults to ``False``.
    :type output_shared: bool, optional

    :param batch_size: Batch size for training. Defaults to ``4000``.
    :type batch_size: int, optional

    :param return_model: If ``True``, include the trained VAE model in the return tuple. Defaults to ``True``.
    :type return_model: bool, optional

    :param epochs: Number of epochs in the initial training phase. Defaults to ``500``.
    :type epochs: int, optional

    :param initial_lr: Initial learning rate for the optimizer. Defaults to ``0.01``.
    :type initial_lr: float, optional

    :param decay_factor: Multiplicative LR decay applied per epoch (``lr *= decay_factor``). Defaults to ``0.999``.
    :type decay_factor: float, optional

    :param beta: Weight of the KL-divergence term in the VAE loss for initial training. Defaults to ``0.001``.
    :type beta: float, optional

    :param device: Compute device, e.g., ``"cpu"`` or ``"cuda"``. If ``None``, selected automatically.
        Defaults to ``None``.
    :type device: str or torch.device or None, optional

    :param max_loops: Maximum number of impute–refit loops. Defaults to ``100``.
    :type max_loops: int, optional

    :param patience: Early stopping patience counted in *loops* without improvement. Defaults to ``2``.
    :type patience: int, optional

    :param epochs_per_loop: Number of epochs per refit loop. If ``None``, reuses ``epochs``. Defaults to ``None``.
    :type epochs_per_loop: int or None, optional

    :param initial_lr_refit: Learning rate for refit loops. If ``None``, uses ``initial_lr``. Defaults to ``None``.
    :type initial_lr_refit: float or None, optional

    :param decay_factor_refit: LR decay factor during refit loops. If ``None``, uses ``decay_factor``.
        Defaults to ``None``.
    :type decay_factor_refit: float or None, optional

    :param beta_refit: KL weight used in refit loops. If ``None``, uses ``beta``. Defaults to ``None``.
    :type beta_refit: float or None, optional

    :param verbose: If ``True``, prints progress and diagnostics. Defaults to ``False``.
    :type verbose: bool, optional

    :param return_clusters: If ``True``, include sample cluster labels in the return tuple. Defaults to ``False``.
    :type return_clusters: bool, optional

    :param return_silhouettes: If ``True``, include clustering silhouette score(s) in the return tuple.
        Defaults to ``False``.
    :type return_silhouettes: bool, optional

    :param return_history: If ``True``, include concatenated training/refit history (e.g., losses, metrics).
        Defaults to ``False``.
    :type return_history: bool, optional

    :param return_dataset: If ``True``, include the constructed/processed ``ClusterDataset`` object.
        Defaults to ``False``.
    :type return_dataset: bool, optional

    :param debug: If ``True``, enables additional checks/logging for troubleshooting. Defaults to ``False``.
    :type debug: bool, optional

    :returns: By default returns the imputed dataset. Depending on flags, may also return:
        ``model``, ``clusters``, ``silhouette_scores``, ``history``, and/or the ``ClusterDataset``.
        The order is:
        ``(imputed_dataset[, model][, clusters][, silhouette_scores][, history][, dataset])``.
    :rtype: pandas.DataFrame
        or tuple[ pandas.DataFrame
                [, CISSVAE]
                [, numpy.ndarray or pandas.Series]
                [, float or dict]
                [, pandas.DataFrame]
                [, ClusterDataset]
               ]"""
    
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

    def _normalize_cols_ignore(cols_ignore):
        if cols_ignore is None:
            return None
        # Convert sets/tuples/Index to list
        if not isinstance(cols_ignore, list):
            cols_ignore = list(cols_ignore)
        # Flatten one level if user passed [("a","b")] or [[...]]
        if len(cols_ignore) == 1 and isinstance(cols_ignore[0], (list, tuple)):
            cols_ignore = list(cols_ignore[0])
        # Ensure all are strings
        return [str(c) for c in cols_ignore]

    columns_ignore = _normalize_cols_ignore(columns_ignore)


    # ------------
    # Cluster if needed
    # ------------
    if clusters is None:
        from ciss_vae.utils.clustering import cluster_on_missing, cluster_on_missing_prop
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
    # MAJOR FIX: Ensure that cluster labeling is consistant and goes from 0 to n-1
    # --------------------------
    unique_clusters = np.unique(clusters) 
    cluster_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_clusters)}
    clusters = np.array([cluster_mapping[label] for label in clusters])
    

    dataset = ClusterDataset(data = data, 
                            cluster_labels = clusters, 
                            val_proportion = val_proportion,
                            replacement_value = replacement_value, 
                            columns_ignore = columns_ignore,
                            imputable = imputable_matrix,
                            binary_feature_mask=binary_feature_mask)

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
        debug = debug,
        binary_feature_mask = dataset.binary_feature_mask
    )

    if return_history: 
        vae, initial_history_df = train_vae_initial(
        model=vae,
        train_loader=train_loader,
        epochs=epochs,
        initial_lr=initial_lr,
        decay_factor=decay_factor,
        weight_decay = weight_decay,
        beta=beta,
        device=device,
        verbose=verbose,
        return_history = return_history
        )
    else:
        vae = train_vae_initial(
        model=vae,
        train_loader=train_loader,
        epochs=epochs,
        initial_lr=initial_lr,
        decay_factor=decay_factor,
        weight_decay = weight_decay,
        beta=beta,
        device=device,
        verbose=verbose,
        return_history = False
        )

    imputed_dataset, vae, _ = impute_and_refit_loop(
        model=vae,
        train_loader=train_loader,
        max_loops=max_loops,
        patience=patience,
        epochs_per_loop=epochs_per_loop,
        initial_lr=initial_lr_refit, ## should start from last learning rate
        weight_decay = weight_decay,
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
        combined_history_df = vae.training_history_
        
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

    


