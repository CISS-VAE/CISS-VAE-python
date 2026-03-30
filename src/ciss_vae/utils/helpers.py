import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import torch.nn as nn
import torch
import torch.nn.functional as F
import pandas as pd
from ciss_vae.classes.vae import CISSVAE
import copy
import numpy as np

def plot_vae_architecture(model: nn.Module, 
title = None, 
color_shared = "skyblue", 
color_unshared ="lightcoral",
color_latent = "gold",
color_input = "lightgreen",
color_output = "lightgreen",
figsize=(16, 8),
return_fig = False,
fontsize_layer = 12,
fontsize_section=14,
fontsize_title=16,):
    """Plots a horizontal schematic of the VAE architecture, showing shared and cluster-specific layers.
    
    :param model: An instance of CISSVAE model to visualize
    :type model: nn.Module
    :param title: Title of the plot, defaults to None
    :type title: str, optional
    :param color_shared: Color for shared hidden layers, defaults to "skyblue"
    :type color_shared: str, optional
    :param color_unshared: Color for unshared hidden layers, defaults to "lightcoral"
    :type color_unshared: str, optional
    :param color_latent: Color for latent layer, defaults to "gold"
    :type color_latent: str, optional
    :param color_input: Color for input layer, defaults to "lightgreen"
    :type color_input: str, optional
    :param color_output: Color for output layer, defaults to "lightgreen"
    :type color_output: str, optional
    :param figsize: Size of the matplotlib figure, defaults to (16, 8)
    :type figsize: tuple, optional
    :param return_fig: Whether to return the figure object instead of displaying, defaults to False
    :type return_fig: bool, optional
    :param fontsize_layer: Font size of layer blocks, defaults to 12
    :type fontsize_layer: int, optional
    :param fontsize_section: Font size of encoder/decoder labels, defaults to 14
    :type fontsize_section: int, optional
    :param fontsize_title: Font size of title, defaults to 16
    :type fontsize_title: int, optional
    :return: Matplotlib figure object if return_fig is True, otherwise None
    :rtype: matplotlib.figure.Figure or None
    """
    fig, ax = plt.subplots(figsize=figsize)

    box_width = 3
    box_height = 0.8
    n_clusters = model.num_clusters
    cluster_gap = 1.0
    box_height_shared = box_height * n_clusters + (n_clusters - 2) * cluster_gap / 2
    x = 1  # starting x-coordinate
    x_gap = 5
    y_base = 0
    

    def draw_box(center, text, style="shared", color=color_shared, color_override = False):
        x0, y0 = center
        alpha = 0.95 if style == "shared" else 0.75
        if color_override:
            facecolor = color
        else:
            facecolor = color if style == "shared" else color_unshared
        if style == "shared": 
            box = Rectangle((x0 - box_width / 2, y0 - box_height_shared / 2),
                box_width, box_height_shared,
                linewidth=1.5, edgecolor='black',
                facecolor=facecolor, alpha=alpha)
        else: 
            box = Rectangle((x0 - box_width / 2, y0 - box_height / 2),
                box_width, box_height,
                linewidth=1.5, edgecolor='black',
                facecolor=facecolor, alpha=alpha)
        ax.add_patch(box)
        ax.text(x0, y0, text, fontsize=fontsize_layer, ha='center', va='center', weight='bold')

    def draw_arrow(start, end):
        ax.annotate("", xy=end, xytext=start,
                    arrowprops=dict(arrowstyle="->", lw=1.5))

    def draw_section_box(x_start, x_end, label):
        ax.add_patch(Rectangle(
            (x_start - x_gap / 2, y_base - (n_clusters * cluster_gap) / 2 - 1.0),
            x_end - x_start + x_gap,
            n_clusters * cluster_gap + 2.0,
            edgecolor="gray", facecolor="none", linestyle="--", linewidth=1.2
        ))
        ax.text((x_start + x_end) / 2, y_base + (n_clusters * cluster_gap) / 2 + 1.2,
                label, fontsize=fontsize_section, ha='center', weight='bold')

    # --------------------------
    # Input Layer
    # --------------------------
    in_dim = model.input_dim
    draw_box((x, y_base), f"Input\n{in_dim}", style="shared", color=color_input)

    encoder_start = x + x_gap
    x = encoder_start 
    # --------------------------
    # Encoder layers
    # --------------------------
    shared_idx = 0
    unshared_idx = 0
    for i, layer_type in enumerate(model.layer_order_enc):
        if layer_type == "shared":
            dim = model.encoder_layers[shared_idx][0].out_features
            draw_box((x, y_base), f"Enc {i+1}\n{dim}", style="shared")
            if i >= 0:
                draw_arrow((x + box_width/2 - x_gap, y_base), (x - box_width / 2, y_base))
            shared_idx += 1
        else:
            dim = list(model.cluster_encoder_layers.values())[0][unshared_idx][0].out_features
            for c in range(n_clusters):
                y = y_base + (c - (n_clusters - 1) / 2) * cluster_gap
                draw_box((x, y), f"Enc {i+1}\nC{c}\n{dim}", style="unshared")
                if i >= 0:
                    draw_arrow((x + box_width/2 - x_gap, y), (x - box_width / 2, y))
            unshared_idx += 1
        x += x_gap
    encoder_end = x - x_gap

    # --------------------------
    # Latent layer
    # --------------------------
    latent_dim = (
        model.fc_mu.out_features if model.latent_shared
        else list(model.cluster_fc_mu.values())[0].out_features
    )
    style = "shared" if model.latent_shared else "unshared"

    if style == "shared":
        draw_box((x, y_base), f"Latent\nμ/σ²\n{latent_dim}", style=style, color=color_latent, color_override=True)
        # Arrow from last encoder layer
        draw_arrow((x - x_gap + box_width / 2, y_base), (x - box_width / 2, y_base))
    else:
        for c in range(n_clusters):
            y_c = y_base + (c - (n_clusters - 1) / 2) * cluster_gap
            draw_box((x, y_c), f"Latent\nC{c}\nμ/σ²\n{latent_dim}", style=style, color=color_latent, color_override=True)
            draw_arrow((x - x_gap + box_width / 2, y_c), (x - box_width / 2, y_c))
    x += x_gap

    decoder_start = x
    # --------------------------
    # Decoder layers
    # --------------------------
    shared_idx = 0
    unshared_idx = 0
    for i, layer_type in enumerate(model.layer_order_dec):
        if layer_type == "shared":
            dim = model.decoder_layers[shared_idx][0].out_features
            draw_box((x, y_base), f"Dec {i+1}\n {dim}", style="shared")
            draw_arrow((x + box_width/2 - x_gap, y_base), (x - box_width / 2, y_base))
            shared_idx += 1
        else:
            dim = list(model.cluster_decoder_layers.values())[0][unshared_idx][0].out_features
            for c in range(n_clusters):
                y = y_base + (c - (n_clusters - 1) / 2) * cluster_gap
                draw_box((x, y), f"Dec {i+1}\nC{c}\n{dim}", style="unshared")
                draw_arrow((x + box_width/2 - x_gap, y), (x - box_width / 2, y))
            unshared_idx += 1
        x += x_gap
    decoder_end = x - x_gap

    # --------------------------
    # Final output layer
    # --------------------------
    try:
        # Shared final layer
        out_dim = model.final_layer.out_features
        draw_box((x, y_base), f"Output\n{out_dim}", style="shared", color=color_output)
        draw_arrow((x + box_width/2 - x_gap, y_base), (x - box_width / 2, y_base))
    except AttributeError:
        # Unshared final layers
        out_dim = list(model.cluster_final_layer.values())[0].out_features
        for c in range(n_clusters):
            y = y_base + (c - (n_clusters - 1) / 2) * cluster_gap
            draw_box((x, y), f"Output\nC{c}\n{out_dim}", style="unshared", color=color_output, color_override=True)
            draw_arrow((x + box_width/2 - x_gap, y), (x - box_width / 2, y))


    # --------------------------
    # Annotations
    # --------------------------
    draw_section_box(encoder_start, encoder_end, "Encoder")
    draw_section_box(decoder_start, decoder_end, "Decoder")

    ax.set_xlim(-1, x + 2)
    ax.set_ylim(y_base - (n_clusters * cluster_gap) / 2 - 2, y_base + (n_clusters * cluster_gap) / 2 + 2)
    ax.axis("off")
    ax.set_title(title, fontsize=fontsize_title, weight='bold')

    if return_fig:
        return fig
    else:
        plt.tight_layout()
        plt.show()





def get_imputed_df(model: CISSVAE, data_loader, device = "cpu"):
    """Given trained model and cluster dataset object, get imputed dataset as pandas DataFrame.
    
    Reconstructs missing values using the trained VAE model and returns the complete dataset
    with original scaling restored and validation entries replaced with true values.
    
    :param model: Trained CISSVAE model (should be in eval() mode)
    :type model: CISSVAE
    :param data_loader: DataLoader for the original ClusterDataset
    :type data_loader: torch.utils.data.DataLoader
    :param device: Device to run computations on, defaults to "cpu"
    :type device: str, optional
    :return: DataFrame containing imputed (unscaled) data with original row ordering
    :rtype: pandas.DataFrame
    """

    model.eval()
    dataset = data_loader.dataset

    # -------------------------------
    # Get imputed (normalized space, but already converted from logits)
    # -------------------------------
    imputed = get_imputed(model, data_loader, device)
    x_all = imputed.data.to(device)

    # -------------------------------
    # Prepare stats
    # -------------------------------
    means = torch.as_tensor(imputed.feature_means, dtype=torch.float32, device=device)
    stds = torch.as_tensor(imputed.feature_stds, dtype=torch.float32, device=device)

    stds = stds.clone()
    stds[stds == 0] = 1.0

    # -------------------------------
    # Convert + denormalize properly
    # -------------------------------
    x_all_denorm = x_all.clone()

    for name, cols in dataset.activation_groups.items():
        cols = torch.tensor(cols, device=device)

        # -------------------------
        # Continuous → denormalize
        # -------------------------
        if name == "continuous":
            x_all_denorm[:, cols] = x_all[:, cols] * stds[cols] + means[cols]

        # -------------------------
        # Binary → already sigmoid from get_imputed
        # -------------------------
        elif name == "binary":
            x_all_denorm[:, cols] = x_all[:, cols].clamp(0.0, 1.0)

        # -------------------------
        # Categorical → already one-hot from get_imputed
        # -------------------------
        else:
            x_all_denorm[:, cols] = x_all[:, cols]

    # -------------------------------------
    # Restore validation values
    # -------------------------------------
    val_data_tensor = dataset.val_data.to(device)
    val_mask_tensor = ~torch.isnan(val_data_tensor)

    x_all_denorm[val_mask_tensor] = val_data_tensor[val_mask_tensor]

    # -------------------------------------
    # Apply imputable mask
    # -------------------------------------
    if hasattr(dataset, "imputable") and dataset.imputable is not None:
        imputable_mask = dataset.imputable.to(device)
        x_all_denorm[imputable_mask == 0] = float("nan")

    # -------------------------------------
    # Convert to numpy / DataFrame
    # -------------------------------------
    x_all_denorm_np = x_all_denorm.detach().cpu().numpy()

    feature_names = getattr(
        dataset,
        "feature_names",
        [f"V{i}" for i in range(x_all.shape[1])]
    )

    df_unsorted = pd.DataFrame(
        x_all_denorm_np,
        columns=feature_names,
        index=dataset.indices.cpu().numpy(),
    )

    imputed_df = df_unsorted.sort_index()

    return imputed_df
    

def get_imputed(model, data_loader, device="cpu"):
    """Returns a ClusterDataset where originally missing values have been replaced with model reconstructions.
    
    Processes the dataset through the trained VAE model to reconstruct missing values,
    including validation-masked entries. The returned dataset maintains the same structure
    as the original but with missing values filled in.
    
    :param model: Trained VAE model
    :type model: nn.Module
    :param data_loader: DataLoader for the original ClusterDataset
    :type data_loader: torch.utils.data.DataLoader
    :param device: Torch device for computations, defaults to "cpu"
    :type device: str, optional
    :return: ClusterDataset with reconstructed values filled in at originally missing positions
    :rtype: ClusterDataset
    """
    model.eval()
    dataset = data_loader.dataset

    # Collect all batches
    all_recon = []
    all_masks = []
    all_indices = []

    ## NEW 11SEP2025 - Collect imputable masks if they exist
    all_imputable = []
    has_imputable = hasattr(dataset, 'imputable') and dataset.imputable is not None

    with torch.no_grad():
        for batch in data_loader:
            x_batch, cluster_batch, mask_batch, idx_batch = batch

            x_batch = x_batch.to(device)
            cluster_batch = cluster_batch.to(device)

            # Predict full reconstruction
            recon_batch, _, _ = model.forward(x_batch, cluster_batch, deterministic = True)

            all_recon.append(recon_batch.cpu())
            all_masks.append(mask_batch.cpu())
            all_indices.append(idx_batch)

            ## NEW 11SEP2025 - Add imputable mask thingie

            if has_imputable:
                imputable_batch = dataset.imputable[idx_batch]
                all_imputable.append(imputable_batch)

    # Concatenate all batches
    recon_all = torch.cat(all_recon, dim=0)
    mask_all = torch.cat(all_masks, dim=0)
    idx_all = torch.cat(all_indices, dim=0)

    ## NEW 11SEP2025 - imputable
    if has_imputable:
        imputable_all = torch.cat(all_imputable, dim=0)

    # Restore correct row order
    recon_sorted = torch.zeros_like(dataset.data)
    recon_sorted[idx_all] = recon_all

    ## NEW 11SEP2025 - imputable
    if has_imputable:
        imputable_sorted = torch.zeros_like(dataset.imputable)
        imputable_sorted[idx_all] = imputable_all

    # Convert logits → real values
    recon_real = recon_sorted.clone()

    for name, cols in dataset.activation_groups.items():
        cols = torch.tensor(cols)

        if name == "continuous":
            pass  # already correct scale (still normalized)

        elif name == "binary":
            recon_real[:, cols] = torch.sigmoid(recon_real[:, cols])

        else:  # categorical
            probs = torch.softmax(recon_real[:, cols], dim=1)
            idx = torch.argmax(probs, dim=1)

            recon_real[:, cols] = 0
            recon_real[torch.arange(recon_real.shape[0]), cols[idx]] = 1

    # Now insert reconstructed values
    new_data = dataset.data.clone()
    missing_mask = ~dataset.masks

    if has_imputable:
        can_impute_mask = missing_mask & (dataset.imputable == 1)
        new_data[can_impute_mask] = recon_real[can_impute_mask]
    else:
        new_data[missing_mask] = recon_real[missing_mask]

    # Create new dataset object
    new_dataset = copy.deepcopy(dataset)
    new_dataset.data = new_data
    new_dataset.indices = dataset.indices  # keep full index

    return new_dataset


def compute_val_mse(model, dataset, device="cpu", auto_fix_binary=False, eps: float = 1e-7):
    model.eval()

    # ------------------------
    # 0) Tensors
    # ------------------------
    X = dataset.data.to(device)
    C = dataset.cluster_labels.to(device)
    val_data = dataset.val_data.to(device)
    val_mask = ~torch.isnan(val_data)

    means = torch.as_tensor(dataset.feature_means, dtype=torch.float32, device=device)
    stds  = torch.as_tensor(dataset.feature_stds,  dtype=torch.float32, device=device)

    if (stds == 0).any():
        stds = stds.clone()
        stds[stds == 0] = 1.0

    # ------------------------
    # 1) Forward (LOGITS)
    # ------------------------
    with torch.no_grad():
        recon, _, _ = model.forward(X, C, deterministic=True)

    pred = recon.clone()

    # ------------------------
    # 2) Convert logits → usable values
    # ------------------------
    for name, cols in dataset.activation_groups.items():
        cols = torch.tensor(cols, device=device)

        if name == "continuous":
            pred[:, cols] = recon[:, cols] * stds[cols] + means[cols]

        elif name == "binary":
            pred[:, cols] = torch.sigmoid(recon[:, cols]).clamp(eps, 1 - eps)

        else:  # categorical
            probs = torch.softmax(recon[:, cols], dim=1)
            idx = torch.argmax(probs, dim=1)

            pred[:, cols] = 0
            pred[torch.arange(pred.shape[0]), cols[idx]] = 1

    # ------------------------
    # 3) Continuous MSE
    # ------------------------
    cont_cols = torch.tensor(dataset.activation_groups.get("continuous", []), device=device)

    if len(cont_cols) > 0:
        use_c = val_mask[:, cont_cols]
        se = (pred[:, cont_cols] - val_data[:, cont_cols]) ** 2
        mse = se[use_c].mean() if use_c.any() else pred.new_zeros(())
    else:
        mse = pred.new_zeros(())

    # ------------------------
    # 4) Binary BCE
    # ------------------------
    bin_cols = torch.tensor(dataset.activation_groups.get("binary", []), device=device)

    if len(bin_cols) > 0:
        target_b = val_data[:, bin_cols].clone()
        nan_mask = torch.isnan(target_b)
        if nan_mask.any():
            target_b[nan_mask] = 0.0

        use_b = val_mask[:, bin_cols]

        masked_targets = target_b[use_b]
        if masked_targets.numel():
            bad = (~torch.isfinite(masked_targets)) | (masked_targets < 0) | (masked_targets > 1)
            if bad.any():
                if not auto_fix_binary:
                    raise RuntimeError("Binary target(s) out of [0,1].")
                target_b[use_b] = (masked_targets > 0.5).float()

        prob = pred[:, bin_cols]
        bce_elem = F.binary_cross_entropy(prob, target_b, reduction="none")

        bmask = use_b.to(bce_elem.dtype)
        bce = (bce_elem * bmask).sum() / bmask.sum().clamp_min(1.0)
    else:
        bce = pred.new_zeros(())

    # ------------------------
    # 5) Categorical CE (NEW)
    # ------------------------
    ce = pred.new_zeros(())

    for name, cols in dataset.activation_groups.items():
        if name in ["continuous", "binary"]:
            continue

        cols = torch.tensor(cols, device=device)

        # valid rows: at least one observed entry in group
        mask_sub = val_mask[:, cols]
        valid_rows = mask_sub.sum(dim=1) > 0

        if valid_rows.any():
            logits = recon[valid_rows][:, cols]

            # convert one-hot target → class index
            target = torch.argmax(val_data[valid_rows][:, cols], dim=1)

            ce_loss = F.cross_entropy(logits, target, reduction="mean")
            ce += ce_loss

    # ------------------------
    # 6) Final metrics
    # ------------------------
    imputation_error = (mse + bce + ce).item()

    return imputation_error, mse.item(), bce.item(), ce.item()
    
    


def evaluate_imputation(imputed_df, df_complete, df_missing, activation_groups=None):
    """
    Test CISSVAE performance by evaluating imputed dataset vs true complete dataset. 

    Supports mixed data types:
    - continuous → MSE
    - binary → BCE-style squared error
    - categorical → classification error

    Returns overall error and detailed comparison dataframe.

    :param imputed_df: An imputed version of df_missing.
    :type imputed_df: pd.DataFrame()
    :param df_complete: A complete dataset with no missingness. 
    :type df_complete: pd.DataFrame()
    :param df_missing: A version of df_complete with induced missingness. 
    :type df_missing: pd.DataFrame()
    :param activation_groups: Dictionary mapping feature types to column indices. 
        Expected format:
            {
                "continuous": [int, ...],
                "binary": [int, ...],
                "<categorical_name>": [int, ...],
                ...
            }
        Each key defines a feature group, and values are lists of column indices 
        corresponding to that group. Categorical variables must be represented as 
        grouped indices (e.g., one-hot encoded columns belonging to the same variable).
    :type activation_groups: dict[str, list[int]]
    """

    # -------------------------
    # Validation
    # -------------------------
    if not (imputed_df.shape == df_complete.shape == df_missing.shape):
        raise ValueError("All input DataFrames must have the same shape.")

    if not (list(imputed_df.columns) == list(df_complete.columns) == list(df_missing.columns)):
        raise ValueError("All DataFrames must have identical columns in the same order.")

    # -------------------------
    # Missing mask
    # -------------------------
    missing_mask = df_missing.isna()

    if not missing_mask.values.any():
        return 0.0, pd.DataFrame()

    # -------------------------
    # Extract indices
    # -------------------------
    row_idx, col_idx = np.where(missing_mask.values)

    rows = df_missing.index[row_idx]
    cols = df_missing.columns[col_idx]

    true_vals = df_complete.values[row_idx, col_idx]
    imputed_vals = imputed_df.values[row_idx, col_idx]

    # -------------------------
    # Default: MSE
    # -------------------------
    errors = (true_vals - imputed_vals) ** 2

    # -------------------------
    # If activation_groups provided → fix per type
    # -------------------------
    if activation_groups is not None:
        col_to_type = {}

        for name, indices in activation_groups.items():
            for idx in indices:
                col_to_type[df_complete.columns[idx]] = name

        for i, col in enumerate(cols):
            col_type = col_to_type.get(col, "continuous")

            # -----------------
            # Binary → keep squared error (OK)
            # -----------------
            if col_type == "binary":
                continue

            # -----------------
            # Categorical → classification error
            # -----------------
            elif col_type not in ["continuous", "binary"]:
                errors[i] = float(true_vals[i] != imputed_vals[i])

    # -------------------------
    # Build comparison df
    # -------------------------
    comparison_df = pd.DataFrame({
        "row": rows,
        "col": cols,
        "true": true_vals,
        "imputed": imputed_vals,
        "error": errors
    })

    # -------------------------
    # Aggregate metrics
    # -------------------------
    mse = errors.mean()

    print(f"[INFO] Imputation error on missing entries: {mse:.6f}")

    return mse, comparison_df


# def get_val_comp_df(model, dataset, device="cpu"):
#     """Get model predictions, denormalize them, and return as DataFrame with cluster labels.
    
#     Runs the model on the full dataset to generate predictions, denormalizes the output
#     using the dataset's feature statistics, and returns the results as a pandas DataFrame
#     with cluster labels included.
    
#     :param model: Trained model in evaluation mode
#     :type model: nn.Module
#     :param dataset: Dataset containing normalized data and feature statistics
#     :type dataset: ClusterDataset
#     :param device: Device for computations, defaults to "cpu"
#     :type device: str, optional
#     :return: DataFrame containing denormalized predictions and cluster labels
#     :rtype: pandas.DataFrame
#     """
#     model.eval()
    
#     # Get inputs and labels
#     full_x = dataset.data.to(device)                       # (N, D), normalized
#     full_cluster = dataset.cluster_labels.to(device)       # (N,)
    
#     # Get model predictions
#     with torch.no_grad():
#         recon_x, _, _ = model(full_x, full_cluster, deterministic = True)        # (N, D), normalized output
    
#     # Retrieve per-feature stats for denormalization
#     means = torch.tensor(dataset.feature_means, dtype=torch.float32, device=device)  # (D,)
#     stds = torch.tensor(dataset.feature_stds, dtype=torch.float32, device=device)    # (D,)
    
#     # Denormalize model output
#     recon_x_denorm = recon_x * stds + means               # (N, D), denormalized
    
#     # Convert to numpy/CPU
#     predictions = recon_x_denorm.cpu().numpy()            # (N, D)
#     cluster_labels = full_cluster.cpu().numpy()           # (N,)
    
#     # Create DataFrame
#     # Assuming dataset has feature names, otherwise use generic names
#     if hasattr(dataset, 'feature_names') and dataset.feature_names is not None:
#         feature_names = dataset.feature_names
#     else:
#         feature_names = [f"feature_{i}" for i in range(predictions.shape[1])]
    
#     # Create DataFrame with predictions
#     df = pd.DataFrame(predictions, columns=feature_names)
    
#     # Add cluster labels as a column
#     df['cluster'] = cluster_labels
    
#     return df