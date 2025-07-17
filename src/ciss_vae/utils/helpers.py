import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import torch.nn as nn
import torch
import pandas as pd
from ciss_vae.classes.vae import CISSVAE
import copy

def plot_vae_architecture(model: nn.Module, 
title = None, 
color_shared = "skyblue", 
color_unshared ="lightcoral",
color_latent = "gold",
color_input = "lightgreen",
color_output = "lightgreen",
figsize=(16, 8)):
    """
    Plots a horizontal schematic of the VAE architecture, showing shared and cluster-specific layers.

    Parameters:
    - model: An instance of CISSVAE.
    - title: Title of plot
    - color_shared: Color of shared hidden layers
    - color_unshared: Color of unshared hidden layers
    - color_input: Color of input layer
    - color_output: Color of output layer
    - figsize: Tuple, size of the matplotlib figure.
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
        ax.text(x0, y0, text, fontsize=9, ha='center', va='center', weight='bold')

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
                label, fontsize=11, ha='center', weight='bold')

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
        draw_box((x, y_base), f"Latent\nμ/σ²\n{latent_dim}", style=style, color=color_latent)
        # Arrow from last encoder layer
        draw_arrow((x - x_gap + box_width / 2, y_base), (x - box_width / 2, y_base))
    else:
        for c in range(n_clusters):
            y_c = y_base + (c - (n_clusters - 1) / 2) * cluster_gap
            draw_box((x, y_c), f"Latent\nC{c}\nμ/σ²\n{latent_dim}", style=style, color=color_latent)
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
    ax.set_title(title, fontsize=15, weight='bold')
    plt.tight_layout()
    plt.show()




def get_imputed_df(model: CISSVAE, data_loader, device = "cpu"):
    """Given trained model and cluster dataset object, get imputed dataset.
    Args:
        model: Trained CISSVAE model (should be in eval() mode)
        data_loader: DataLoader for the original ClusterDataset 
    Returns:
        imputed_df: pandas DataFrame of imputed (unscaled) data
    """
    dataset = data_loader.dataset
    # -------------------------------
    # Get scaled impued data
    # -------------------------------
    imputed = get_imputed(model, data_loader, device)
    x_all = imputed.data


    # -------------------------------
    # Unscale the imputed data
    # -------------------------------
        
    means = torch.tensor(imputed.feature_means, dtype=torch.float32)
    stds = torch.tensor(imputed.feature_stds, dtype=torch.float32)
    # Denormalize imputed values
    x_all_denorm = x_all * stds + means

    # -------------------------------------
    # Replace validation-masked entries with true values
    # -------------------------------------

     # Ensure val_data and val_mask are on the same device as x_all_denorm
    val_data_tensor = dataset.val_data.to(x_all_denorm.device)
    val_mask_tensor = ~torch.isnan(val_data_tensor)

    # Overwrite imputed values with ground truth at validation positions
    x_all_denorm[val_mask_tensor] = val_data_tensor[val_mask_tensor]

    x_all_denorm_np = x_all_denorm.detach().cpu().numpy()  # Now safe to convert


    # -------------------------------------
    # Create final DataFrame
    # -------------------------------------
    feature_names = getattr(dataset, "feature_names", [f"V{i}" for i in range(x_all.shape[1])])

    # The imputed dataset should be in the right order os this should be unecessary
    # # Recover original index from dataset.indices
    # if hasattr(dataset, "indices"):
    #     base_index = dataset.indices
    #     if isinstance(base_index, torch.Tensor):
    #         base_index = base_index.cpu().numpy()
    #     full_index = base_index[idx_all_np]
    # else:
    #     full_index = idx_all_np

    # Build DataFrame and sort to match original row order
    df_unsorted = pd.DataFrame(x_all_denorm_np, columns=feature_names, index=dataset.indices.cpu().numpy())
    imputed_df = df_unsorted.sort_index()

    return imputed_df

    

def get_imputed(model, data_loader, device="cpu"):
    """
    Returns a ClusterDataset where originally missing values have been replaced
    with the model's reconstructed outputs. Validation-masked values are also filled.

    Parameters:
    - model: trained VAE model
    - data_loader: DataLoader for the original ClusterDataset
    - device: torch device ("cpu" or "cuda")

    Returns:
    - ClusterDataset with reconstructed values filled in
    """
    model.eval()
    dataset = data_loader.dataset

    # Collect all batches
    all_recon = []
    all_masks = []
    all_indices = []

    with torch.no_grad():
        for batch in data_loader:
            x_batch, cluster_batch, mask_batch, idx_batch = batch

            x_batch = x_batch.to(device)
            cluster_batch = cluster_batch.to(device)

            # Predict full reconstruction
            recon_batch, _, _ = model.forward(x_batch, cluster_batch)

            all_recon.append(recon_batch.cpu())
            all_masks.append(mask_batch.cpu())
            all_indices.append(idx_batch)

    # Concatenate all batches
    recon_all = torch.cat(all_recon, dim=0)
    mask_all = torch.cat(all_masks, dim=0)
    idx_all = torch.cat(all_indices, dim=0)

    # Restore correct row order
    recon_sorted = torch.zeros_like(dataset.data)
    recon_sorted[idx_all] = recon_all

    # Replace only missing values in a clone of the original data
    new_data = dataset.data.clone()
    missing_mask = ~dataset.masks  # True where values were missing
    new_data[missing_mask] = recon_sorted[missing_mask]

    # Create new dataset object
    new_dataset = copy.deepcopy(dataset)
    new_dataset.data = new_data
    new_dataset.indices = dataset.indices  # keep full index
    return new_dataset




def compute_val_mse(model, dataset, device="cpu"):
    """
    Compute MSE on validation-masked entries using consistent model predictions.
    """
    model.eval()

    # Get normalized inputs and val data
    full_x = dataset.data.to(device)                       # (N, D), normalized
    full_cluster = dataset.cluster_labels.to(device)       # (N,)
    val_data = dataset.val_data.to(device)                 # (N, D), not normalized
    val_mask = ~torch.isnan(val_data)                      # (N, D)

    with torch.no_grad():
        recon_x, _, _ = model(full_x, full_cluster)        # normalized output

    # Retrieve per-feature stats
    means = torch.tensor(dataset.feature_means, dtype=torch.float32, device=device)
    stds = torch.tensor(dataset.feature_stds, dtype=torch.float32, device=device)

    # Denormalize model output
    recon_x_denorm = recon_x * stds + means

    if getattr(model, "debug", False):
        print("[DEBUG] recon_x_denorm (first 2 rows):")
        print(recon_x_denorm[:2])

    # Only compute error on masked validation entries
    squared_error = (recon_x_denorm - val_data) ** 2
    masked_error = squared_error[val_mask]

    if getattr(model, "debug", False):
        print(f"[DEBUG] squared_error (first 2 rows): {squared_error[:2]}")
        print(f"[DEBUG] val_mask (first 2 rows): {val_mask[:2]}")
        print(f"[DEBUG] masked_error stats: count={masked_error.numel()}, mean={masked_error.mean().item():.4f}")

    if masked_error.numel() == 0:
        raise ValueError("No validation entries found. Increase `val_percent` in ClusterDataset.")

    return masked_error.mean().item()
    



import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

def evaluate_imputation(imputed_df, df_complete, df_missing):
    """
    Compare imputed values to true values at originally missing positions.

    Parameters:
    - imputed_df: DataFrame with imputed values (same shape as df_complete)
    - df_complete: DataFrame with true values (ground truth, no NaNs)
    - df_missing: DataFrame with NaNs at original missing locations

    Returns:
    - mse: Mean squared error at imputed (originally missing) positions
    - comparison_df: DataFrame with row, column, true, imputed, and squared error
    """
    # Boolean DataFrame: True where value was originally missing
    missing_mask = df_missing.isna()

    # Get row, col indices of missing entries
    row_idx, col_idx = np.where(missing_mask.values)

    # Lookup corresponding values
    rows = df_missing.index[row_idx]
    cols = df_missing.columns[col_idx]

    true_vals = df_complete.values[row_idx, col_idx]
    imputed_vals = imputed_df.values[row_idx, col_idx]
    errors = (true_vals - imputed_vals) ** 2

    comparison_df = pd.DataFrame({
        "row": rows,
        "col": cols,
        "true": true_vals,
        "imputed": imputed_vals,
        "squared_error": errors
    })

    mse = errors.mean()
    print(f"[INFO] MSE on originally missing entries: {mse:.6f}")
    return mse, comparison_df
