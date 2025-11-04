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
return_fig = False):
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
    ax.set_title(title, fontsize=15, weight='bold')

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
    # Get scaled impued data
    # -------------------------------
    imputed = get_imputed(model, data_loader, device)
    x_all = imputed.data


    # -------------------------------
    # Unscale the imputed data (only for continuous vars)
    # -------------------------------
    # Binary vs continuous mask
    if(dataset.binary_feature_mask is not None):
        binary_feature_mask = torch.as_tensor(dataset.binary_feature_mask, dtype=torch.bool, device=device)
        cont_feat = ~binary_feature_mask  # continuous columns are True

        # Full mean and std tensors
        means = torch.as_tensor(imputed.feature_means, dtype=torch.float32, device=device)
        stds  = torch.as_tensor(imputed.feature_stds,  dtype=torch.float32, device=device)

        # Replace zero stds with 1.0 (avoid divide-by-zero or NaN)
        stds = stds.clone()
        stds[stds == 0] = 1.0

        # Clone imputed data (still normalized)
        x_all_denorm = x_all.clone()

        # --- Only denormalize continuous columns ---
        if cont_feat.any():
            cont_idx = torch.nonzero(cont_feat, as_tuple=False).squeeze(1)
            x_all_denorm[:, cont_idx] = x_all[:, cont_idx] * stds[cont_idx] + means[cont_idx]

        # --- Binary columns are already sigmoid probabilities ---
        if binary_feature_mask.any():
            bin_idx = torch.nonzero(binary_feature_mask, as_tuple=False).squeeze(1)
            # Clamp to [0,1] just to be safe
            x_all_denorm[:, bin_idx] = x_all[:, bin_idx]
    else:
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


    x_all_denorm[val_mask_tensor] = val_data_tensor[val_mask_tensor]
    
    # NEW 11SEP2025: Set imputable positions to NaN
    if hasattr(dataset, 'imputable') and dataset.imputable is not None:
        imputable_mask = dataset.imputable.to(x_all_denorm.device)
        # Set positions where imputable == 0 to NaN
        x_all_denorm[imputable_mask == 0] = float('nan')



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
            recon_batch, _, _ = model.forward(x_batch, cluster_batch)

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

    # Replace only missing values in a clone of the original data
    new_data = dataset.data.clone()
    missing_mask = ~dataset.masks  # True where values were missing

    ## NEW 11SEP2025 - imputable
    if has_imputable:
        # Only impute where: (1) value was missing AND (2) imputable allows it
        can_impute_mask = missing_mask & (dataset.imputable == 1)
        new_data[can_impute_mask] = recon_sorted[can_impute_mask]
    else:
        # Original behavior if no imputable mask
        new_data[missing_mask] = recon_sorted[missing_mask]

    # Create new dataset object
    new_dataset = copy.deepcopy(dataset)
    new_dataset.data = new_data
    new_dataset.indices = dataset.indices  # keep full index

    return new_dataset




def compute_val_mse(model, dataset, device="cpu", auto_fix_binary = False, eps: float = 1e-7):
    """Compute MSE on validation-masked entries using consistent model predictions.
    
    Evaluates model performance by computing mean squared error between model predictions
    and true values at validation-masked positions. The model output is denormalized
    before comparison with the original validation data.
    
    :param model: Trained model in evaluation mode
    :type model: nn.Module
    :param dataset: Dataset containing validation masks and true values
    :type dataset: ClusterDataset
    :param device: Device for computations, defaults to "cpu"
    :type device: str, optional
    :return: Total Imputation Error, Validation MSE and Validation BCE on validation dataset
    :rtype: tuple(float)
    :raises ValueError: If no validation entries are found in the dataset
    """
    model.eval()

    # ------------------------
    # 0) Tensors on device
    # ------------------------
    X = dataset.data.to(device)                 # (N, D) normalized
    C = dataset.cluster_labels.to(device)       # (N,)
    val_data = dataset.val_data.to(device)      # (N, D) original scale (binary cols should be 0/1 where valid)
    val_mask = ~torch.isnan(val_data)           # (N, D) bool

    means = torch.as_tensor(dataset.feature_means, dtype=torch.float32, device=device)  # (D,)
    stds  = torch.as_tensor(dataset.feature_stds,  dtype=torch.float32, device=device)  # (D,)
    if (stds == 0).any():
        stds = stds.clone()
        stds[stds == 0] = 1.0

    # Column masks
    if getattr(dataset, "binary_feature_mask", None) is None:
        bin_1d = torch.zeros(X.shape[1], dtype=torch.bool, device=device)
    else:
        bin_1d = torch.as_tensor(dataset.binary_feature_mask, dtype=torch.bool, device=device)
    cont_1d = ~bin_1d

    # Expand to (N, D) for weighting
    cont_2d = cont_1d.unsqueeze(0).expand_as(val_mask)   # (N, D)
    bin_2d  = bin_1d.unsqueeze(0).expand_as(val_mask)    # (N, D)

    # Masks we will use to weight losses
    use_c = (val_mask & cont_2d)                         # (N, D) continuous validation entries
    use_b = (val_mask & bin_2d)                          # (N, D) binary    validation entries

    # ------------------------
    # 1) Forward pass
    # ------------------------
    with torch.no_grad():
        recon, _, _ = model(X, C)                        # (N, D)

        # Build "predictions in evaluation space":
        #   - Continuous: denormalize
        #   - Binary: keep probabilities; clamp to avoid log(0)
        pred = recon.clone()

        if cont_1d.any():
            ccols = torch.nonzero(cont_1d, as_tuple=False).squeeze(1)
            pred[:, ccols] = recon[:, ccols] * stds[ccols] + means[ccols]

        if bin_1d.any():
            bcols = torch.nonzero(bin_1d, as_tuple=False).squeeze(1)
            pred[:, bcols] = pred[:, bcols].clamp_(eps, 1 - eps)  # probs

        # ------------------------
        # 2) Continuous: elementwise MSE, mask & normalize
        # ------------------------
        se = (pred - val_data).pow(2)                         # (N, D)
        if model.debug:
            np.savetxt("predicted_vals.csv", pred.numpy(), delimiter=",")
            np.savetxt("valdata.csv", val_data.numpy(), delimiter=",")
            np.savetxt("se.csv", se.numpy(), delimiter=",")

        

        use_c = val_mask & cont_2d                            # (N, D)
        mse = se[use_c].mean() if use_c.any() else pred.new_zeros(())
        if(model.debug):
            print(f"Masked Error: {min(se[use_c])}, {max(se[use_c])} \n")
            print(f"MSE: {mse}\n\n")
            mse_elem = F.mse_loss(pred, val_data, reduction="none")
            print(
                f"MSE_elem: min={mse_elem[use_c].min().item():.6g}, "
                f"max={mse_elem[use_c].max().item():.6g} | "
                f"MSE_mean(masked)={(mse_elem[use_c].mean().item() if use_c.any() else float('nan')):.6g}"
            )

        # ------------------------
        # 3) Binary: elementwise BCE, mask & normalize
        # ------------------------
        if bin_1d.any():
            # For BCE validation to pass, targets must be finite and in [0,1] EVERYWHERE
            # we pass to BCE. We’ll create a "filled" target where non-validation (or NaN)
            # entries are replaced by a safe dummy in [0,1], e.g., 0.0.
            target_b = val_data.clone()                            # (N, D)
            # Set non-binary columns to 0 (they'll be masked out anyway)
            target_b[:, ~bin_1d] = 0.0
            # Replace NaNs with 0.0 on binary columns
            nan_mask = torch.isnan(target_b) & bin_2d
            if nan_mask.any():
                target_b[nan_mask] = 0.0

            # Optional strict check on *masked* binary entries only
            masked_targets = target_b[use_b]
            if masked_targets.numel():
                bad = (~torch.isfinite(masked_targets)) | (masked_targets < 0) | (masked_targets > 1)
                if bad.any():
                    if not auto_fix_binary:
                        raise RuntimeError("Binary target(s) out of [0,1] under validation mask.")
                    # Coerce masked offending targets to {0,1} via threshold
                    masked_targets = (masked_targets > 0.5).to(masked_targets.dtype)
                    target_b[use_b] = masked_targets

            # Now it's safe to compute BCE elementwise over the whole matrix.
            # Non-validation entries have valid targets (0.0), but we will weight them by 0.
            # Use only the binary columns for BCE computation.
            prob_full   = pred[:, bin_1d]                          # (N, Db)
            target_full = target_b[:, bin_1d]                      # (N, Db)
            bce_elem = F.binary_cross_entropy(prob_full, target_full, reduction='none')  # (N, Db)


            # Weight by validation mask on the same columns
            bmask_full = use_b[:, bin_1d].to(bce_elem.dtype)       # (N, Db)
            bce_sum = (bce_elem * bmask_full).sum()
            bce_den = bmask_full.sum().clamp_min(1.0)
            bce = bce_sum / bce_den
            if model.debug:
                # use the (N, Db) mask we already built
                masked_bce = bce_elem.masked_select(bmask_full.bool())
                if masked_bce.numel() > 0:
                    print(f"BCE Elems: min = {masked_bce.min().item()}, max = {masked_bce.max().item()}\n")
                else:
                    print("BCE Elems: (no validated binary entries)\n")
                print(f"BCE: {bce.item()}\n")

        else:
            bce = pred.new_zeros(())

        if(model.debug):
            print(f"BCE: bce{bce}\n\n MSE: {mse}\n\n")

        # ------------------------
        # 4) Return combined metric
        # ------------------------
        imputation_error = (mse + bce).item()
        val_mse = mse.item()
        val_bce = bce.item()
    
        return imputation_error, val_mse, val_bce

    # model.eval()

    # # Get normalized inputs and val data
    # full_x = dataset.data.to(device)                       # (N, D), normalized
    # full_cluster = dataset.cluster_labels.to(device)       # (N,)
    # val_data = dataset.val_data.to(device)                 # (N, D), not normalized
    # val_mask = ~torch.isnan(val_data)                      # (N, D)

    # with torch.no_grad():
    #     recon_x, _, _ = model(full_x, full_cluster)        # normalized output

    # # Retrieve per-feature stats
    # means = torch.tensor(dataset.feature_means, dtype=torch.float32, device=device)
    # stds = torch.tensor(dataset.feature_stds, dtype=torch.float32, device=device)

    #     # Check for zero std features
    # zero_std_idx = torch.where(stds == 0)[0]
    # if zero_std_idx.numel() > 0:
    #     bad_feats = [dataset.feature_names[i] for i in zero_std_idx.tolist()]
    #     print(
    #         f"[Warning] {len(bad_feats)} feature(s) have std == 0. "
    #         f"Replaced with 1.0. Features: {bad_feats}"
    #     )
    #     stds[zero_std_idx] = 1.0  # safe replacement

    # ## ---------------------------
    # ## 15 OCT 2025 - Added handling for binary features
    # ## ---------------------------

    # # Denormalize model output
    # if dataset.binary_feature_mask is None:
    #     recon_x_denorm = recon_x * stds + means
    #         ## Masked validation entries should not include the imputable_mask
    #     squared_error = (recon_x_denorm - val_data) ** 2
    #     masked_error = squared_error[val_mask]

    #     if getattr(model, "debug", True):
    #         print("[DEBUG] recon_x_denorm (first 2 rows):")
    #         print(recon_x_denorm[:2])

    #     if getattr(model, "debug", True):
    #         print(f"[DEBUG] squared_error (first 2 rows): {squared_error[:2]}")
    #         print(f"[DEBUG] val_mask (first 2 rows): {val_mask[:2]}")
    #         print(f"[DEBUG] masked_error stats: count={masked_error.numel()}, mean={masked_error.mean().item():.4f}")

    #     if masked_error.numel() == 0:
    #         raise ValueError("No validation entries found. Increase `val_proportion` in ClusterDataset.")
    #     return masked_error.mean().item()
    # else:
    #     recon_x_cont = recon_x * stds + means
    #     recon_x_denorm = recon_x * dataset.binary_feature_mask + recon_x_cont * ~dataset.binary_feature_mask
    #     if getattr(model, "debug", True):
    #         print(f"[DEBUG]: Recon_x_cont: \n{recon_x_cont}\n\n Recon_x_denorm \n{recon_x_denorm}\n\n")
    #         ## Err for continuous data
    #     squared_error = (recon_x_denorm *  ~dataset.binary_feature_mask - val_data *  ~dataset.binary_feature_mask) ** 2
    #     masked_error = squared_error[val_mask *  ~dataset.binary_feature_mask]
        
    #     y_true = val_data * dataset.binary_feature_mask
    #     y_pred = recon_x_denorm * val_mask * dataset.binary_feature_mask
    
    #     bce_mat =  -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    #     total_error = masked_error + bce_mat
    #     if getattr(model, "debug", True):
    #         print(f"[DEBUG]: masked_err_continuous: \n{masked_error}\n\n BCE matrix \n{bce_mat}\n\n")
    #         print(f"[DEBUG]: total error: \n{total_error}\n\n")
    #         ## err for binary data

    #     return(np.mean(total_error))




    
    



import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

def evaluate_imputation(imputed_df, df_complete, df_missing):
    """Compare imputed values to true values at originally missing positions.
    
    Calculates mean squared error between imputed and true values specifically at locations
    that were originally missing, providing a detailed comparison DataFrame for analysis.
    
    :param imputed_df: DataFrame with imputed values (same shape as df_complete)
    :type imputed_df: pandas.DataFrame
    :param df_complete: DataFrame with true values (ground truth, no NaNs)
    :type df_complete: pandas.DataFrame
    :param df_missing: DataFrame with NaNs at original missing locations
    :type df_missing: pandas.DataFrame
    :return: Tuple containing MSE value and detailed comparison DataFrame
    :rtype: tuple[float, pandas.DataFrame]
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


def get_val_comp_df(model, dataset, device="cpu"):
    """Get model predictions, denormalize them, and return as DataFrame with cluster labels.
    
    Runs the model on the full dataset to generate predictions, denormalizes the output
    using the dataset's feature statistics, and returns the results as a pandas DataFrame
    with cluster labels included.
    
    :param model: Trained model in evaluation mode
    :type model: nn.Module
    :param dataset: Dataset containing normalized data and feature statistics
    :type dataset: ClusterDataset
    :param device: Device for computations, defaults to "cpu"
    :type device: str, optional
    :return: DataFrame containing denormalized predictions and cluster labels
    :rtype: pandas.DataFrame
    """
    model.eval()
    
    # Get inputs and labels
    full_x = dataset.data.to(device)                       # (N, D), normalized
    full_cluster = dataset.cluster_labels.to(device)       # (N,)
    
    # Get model predictions
    with torch.no_grad():
        recon_x, _, _ = model(full_x, full_cluster)        # (N, D), normalized output
    
    # Retrieve per-feature stats for denormalization
    means = torch.tensor(dataset.feature_means, dtype=torch.float32, device=device)  # (D,)
    stds = torch.tensor(dataset.feature_stds, dtype=torch.float32, device=device)    # (D,)
    
    # Denormalize model output
    recon_x_denorm = recon_x * stds + means               # (N, D), denormalized
    
    # Convert to numpy/CPU
    predictions = recon_x_denorm.cpu().numpy()            # (N, D)
    cluster_labels = full_cluster.cpu().numpy()           # (N,)
    
    # Create DataFrame
    # Assuming dataset has feature names, otherwise use generic names
    if hasattr(dataset, 'feature_names') and dataset.feature_names is not None:
        feature_names = dataset.feature_names
    else:
        feature_names = [f"feature_{i}" for i in range(predictions.shape[1])]
    
    # Create DataFrame with predictions
    df = pd.DataFrame(predictions, columns=feature_names)
    
    # Add cluster labels as a column
    df['cluster'] = cluster_labels
    
    return df