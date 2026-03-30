import torch
import torch.nn.functional as F
import warnings ## let's see if/when my reconX goes nan
import numpy as np

def loss_function(
    cluster,
    mask,
    recon_x,
    x,
    activation_groups,   # <-- NEW
    mu,
    logvar,
    beta=0.001,
    return_components=False,
    imputable_mask=None,
    device="cpu",
    debug=False,
):
    if torch.isnan(recon_x).any():
        warnings.warn("recon_x contains NaNs", RuntimeWarning)

    if torch.isnan(x).any():
        warnings.warn("x contains NaNs", RuntimeWarning)

    total_recon_loss = 0.0
    mse_loss = 0.0
    bce_loss = 0.0
    ce_loss = 0.0

    # ----------------------------------
    # Loop over activation groups
    # ----------------------------------
    for name, cols in activation_groups.items():

        cols = torch.as_tensor(cols, dtype=torch.long, device=device)

        x_sub = x[:, cols]
        recon_sub = recon_x[:, cols]
        mask_sub = mask[:, cols]

        # ------------------------------
        # CONTINUOUS → MSE
        # ------------------------------
        if name == "continuous":
            loss = F.mse_loss(
                recon_sub * mask_sub,
                x_sub * mask_sub,
                reduction="sum",
            )
            mse_loss += loss
            total_recon_loss += loss

        # ------------------------------
        # BINARY → BCE
        # ------------------------------
        elif name == "binary":
            loss = F.binary_cross_entropy_with_logits(
                recon_sub * mask_sub,
                x_sub * mask_sub,
                reduction="sum",
            )
            bce_loss += loss
            total_recon_loss += loss

        # ------------------------------
        # CATEGORICAL → CrossEntropy
        # ------------------------------
        else:
            # IMPORTANT:
            # recon_sub must be logits (NO softmax)
            # target must be class index

            # mask rows where any column is observed
            valid_rows = mask_sub.sum(dim=1) > 0

            if valid_rows.any():
                logits = recon_sub[valid_rows]

                # convert one-hot → class index
                target = torch.argmax(x_sub[valid_rows], dim=1)

                loss = F.cross_entropy(logits, target, reduction="sum")
                ce_loss += loss
                total_recon_loss += loss

    # ------------------------------
    # KL loss
    # ------------------------------
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    total_loss = total_recon_loss + beta * kl_loss

    if return_components:
        return total_loss, mse_loss, bce_loss, ce_loss

    return total_loss


def loss_function_nomask(
    cluster,
    recon_x,
    x,
    activation_groups,   # <-- NEW
    mu,
    logvar,
    beta=0.001,
    return_components=False,
    imputable_mask=None,
    device="cpu",
    debug=False,
):
    if torch.isnan(recon_x).any():
        warnings.warn("[Warning] recon_x contains NaN values", RuntimeWarning)

    if torch.isnan(x).any():
        warnings.warn("[Warning] x contains NaN values", RuntimeWarning)

    total_recon_loss = 0.0
    mse_loss = 0.0
    bce_loss = 0.0
    ce_loss = 0.0

    # ----------------------------------
    # Loop over activation groups
    # ----------------------------------
    for name, cols in activation_groups.items():

        cols = torch.as_tensor(cols, dtype=torch.long, device=device)

        x_sub = x[:, cols]
        recon_sub = recon_x[:, cols]

        # Apply imputable mask if present
        if imputable_mask is not None:
            mask_sub = imputable_mask[:, cols]
            x_sub = x_sub * mask_sub
            recon_sub = recon_sub * mask_sub

        # ------------------------------
        # CONTINUOUS → MSE
        # ------------------------------
        if name == "continuous":
            loss = F.mse_loss(recon_sub, x_sub, reduction="sum")
            mse_loss += loss
            total_recon_loss += loss

        # ------------------------------
        # BINARY → BCE
        # ------------------------------
        elif name == "binary":
            loss = F.binary_cross_entropy_with_logits(recon_sub, x_sub, reduction="sum")
            bce_loss += loss
            total_recon_loss += loss

        # ------------------------------
        # CATEGORICAL → CrossEntropy
        # ------------------------------
        else:
            # IMPORTANT:
            # recon_sub must be logits (NO softmax)
            # CrossEntropyLoss expects logits + class indices

            # Filter rows with valid signal (important if imputable_mask used)
            if imputable_mask is not None:
                valid_rows = mask_sub.sum(dim=1) > 0
            else:
                valid_rows = torch.ones(x_sub.shape[0], dtype=torch.bool, device=device)

            if valid_rows.any():
                logits = recon_sub[valid_rows]

                # Convert one-hot → class index
                target = torch.argmax(x_sub[valid_rows], dim=1)

                loss = F.cross_entropy(logits, target, reduction="sum")
                ce_loss += loss
                total_recon_loss += loss

    # ------------------------------
    # KL divergence
    # ------------------------------
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    total_loss = total_recon_loss + beta * kl_loss

    if return_components:
        return total_loss, mse_loss, bce_loss, ce_loss

    return total_loss