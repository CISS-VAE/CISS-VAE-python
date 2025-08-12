import torch
import pandas as pd
import torch.optim as optim
from torch.optim import lr_scheduler
from ciss_vae.utils.loss import loss_function
import torch.nn.functional as F
from ciss_vae.utils.helpers import compute_val_mse

def train_vae_initial(
    model,
    train_loader,
    epochs: int = 10,
    initial_lr: float = 0.01,
    decay_factor: float = 0.999,
    beta: float = 0.1,
    device: str = "cpu",
    verbose: bool = False,
    *,
    return_history: bool = True,
    progress_callback = None,
):
    """
    Train a VAE on masked data with validation set aside inside the ClusterDataset.

    Each sample from `train_loader` must yield a tuple of:
        (x, cluster_id, mask, index)

    Parameters
    ----------
    model : torch.nn.Module
        Your CISSVAE (or compatible) model. Must implement forward(x, cluster_id).
    train_loader : torch.utils.data.DataLoader
        DataLoader built on a ClusterDataset that contains `.val_data`.
    epochs : int, default=10
        Number of epochs for the initial training pass.
    initial_lr : float, default=0.01
        Starting learning rate for Adam.
    decay_factor : float, default=0.999
        ExponentialLR decay gamma applied at the end of each epoch.
    beta : float, default=0.1
        Weight of the KL term in the VAE loss.
    device : {"cpu","cuda"}, default="cpu"
        Torch device to run training on.
    verbose : bool, default=False
        If True, prints per-epoch metrics (also stored in the history DataFrame).
    return_history : bool, default=True
        If True, returns a tuple (model, history_df). Otherwise returns just `model`.
        In both cases, the DataFrame is attached as `model.training_history_`.

    Returns
    -------
    model : torch.nn.Module
        The trained model (always).
    history_df : pd.DataFrame, optional
        Returned only when `return_history=True`. Has columns:
            ["epoch", "train_loss", "train_recon", "train_kl", "val_mse", "lr"]

    Notes
    -----
    - The function expects `loss_function(...)` and `compute_val_mse(model, dataset, device)`
      to be available in scope.
    - For convenience, the history DataFrame is also attached as `model.training_history_`.
    """

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=initial_lr)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=decay_factor)

    # Pull dataset object from loader to get validation targets
    dataset = train_loader.dataset
    if not hasattr(dataset, "val_data"):
        raise ValueError("Dataset must include 'val_data' for validation-based early stopping.")

    # Container to collect per-epoch metrics
    history = {
        "epoch": [],
        "train_loss": [],   # average per-sample loss across the dataset
        "val_mse": [],      # validation MSE computed on validation-held positions
        "lr": [],           # learning rate at epoch end
    }

    n_samples = len(train_loader.dataset)


    for epoch in range(epochs):
        model.train()
        total_loss = 0


        for batch in train_loader:
            x_batch, cluster_batch, mask_batch, _ = batch
            x_batch = x_batch.to(device)
            cluster_batch = cluster_batch.to(device)
            mask_batch = mask_batch.to(device)

            recon_x, mu, logvar = model(x_batch, cluster_batch)

            loss, recon_loss, kl_loss = loss_function(
                cluster_batch, mask_batch, recon_x, x_batch, mu, logvar,
                beta=beta,
                return_components=True
            )


            ## Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader.dataset)

        ## Learning rate BEFORE stepping scheduler (aka for current epoch)
        current_lr = optimizer.param_groups[0]["lr"]

        # -----------------------------------
        # Validation MSE on val_data entries
        # -----------------------------------
        try:
            val_mse = compute_val_mse(model, dataset, device)
        except ValueError as e:
            if verbose:
                print(f"[WARNING] Epoch {epoch+1}: {e}")
            val_mse = float("inf")

        # -----------------------------------
        # Logging to history
        # -----------------------------------

        history["epoch"].append(epoch)
        history["train_loss"].append(avg_train_loss)
        history["val_mse"].append(val_mse)
        history["lr"].append(current_lr)

        if verbose:
            print(
                f"Epoch {epoch:3d} | "
                f"Train Loss: {avg_train_loss:.6f} | "
                f"Val MSE: {val_mse:.6f} | LR: {current_lr:.6f}"
            )

        #-----------------------
        # Hook for progress bar
        #-----------------------
        if progress_callback:
            progress_callback(1)   # tick one epoch
        scheduler.step()

    model.set_final_lr(optimizer.param_groups[0]["lr"])

    # Build a DataFrame and attach to the model
    history_df = pd.DataFrame(history, columns=["epoch", "train_loss", "val_mse", "lr"])
    model.training_history_ = history_df

    return (model, history_df) if return_history else model
