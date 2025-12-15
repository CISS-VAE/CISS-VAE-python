import torch
import pandas as pd
import torch.optim as optim
from torch.optim import lr_scheduler
from ciss_vae.utils.loss import loss_function
import torch.nn.functional as F
from ciss_vae.utils.helpers import compute_val_mse, get_imputed_df

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
    return_history: bool = False,
    progress_callback = None,
    weight_decay = 0.001,
):
    """Train a VAE on masked data with validation monitoring for initial training phase.
    
    Performs the initial training of a CISSVAE model on data with missing values using
    masked loss computation. Tracks training loss and validation MSE across epochs,
    with optional progress reporting and learning rate decay.
    
    :param model: CISSVAE or compatible VAE model that implements forward(x, cluster_id)
    :type model: torch.nn.Module
    :param train_loader: DataLoader built on ClusterDataset containing validation data
    :type train_loader: torch.utils.data.DataLoader
    :param epochs: Number of training epochs, defaults to 10
    :type epochs: int, optional
    :param initial_lr: Starting learning rate for Adam optimizer, defaults to 0.01
    :type initial_lr: float, optional
    :param decay_factor: Exponential learning rate decay factor applied per epoch, defaults to 0.999
    :type decay_factor: float, optional
    :param beta: Weight coefficient for KL divergence term in VAE loss, defaults to 0.1
    :type beta: float, optional
    :param device: Device for training computations ("cpu" or "cuda"), defaults to "cpu"
    :type device: str, optional
    :param verbose: Whether to print per-epoch training metrics, defaults to False
    :type verbose: bool, optional
    :param return_history: Whether to return training history DataFrame along with model, defaults to False
    :type return_history: bool, optional
    :param progress_callback: Optional callback function for progress reporting, defaults to None
    :type progress_callback: callable, optional
    :return: Trained model, or tuple of (model, history_dataframe) if return_history=True
    :rtype: torch.nn.Module or tuple[torch.nn.Module, pandas.DataFrame]
    :raises ValueError: If dataset does not contain 'val_data' attribute for validation
    """

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=initial_lr, weight_decay = weight_decay)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=decay_factor)

    # Pull dataset object from loader to get validation targets
    dataset = train_loader.dataset
    if not hasattr(dataset, "val_data"):
        raise ValueError("Dataset must include 'val_data' for validation-based early stopping.")

    def _to_scalar(x):
        """Convert torch tensors to Python scalars safely."""
        if torch.is_tensor(x):
            return x.detach().cpu().item()
        return x


    # Container to collect per-epoch metrics
    history = {
        "epoch": [],
        "train_loss": [],   # average per-sample loss across the dataset
        "train_mse": [],
        "train_bce": [],
        "imputation_error": [],
        "val_mse": [],      # validation MSE computed on validation-held positions
        "val_bce":[],
        "lr": [],           # learning rate at epoch end
    }

    n_samples = len(train_loader.dataset)


    for epoch in range(epochs):
        model.train()
        total_loss = 0


        for batch in train_loader:
            # print(f"Batch is: {len(batch)}\n")
            x_batch, cluster_batch, mask_batch, idx_batch= batch
            x_batch = x_batch.to(device)
            cluster_batch = cluster_batch.to(device)
            mask_batch = mask_batch.to(device)
            
            if hasattr(dataset, 'imputable') and dataset.imputable is not None:
                imputable_batch = dataset.imputable[idx_batch].to(device).float()
            else:
                imputable_batch = None

            recon_x, mu, logvar = model(x_batch, cluster_batch)

            loss, train_mse, train_bce  = loss_function(
                cluster_batch, mask_batch, recon_x, x_batch, dataset.binary_feature_mask, mu, logvar,
                beta=beta,
                return_components=True,
                device = device,
                debug = model.debug
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
            imputation_error, val_mse, val_bce = compute_val_mse(model, train_loader.dataset, device)
        except ValueError as e:
            if verbose:
                print(f"[WARNING] Epoch {epoch+1}: {e}")
            val_mse = float("inf")

        # -----------------------------------
        # Logging to history
        # -----------------------------------

        

        history["epoch"].append(epoch)
        history["train_loss"].append(avg_train_loss)
        history["train_mse"].append(_to_scalar(train_mse))
        history["train_bce"].append(_to_scalar(train_bce))
        history["imputation_error"].append(_to_scalar(imputation_error))
        history["val_mse"].append(_to_scalar(val_mse))
        history["val_bce"].append(_to_scalar(val_bce))

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
        if progress_callback is not None:
            try:
                progress_callback(n=1)
            except Exception as e:
                if verbose:
                    print(f"Progress callback error: {e}")
        scheduler.step()

    model.set_final_lr(optimizer.param_groups[0]["lr"])

    # Build a DataFrame and attach to the model
    history_df = pd.DataFrame(history, columns=["epoch", "train_loss", "train_mse", "train_bce", "imputation_error", "val_mse", "val_bce", "lr"])
    model.training_history_ = history_df


    return (model, history_df) if return_history else model
