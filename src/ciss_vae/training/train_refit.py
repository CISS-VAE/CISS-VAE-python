import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam, lr_scheduler
import pandas as pd
import numpy as np
from ciss_vae.utils.loss import loss_function, loss_function_nomask
from ciss_vae.classes.cluster_dataset import ClusterDataset
from torch.utils.data import DataLoader
from ciss_vae.utils.helpers import get_imputed_df, get_imputed, compute_val_mse
import copy


def train_vae_refit(model, 
    imputed_data, 
    epochs=10, 
    initial_lr=0.01,
    decay_factor=0.999, 
    beta=0.1,
    device="cpu", 
    verbose=False, 
    progress_callback = None,
    weight_decay = 0.001):
    """Train the VAE model on imputed data without masking for one refit iteration.
    
    Performs training on the complete imputed dataset.
    
    :param model: VAE model to train
    :type model: torch.nn.Module
    :param imputed_data: DataLoader containing imputed dataset with complete values
    :type imputed_data: torch.utils.data.DataLoader
    :param epochs: Number of training epochs, defaults to 10
    :type epochs: int, optional
    :param initial_lr: Initial learning rate for the optimizer, defaults to 0.01
    :type initial_lr: float, optional
    :param decay_factor: Exponential decay factor for learning rate scheduler, defaults to 0.999
    :type decay_factor: float, optional
    :param beta: Weight for KL divergence term in loss function, defaults to 0.1
    :type beta: float, optional
    :param device: Device to run training on, defaults to "cpu"
    :type device: str, optional
    :param verbose: Whether to print training progress information, defaults to False
    :type verbose: bool, optional
    :param progress_callback: Optional callback function to report epoch progress, defaults to None
    :type progress_callback: callable, optional
    :return: Trained model with updated final learning rate
    :rtype: torch.nn.Module
    """
    model.to(device)
    optimizer = Adam(model.parameters(), lr=initial_lr, weight_decay = weight_decay)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=decay_factor)
    refit_history = pd.DataFrame()

    def _to_scalar(x):
        """Convert torch tensors to Python scalars safely."""
        if torch.is_tensor(x):
            return x.detach().cpu().item()
        return x

    ## Added to handle return history
        # Container to collect per-epoch metrics

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in imputed_data:
            # MODIFIED: Capture idx_batch properly instead of using *_
            # print(f"Batch is: {len(batch)}\n")
            x_batch, cluster_batch, mask_batch, idx_batch= batch
            x_batch = x_batch.to(device)
            cluster_batch = cluster_batch.to(device)
            mask_batch = mask_batch.to(device)

            # ADDED: Get imputable mask for this batch
            dataset = imputed_data.dataset
            if hasattr(dataset, 'imputable') and dataset.imputable is not None:
                imputable_batch = dataset.imputable[idx_batch].to(device).float()
            else:
                imputable_batch = None
            
            recon_x, mu, logvar = model(x_batch, cluster_batch)
            
            # MODIFIED: Pass imputable_mask to loss function
            loss, train_mse, train_bce = loss_function_nomask(
                cluster_batch, recon_x, x_batch, dataset.binary_feature_mask, mu, logvar,
                beta=beta, return_components=True,
                imputable_mask=imputable_batch,  # ADDED
                device = device,
                debug = model.debug
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(imputed_data.dataset)
        if verbose:
            print(f"Epoch {epoch + 1}, Refit Loss: {avg_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # -------------------------------
        # Logging to history
        # -------------------------------
        record = {
            "epoch": epoch,
            "train_loss": avg_loss,
            "train_mse": _to_scalar(train_mse),
            "train_bce":_to_scalar(train_bce),
            "imputation_error": np.nan,
            "val_mse":np.nan,
            "val_bce":np.nan,
            "lr": optimizer.param_groups[0]["lr"],
            "phase": "refit_training",
            "loop": np.nan
        }
        ## this weird thing b/c pandas doesn't have append anymore
        refit_history = pd.concat([refit_history,
         pd.DataFrame([record])],
        ignore_index=True)

        #------------------
        # progress bar hook
        #------------------
        if progress_callback:
            progress_callback(1)
        scheduler.step()

        # if avg_loss < best_loss:
        #     best_loss = avg_loss
        #     patience_counter = 0
        # else:
        #     patience_counter += 1
        #     if patience_counter >= patience:
        #         if verbose:
        #             print("Early stopping triggered.")
        #         break

    model.set_final_lr(optimizer.param_groups[0]['lr'])

    return model, refit_history




def impute_and_refit_loop(model, train_loader, max_loops=10, patience=2,
                          epochs_per_loop=5, initial_lr=None, decay_factor=0.999, weight_decay = 0.001,
                          beta=0.1, device="cpu", verbose=False, batch_size=4000,
                          progress_epoch=None):
    """Iterative impute-refit loop with validation MSE early stopping.
    
    Performs alternating cycles of imputation (filling missing values with model predictions)
    and refitting (training on the complete imputed data). Uses early stopping based on
    validation MSE to prevent overfitting and selects the best performing model.
    
    :param model: Trained VAE model to start the impute-refit process
    :type model: torch.nn.Module
    :param train_loader: DataLoader for the original training dataset with missing values
    :type train_loader: torch.utils.data.DataLoader
    :param max_loops: Maximum number of impute-refit cycles to perform, defaults to 10
    :type max_loops: int, optional
    :param patience: Number of loops to wait for improvement before early stopping, defaults to 2
    :type patience: int, optional
    :param epochs_per_loop: Number of training epochs per refit cycle, defaults to 5
    :type epochs_per_loop: int, optional
    :param initial_lr: Learning rate for refit training, uses model's final LR if None, defaults to None
    :type initial_lr: float, optional
    :param decay_factor: Exponential decay factor for learning rate, defaults to 0.999
    :type decay_factor: float, optional
    :param beta: Weight for KL divergence term in loss function, defaults to 0.1
    :type beta: float, optional
    :param device: Device to run computations on, defaults to "cpu"
    :type device: str, optional
    :param verbose: Whether to print detailed progress information, defaults to False
    :type verbose: bool, optional
    :param batch_size: Batch size for refit training, defaults to 4000
    :type batch_size: int, optional
    :param progress_epoch: Optional callback function to report epoch progress, defaults to None
    :type progress_epoch: callable, optional
    :return: Tuple containing (imputed_dataframe, best_model, best_dataset, refit_history_dataframe)
        refit_history_dataframe Columns:
          - epoch (int)          : cumulative epoch counter (continues from initial)
          - train_loss (float)   : NaN (not tracked during refit here)
          - val_mse (float)      : validation MSE after each refit loop
          - lr (float)           : learning rate after each refit loop
          - phase (str)          : {"refit_init", "refit_loop"}
          - loop (int)           : 0 for baseline (pre-refit), then 1..k per loop
    :rtype: tuple[pandas.DataFrame, torch.nn.Module, ClusterDataset, pandas.DataFrame]

    """
    # --------------------------
    # Get imputed dataset, save 'best' states of dataset, model
    # Create data loader to start loop, initialize patience counter
    # Start list for val_mse_history
    # --------------------------

    ## get initial imputed dataset and hold it, create data loader, preserve model
    dataset  = get_imputed(model, train_loader, device=device)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    best_dataset = copy.deepcopy(dataset)
    best_imputation_error = float("inf")
    best_model = copy.deepcopy(model)
    patience_counter = 0
    val_mse_history = []
    
    ## set lrs
    if initial_lr is None:
        if verbose:
            print("No LR givin, using last lr from initial training!")
    else:
        model.set_final_lr(initial_lr)
        if verbose:
            print(f"Set lr to {initial_lr}")

    
    refit_lr = model.get_final_lr()



    # Determine where to continue the epoch counter
    # If user trained with train_vae_initial and attached .training_history_, keep continuity.
    if hasattr(model, "training_history_") and isinstance(model.training_history_, pd.DataFrame):
        try:
            start_epoch = int(np.nanmax(model.training_history_["epoch"].values))
        except Exception:
            start_epoch = 0
    else:
        start_epoch = 0

    # --------------------------
    # Compute initial MSE (before loop)
    # --------------------------
    imputation_error, val_mse, val_bce = compute_val_mse(model, dataset, device)

        # -------------------------------
        # Logging to history
        # -------------------------------
    record = {
            "epoch": start_epoch,
            "train_loss": np.nan,
            "train_mse": np.nan,
            "train_bce":np.nan,
            "imputation_error": imputation_error,
            "val_mse": val_mse,
            "val_bce":val_bce,
            "lr": refit_lr,
            "phase": "refit",
            "loop": 0
        }

    model.training_history_ = pd.concat([model.training_history_,
         pd.DataFrame([record])],
        ignore_index=True)

    if verbose:
        print(f"Initial Validation MSE (pre-refit): {val_mse:.6f}")

    loop_history = pd.DataFrame()

    for loop in range(max_loops):
        
        if verbose:
            print(f"\n=== Impute-Refit Loop {loop + 1}/{max_loops} ===")
        
        if verbose:
            print(f"Current lr is {refit_lr}")
        # --------------------------
        # Refit the model
        # --------------------------
        model, refit_history = train_vae_refit(
            model=model,
            imputed_data=data_loader,
            epochs=epochs_per_loop,
            initial_lr=refit_lr,
            decay_factor=decay_factor,
            beta=beta,
            weight_decay = weight_decay,
            device=device,
            verbose=verbose,
            progress_callback = progress_epoch
        )

        # --------------------------
        # Compute validation MSE
        # If val MSE for this loop is better than current best, 
        # replace best_imputation_error and best_model + reset patience_counter,
        # and get new imputed dataset + data_loader
        # If not better, increment patience_counter and if patience_counter >= patience, break loop. 
        # --------------------------
        imputation_error, val_mse, val_bce = compute_val_mse(model, data_loader.dataset, device)
        # Advance epoch counter by the epochs we just trained
        epoch_after_loop = start_epoch + (loop + 1) * epochs_per_loop
        refit_lr = float(model.get_final_lr())
                # Log history row for this loop
        # -------------------------------
        # Logging to history
        # -------------------------------
        record = {
            "epoch": epoch_after_loop,
            "train_loss": np.nan,
            "train_mse": np.nan,
            "train_bce":np.nan,
            "imputation_error": imputation_error,
            "val_mse": val_mse,
            "val_bce":val_bce,
            "lr": refit_lr,
            "phase": "refit",
            "loop": 0
        }

        loop_history = pd.concat([loop_history,
         pd.DataFrame([record])],
        ignore_index=True)

        loop_history = pd.concat([loop_history, refit_history])

        if verbose:
            print(f"Loop {loop + 1} Validation Loss: {imputation_error:.6f}")

        if imputation_error < best_imputation_error:
            best_imputation_error = imputation_error
            best_model = copy.deepcopy(model)
            patience_counter = 0
            best_dataset = get_imputed(model, data_loader, device=device)
            data_loader = DataLoader(best_dataset, batch_size=batch_size, shuffle=True)
        else:
            patience_counter += 1
            imputed_dataset = get_imputed(model, data_loader, device = device)
            data_loader = DataLoader(imputed_dataset, batch_size=batch_size, shuffle=True)
            if patience_counter >= patience:
                if verbose:
                    print("Early stopping triggered.")
                break

    ## Add histories to best model
    best_model.training_history_ = pd.concat([best_model.training_history_,
         loop_history],
        ignore_index=True)            
    # -----------------------------
    # Final denormalized output
    # Get mean and sd from this
    # Apply this final model on the original dataset 
    # -----------------------------
    # final_val_mse = compute_val_mse(best_model, dataset, device)
    # final_imputed = get_imputed(best_model, train_loader, device)

    # ## try using the best dataset
    final_imputation_error, final_val_mse, final_val_bce = compute_val_mse(best_model, dataset, device)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    #final_imputed = get_imputed(best_model, data_loader, device)

    imputed_df = get_imputed_df(best_model, data_loader, device)

    if verbose: 
        print(f"Best Imputation Error {best_imputation_error}. Imputed Dataset Error {final_imputation_error}")


    # # --- Assemble the refit history DataFrame ---
    # refit_history_df = pd.DataFrame(
    #     history_rows,
    #     columns=["epoch", "train_loss", "train_recon", "train_kl", "val_mse", "lr", "phase", "loop"],
    # )

    return imputed_df, best_model, best_dataset
