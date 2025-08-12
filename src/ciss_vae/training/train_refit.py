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


def train_vae_refit(model, imputed_data, epochs=10, initial_lr=0.01,
                    decay_factor=0.999, beta=0.1,
                    device="cpu", verbose=False, progress_callback = None):
    model.to(device)
    optimizer = Adam(model.parameters(), lr=initial_lr)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=decay_factor)


    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in imputed_data:
            x_batch, cluster_batch, mask_batch, *_ = batch
            x_batch = x_batch.to(device)
            cluster_batch = cluster_batch.to(device)
            mask_batch = mask_batch.to(device)

            recon_x, mu, logvar = model(x_batch, cluster_batch)
            loss, _, _ = loss_function_nomask(
                cluster_batch, recon_x, x_batch, mu, logvar,
                beta=beta, return_components=True
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(imputed_data.dataset)
        if verbose:
            print(f"Epoch {epoch + 1}, Refit Loss: {avg_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
        
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
    return model




def impute_and_refit_loop(model, train_loader, max_loops=10, patience=2,
                          epochs_per_loop=5, initial_lr=None, decay_factor=0.999,
                          beta=0.1, device="cpu", verbose=False, batch_size=4000,
                          progress_phase=None, progress_epoch=None):
    """
    Iterative impute-refit loop with validation MSE early stopping.
    Returns
    -------
    imputed_df : pd.DataFrame
        Final imputed values produced by the best (early-stopped) model.
    best_model : torch.nn.Module
        Copy of the best-performing model encountered during the loop.
    best_dataset : ClusterDataset
        Copy of the dataset corresponding to `best_model`'s imputations.
    refit_history_df : pd.DataFrame
        Per-loop history aligned to `train_vae_initial`'s history schema so you
        can `pd.concat([history_df, refit_history_df], ignore_index=True)` later.

        Columns:
          - epoch (int)          : cumulative epoch counter (continues from initial)
          - train_loss (float)   : NaN (not tracked during refit here)
          - train_recon (float)  : NaN
          - train_kl (float)     : NaN
          - val_mse (float)      : validation MSE after each refit loop
          - lr (float)           : learning rate after each refit loop
          - phase (str)          : {"refit_init", "refit_loop"}
          - loop (int)           : 0 for baseline (pre-refit), then 1..k per loop
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
    best_val_mse = float("inf")
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

    # --- History container (schema matches train_vae_initial + extras) ---
    history_rows = []
    def _append_history_row(epoch_int: int, val_mse: float, lr_val: float, phase: str, loop_idx: int):
        history_rows.append({
            "epoch": int(epoch_int),
            "train_loss": np.nan,
            "train_recon": np.nan,
            "train_kl": np.nan,
            "val_mse": float(val_mse),
            "lr": float(lr_val),
            "phase": phase,
            "loop": int(loop_idx),
        })

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
    val_mse = compute_val_mse(model, dataset, device)
    _append_history_row(epoch_int=start_epoch, val_mse=val_mse, lr_val=refit_lr, phase="refit_init", loop_idx=0)
    if verbose:
        print(f"Initial Validation MSE (pre-refit): {val_mse:.6f}")


    for loop in range(max_loops):
        if progress_phase:
            # show epochs for this refit segment
            progress_phase(epochs_per_loop, label=f"Refit loop {loop+1}")
        if verbose:
            print(f"\n=== Impute-Refit Loop {loop + 1}/{max_loops} ===")
        
        if verbose:
            print(f"Current lr is {refit_lr}")
        # --------------------------
        # Refit the model
        # --------------------------
        model = train_vae_refit(
            model=model,
            imputed_data=data_loader,
            epochs=epochs_per_loop,
            initial_lr=refit_lr,
            decay_factor=decay_factor,
            beta=beta,
            device=device,
            verbose=verbose,
            progress_callback = progress_epoch
        )

        # --------------------------
        # Compute validation MSE
        # If val MSE for this loop is better than current best, 
        # replace best_val_mse and best_model + reset patience_counter,
        # and get new imputed dataset + data_loader
        # If not better, increment patience_counter and if patience_counter >= patience, break loop. 
        # --------------------------
        val_mse = compute_val_mse(model, data_loader.dataset, device)
        # Advance epoch counter by the epochs we just trained
        epoch_after_loop = start_epoch + loop * epochs_per_loop
        refit_lr = float(model.get_final_lr())
                # Log history row for this loop
        _append_history_row(
            epoch_int=epoch_after_loop,
            val_mse=val_mse,
            lr_val=refit_lr,
            phase="refit_loop",
            loop_idx=loop,
        )

        if verbose:
            print(f"Loop {loop + 1} Validation MSE: {val_mse:.6f}")

        if val_mse < best_val_mse:
            best_val_mse = val_mse
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

    # -----------------------------
    # Final denormalized output
    # Get mean and sd from this
    # Apply this final model on the original dataset 
    # -----------------------------
    # final_val_mse = compute_val_mse(best_model, dataset, device)
    # final_imputed = get_imputed(best_model, train_loader, device)

    # ## try using the best dataset
    final_val_mse = compute_val_mse(best_model, dataset, device)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    #final_imputed = get_imputed(best_model, data_loader, device)

    imputed_df = get_imputed_df(best_model, data_loader, device)

    if verbose: 
        print(f"Best Val MSE {best_val_mse}. Imputed Dataset MSE {final_val_mse}")


    # --- Assemble the refit history DataFrame ---
    refit_history_df = pd.DataFrame(
        history_rows,
        columns=["epoch", "train_loss", "train_recon", "train_kl", "val_mse", "lr", "phase", "loop"],
    )

    return imputed_df, best_model, best_dataset, refit_history_df
