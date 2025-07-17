import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam, lr_scheduler
import pandas as pd
from ciss_vae.utils.loss import loss_function, loss_function_nomask
from ciss_vae.classes.cluster_dataset import ClusterDataset
from torch.utils.data import DataLoader
from ciss_vae.utils.helpers import get_imputed_df, get_imputed, compute_val_mse
import copy


def train_vae_refit(model, imputed_data, epochs=10, initial_lr=0.01,
                    decay_factor=0.999, beta=0.1,
                    device="cpu", verbose=False):
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


# def get_imputed(model, data_loader, device="cpu"): ## needs to return a clusterdataset
#     model.eval()
#     imputed = []

#     with torch.no_grad():
#         for batch in data_loader:
#             x_batch, cluster_batch, mask_batch, *rest = batch
#             idx_batch = rest[0] if rest else None

#             x_batch = x_batch.to(device)
#             cluster_batch = cluster_batch.to(device)
#             mask_batch = mask_batch.to(device)

#             recon_x, _, _ = model(x_batch, cluster_batch)
#             x_filled = x_batch * mask_batch + recon_x * (~mask_batch)

#             if idx_batch is not None:
#                 imputed.append((x_filled.cpu(), cluster_batch.cpu(), mask_batch.cpu(), idx_batch))
#             else:
#                 raise ValueError("Index batch required for proper output alignment.")

#     x_all = torch.cat([x for x, _, _, _ in imputed], dim=0)
#     c_all = torch.cat([c for _, c, _, _ in imputed], dim=0)
#     m_all = torch.cat([m for _, _, m, _ in imputed], dim=0)
#     i_all = torch.cat([i for _, _, _, i in imputed], dim=0)

#     return TensorDataset(x_all, c_all, m_all, i_all)



def impute_and_refit_loop(model, train_loader, max_loops=10, patience=2,
                          epochs_per_loop=5, initial_lr=None, decay_factor=0.999,
                          beta=0.1, device="cpu", verbose=False, batch_size=4000):
    """
    Iterative impute-refit loop with validation MSE early stopping.
    Returns:
        - trained model
        - denormalized, imputed dataframe
        - validation MSE history
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

    # --------------------------
    # Compute initial MSE (before loop)
    # --------------------------
    val_mse = compute_val_mse(model, dataset, device)
    val_mse_history.append(val_mse)
    if verbose:
        print(f"Initial Validation MSE: {val_mse:.6f}")

    for loop in range(max_loops):
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
            verbose=verbose
        )

        # --------------------------
        # Compute validation MSE
        # If val MSE for this loop is better than current best, 
        # replace best_val_mse and best_model + reset patience_counter,
        # and get new imputed dataset + data_loader
        # If not better, increment patience_counter and if patience_counter >= patience, break loop. 
        # --------------------------
        val_mse = compute_val_mse(model, data_loader.dataset, device)
        val_mse_history.append(val_mse)

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

    # # x_all = dataset.data
    # x_all = final_imputed.data
    
    # means = torch.tensor(final_imputed.feature_means, dtype=torch.float32)
    # stds = torch.tensor(final_imputed.feature_stds, dtype=torch.float32)
    
    # # Denormalize imputed values
    # x_all_denorm = x_all * stds + means

    # # -------------------------------------
    # # Replace validation-masked entries with true values
    # # -------------------------------------

    # # Ensure val_data and val_mask are on the same device as x_all_denorm
    # val_data_tensor = dataset.val_data.to(x_all_denorm.device)
    # val_mask_tensor = ~torch.isnan(val_data_tensor)

    # # Overwrite imputed values with ground truth at validation positions
    # x_all_denorm[val_mask_tensor] = val_data_tensor[val_mask_tensor]

    # x_all_denorm_np = x_all_denorm.detach().cpu().numpy()  # Now safe to convert


    # # -------------------------------------
    # # Create final DataFrame
    # # -------------------------------------
    # feature_names = getattr(best_dataset, "feature_names", [f"V{i}" for i in range(x_all.shape[1])])

    # # The imputed dataset should be in the right order os this should be unecessary
    # # # Recover original index from dataset.indices
    # # if hasattr(dataset, "indices"):
    # #     base_index = dataset.indices
    # #     if isinstance(base_index, torch.Tensor):
    # #         base_index = base_index.cpu().numpy()
    # #     full_index = base_index[idx_all_np]
    # # else:
    # #     full_index = idx_all_np

    # # Build DataFrame and sort to match original row order
    # df_unsorted = pd.DataFrame(x_all_denorm_np, columns=feature_names, index=best_dataset.indices.cpu().numpy())
    # imputed_df = df_unsorted.sort_index()

    return imputed_df, best_model, best_dataset, val_mse_history
