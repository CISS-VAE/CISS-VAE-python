import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from ciss_vae.utils.loss import loss_function
import torch.nn.functional as F
from ciss_vae.utils.helpers import compute_val_mse

def train_vae_initial(
    model, 
    train_loader, 
    epochs=10, 
    initial_lr=0.01, 
    decay_factor=0.999, 
    beta=0.1, 
    device="cpu",
    verbose=False
):
    """
    Training loop for VAE on masked data with validation set aside inside the ClusterDataset.
    Tracks early stopping using MSE at validation-masked positions.

    Each sample must include 4 items: (x, cluster_id, mask, index)
    """

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=initial_lr)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=decay_factor)

    # Pull dataset object from loader to get validation targets
    dataset = train_loader.dataset
    if not hasattr(dataset, "val_data"):
        raise ValueError("Dataset must include 'val_data' for validation-based early stopping.")


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

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader.dataset)
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
        # Logging and early stopping
        # -----------------------------------
        if verbose:
            print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val MSE: {val_mse:.6f}, LR: {current_lr:.6f}")

        # if val_mse < best_val_mse - min_delta:
        #     best_val_mse = val_mse
        #     patience_counter = 0
        # else:
        #     patience_counter += 1
        #     if verbose:
        #         print(f"Early stopping counter: {patience_counter}/{patience}")
        #     if patience_counter >= patience:
        #         if verbose:
        #             print("Early stopping triggered. Training stopped.")
        #         break

        scheduler.step()

    model.set_final_lr(optimizer.param_groups[0]["lr"])
    return model
