import torch
import torch.nn.functional as F

def loss_function(cluster, mask, recon_x, x, mu, 
logvar, beta=0.001, return_components=False):
    """
    VAE loss function with masking and KL annealing.

    Parameters:
    - cluster: LongTensor (batch_size,) of cluster labels.
    - mask: FloatTensor (batch_size, input_dim), 1s where observed, 0s where missing.
    - recon_x: FloatTensor (batch_size, input_dim), model reconstruction output.
    - x: FloatTensor (batch_size, input_dim), original input.
    - mu: FloatTensor (batch_size, latent_dim), encoder means.
    - logvar: FloatTensor (batch_size, latent_dim), encoder log-variances.
    - beta: float, KL loss multiplier (e.g., for β-VAE).
    - return_components: bool, return (total, mse, kl) if True.

    Returns:
    - total_loss if return_components is False
    - (total_loss, mse_loss, kl_loss) if return_components is True
    """
    # --------------------------
    # Calculate Losses
    # --------------------------

    ## reconstruction  -- sort recon_x to the right thing.  
    mse_loss = F.mse_loss(recon_x*mask, x*mask, reduction='sum')

    ## KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    total_loss = mse_loss + beta * kl_loss

    if return_components:
        return total_loss, mse_loss, kl_loss
    return total_loss


def loss_function_nomask(cluster, recon_x, x, mu, 
logvar, beta=0.001, return_components=False):
    """
    VAE loss function with masking and KL annealing.

    Parameters:
    - cluster: LongTensor (batch_size,) of cluster labels.
    - mask: FloatTensor (batch_size, input_dim), 1s where observed, 0s where missing.
    - recon_x: FloatTensor (batch_size, input_dim), model reconstruction output.
    - x: FloatTensor (batch_size, input_dim), original input.
    - mu: FloatTensor (batch_size, latent_dim), encoder means.
    - logvar: FloatTensor (batch_size, latent_dim), encoder log-variances.
    - beta: float, KL loss multiplier (e.g., for β-VAE).
    - return_components: bool, return (total, mse, kl) if True.

    Returns:
    - total_loss if return_components is False
    - (total_loss, mse_loss, kl_loss) if return_components is True
    """
    # --------------------------
    # Calculate Losses
    # --------------------------

    ## reconstruction  -- sort recon_x to the right thing.  
    mse_loss = F.mse_loss(recon_x, x, reduction='sum')

    ## KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    total_loss = mse_loss + beta * kl_loss

    if return_components:
        return total_loss, mse_loss, kl_loss
    return total_loss