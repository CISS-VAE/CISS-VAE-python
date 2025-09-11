import torch
import torch.nn.functional as F

def loss_function(cluster, mask, recon_x, x, mu, 
logvar, beta=0.001, return_components=False, do_not_impute_mask=None):
    """VAE loss function with masking and KL annealing.
    
    :param cluster: Cluster labels, shape ``(batch_size,)``
    :type cluster: torch.LongTensor
    :param mask: Binary mask where 1s indicate observed values and 0s indicate missing values, shape ``(batch_size, input_dim)``
    :type mask: torch.FloatTensor
    :param recon_x: Model reconstruction output, shape ``(batch_size, input_dim)``
    :type recon_x: torch.FloatTensor
    :param x: Original input, shape ``(batch_size, input_dim)``
    :type x: torch.FloatTensor
    :param mu: Encoder means, shape ``(batch_size, latent_dim)``
    :type mu: torch.FloatTensor
    :param logvar: Encoder log-variances, shape ``(batch_size, latent_dim)``
    :type logvar: torch.FloatTensor
    :param beta: KL loss multiplier (e.g., for β-VAE), defaults to 0.001
    :type beta: float, optional
    :param return_components: If True, return individual loss components, defaults to False
    :type return_components: bool, optional
    :return: Total loss if ``return_components`` is False, otherwise tuple ``(total_loss, mse_loss, kl_loss)``
    :rtype: torch.Tensor or tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    """
    # --------------------------
    # Calculate Losses
    # --------------------------
        # NEW 11SEP2025: Combine original mask with do_not_impute mask
    if do_not_impute_mask is not None:
        # Final mask: observed values (mask==1) AND can_impute (do_not_impute_mask==1)
        final_mask = mask * do_not_impute_mask
    else:
        # Original behavior if no do_not_impute mask provided
        final_mask = mask

    ## reconstruction  -- sort recon_x to the right thing.  
    mse_loss = F.mse_loss(recon_x*final_mask, x*final_mask, reduction='sum')

    ## KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    total_loss = mse_loss + beta * kl_loss

    if return_components:
        return total_loss, mse_loss, kl_loss
    return total_loss


def loss_function_nomask(cluster, recon_x, x, mu, 
logvar, beta=0.001, return_components=False, do_not_impute_mask=None):
    """VAE loss function without masking and with KL annealing.
    
    :param cluster: Cluster labels, shape ``(batch_size,)``
    :type cluster: torch.LongTensor
    :param recon_x: Model reconstruction output, shape ``(batch_size, input_dim)``
    :type recon_x: torch.FloatTensor
    :param x: Original input, shape ``(batch_size, input_dim)``
    :type x: torch.FloatTensor
    :param mu: Encoder means, shape ``(batch_size, latent_dim)``
    :type mu: torch.FloatTensor
    :param logvar: Encoder log-variances, shape ``(batch_size, latent_dim)``
    :type logvar: torch.FloatTensor
    :param beta: KL loss multiplier (e.g., for β-VAE), defaults to 0.001
    :type beta: float, optional
    :param return_components: If True, return individual loss components, defaults to False
    :type return_components: bool, optional
    :return: Total loss if ``return_components`` is False, otherwise tuple ``(total_loss, mse_loss, kl_loss)``
    :rtype: torch.Tensor or tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    """
    # --------------------------
    # Calculate Losses
    # --------------------------

    # NEW 11SEP2025: Apply do_not_impute mask if provided
    if do_not_impute_mask is not None:
        # Only compute loss where do_not_impute_mask == 1 (can impute)
        mse_loss = F.mse_loss(recon_x * do_not_impute_mask, x * do_not_impute_mask, reduction='sum')
    else:
        # Original behavior if no mask provided
        mse_loss = F.mse_loss(recon_x, x, reduction='sum')

    ## KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    total_loss = mse_loss + beta * kl_loss

    if return_components:
        return total_loss, mse_loss, kl_loss
    return total_loss