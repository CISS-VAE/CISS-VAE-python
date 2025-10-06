import torch
import torch.nn.functional as F

def loss_function(cluster, mask, recon_x, x, mu, 
logvar, beta=0.001, return_components=False, imputable_mask=None):
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
    # Calculate Losses -> for initial loop
    # --------------------------
        # Don't need dni mask here -> dni always 1 where mask is 1
    # if(imputable_mask is not None):
    #     print(f"  imputable_mask: shape={imputable_mask.shape}, dtype={imputable_mask.dtype}, "
    #               f"num ones={(imputable_mask==1).sum().item()}, "
    #               f"num zeros={(imputable_mask==0).sum().item()}")
    #     overlap = (mask.bool() & (imputable_mask == 1)).sum().item()
    #     print(f"  overlap(real mask & imputable=1): {overlap} entries\n")
    #     print(f"mask \n{mask}\n\n recon_x \n{recon_x}")
    

    ## x is x_batch
    ## recon_x is also by batch

    ## reconstruction  -- sort recon_x to the right thing.  
    mse_loss = F.mse_loss(recon_x*mask, x*mask, reduction='sum')

    print(f"mse_loss\n{mse_loss} = F.mse_loss(recon_x\n{recon_x}*mask\n{mask}, x\n{x}*mask\n{mask}, reduction='sum')")

    ## KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    total_loss = mse_loss + beta * kl_loss

    print(f"\nkl_loss =  -0.5 * torch.sum(1 + {logvar} - {mu.pow(2)} - {logvar.exp()}) ")
    print(f"\ntotal_loss {total_loss} = mse_loss {mse_loss} + beta {beta} * kl_loss {kl_loss}")


    if return_components:
        return total_loss, mse_loss, kl_loss
    return total_loss


def loss_function_nomask(cluster, recon_x, x, mu, 
logvar, beta=0.001, return_components=False, imputable_mask=None):
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
    # Calculate Losses -> for the iterative loop
    # --------------------------

    # NEW 11SEP2025: Apply imputable mask if provided
    if imputable_mask is not None:
        # Only compute loss where imputable_mask == 1 (can impute)
        print(f"  from nomask_imputable_mask: shape={imputable_mask.shape}, dtype={imputable_mask.dtype}, "
                f"num ones={(imputable_mask==1).sum().item()}, "
                f"num zeros={(imputable_mask==0).sum().item()}")
        overlap = (recon_x.bool() & (imputable_mask == 1)).sum().item()
        print(f"  overlap(recon_x & imputable=1): {overlap} entries")
        print(f"imputable mask \n{imputable_mask}\n\n recon_x \n{recon_x} \n\n x \n{x}")
        mse_loss = F.mse_loss(recon_x * imputable_mask, x * imputable_mask, reduction='sum')
    else:
        # Original behavior if no mask provided
        mse_loss = F.mse_loss(recon_x, x, reduction='sum')

    ## KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    print(f"\nkl_loss =  -0.5 * torch.sum(1 + {logvar} - {mu.pow(2)} - {logvar.exp()}) ")

    total_loss = mse_loss + beta * kl_loss

    print(f"\ntotal_loss (nomask) {total_loss} = mse_loss {mse_loss} + beta {beta} * kl_loss {kl_loss}")

    if return_components:
        return total_loss, mse_loss, kl_loss
    return total_loss