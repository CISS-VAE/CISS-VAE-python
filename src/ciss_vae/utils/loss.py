import torch
import torch.nn.functional as F
import warnings ## let's see if/when my reconX goes nan
import numpy as np

def loss_function(cluster, mask, recon_x, x, binary_feature_mask, mu, 
logvar, beta=0.001, return_components=False, imputable_mask=None, device = "cpu", debug = False):
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
    # -------------------------
    

    ## x is x_batch
    ## recon_x is also by batch
    if torch.isnan(recon_x).any():
        warnings.warn(f"[Warning] recon_x contains {torch.isnan(recon_x).sum().item()} NaN values", RuntimeWarning)

    if torch.isnan(x).any():
        warnings.warn(f"[Warning] x contains {torch.isnan(x).sum().item()} NaN values", RuntimeWarning)

    # ------------------------
    # Handle binary features
    # Addition 15 OCT 2025
    # - adding sum_loss = mse_loss + bce_loss -> gets added to beta*kl_loss to make total_loss
    # ------------------------

    if binary_feature_mask is None:
            ## reconstruction  -- sort recon_x to the right thing.  
        mse_loss = F.mse_loss(recon_x*mask, x*mask, reduction='sum')
        sum_loss = mse_loss
        bce_loss = np.nan
    else:
        binary_feature_mask =  torch.as_tensor(binary_feature_mask, dtype=torch.bool, device=device)
        cont_feat = ~binary_feature_mask
        mse_loss = F.mse_loss(recon_x*mask*cont_feat, x*mask*cont_feat, reduction='sum')
        bce_loss = F.binary_cross_entropy(recon_x*mask*binary_feature_mask, x*mask*binary_feature_mask, reduction = 'sum')
        sum_loss = mse_loss + bce_loss
        if debug:
            print(f"Loss Function Initial: sum_loss {sum_loss} = mse_loss {mse_loss} + bce_loss {bce_loss}\n\n")


    ## KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    total_loss = sum_loss + beta * kl_loss

    # print(f"\nloss_function(): kl_loss{kl_loss} =  -0.5 *{torch.sum(1 + logvar - mu.pow(2) - logvar.exp())} torch.sum(1 + logvar{logvar} - mu.pow(2) - logvar.exp(){mu.pow(2)} - {logvar.exp()}) ")
    # print(f"\ntotal_loss {total_loss} = mse_loss {mse_loss} + beta {beta} * kl_loss {kl_loss}")


    if return_components:
        return total_loss, mse_loss, bce_loss
    return total_loss


def loss_function_nomask(cluster, recon_x, x, binary_feature_mask, mu, 
logvar, beta=0.001, return_components=False, imputable_mask=None, device = "cpu", debug = False):
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
    if torch.isnan(recon_x).any():
        warnings.warn(f"[Warning] recon_x contains {torch.isnan(recon_x).sum().item()} NaN values", RuntimeWarning)

    if torch.isnan(x).any():
        warnings.warn(f"[Warning] x contains {torch.isnan(x).sum().item()} NaN values", RuntimeWarning)
    # NEW 11SEP2025: Apply imputable mask if provided
    
    # ------------------------
    # Handle binary features
    # Addition 15 OCT 2025
    # - adding sum_loss = mse_loss + bce_loss -> gets added to beta*kl_loss to make total_loss
    # ------------------------
    if imputable_mask is not None:
        # Only compute loss where imputable_mask == 1 (can impute)
        # print(f"  from nomask_imputable_mask: shape={imputable_mask.shape}, dtype={imputable_mask.dtype}, "
        #         f"num ones={(imputable_mask==1).sum().item()}, "
        #         f"num zeros={(imputable_mask==0).sum().item()}")
        # overlap = (recon_x.bool() & (imputable_mask == 1)).sum().item()
        # print(f"  overlap(recon_x & imputable=1): {overlap} entries")
        # print(f"imputable mask \n{imputable_mask}\n\n recon_x \n{recon_x} \n\n x \n{x}")
        if binary_feature_mask is None:
            mse_loss = F.mse_loss(recon_x * imputable_mask, x * imputable_mask, reduction='sum')
            sum_loss = mse_loss
            bce_loss = np.nan
        else:
            binary_feature_mask =  torch.as_tensor(binary_feature_mask, dtype=torch.bool, device=device)
            cont_feat = ~binary_feature_mask
            mse_loss = F.mse_loss(recon_x* imputable_mask*cont_feat, x*imputable_mask*cont_feat, reduction='sum')
            bce_loss = F.binary_cross_entropy(recon_x*imputable_mask*binary_feature_mask, x*imputable_mask*binary_feature_mask, reduction = 'sum')
            sum_loss = mse_loss + bce_loss
            if debug:
                print(f"Loss Function Refit: sum_loss {sum_loss} = mse_loss {mse_loss} + bce_loss {bce_loss}\n\n")
    else:
        # Original behavior if no mask provided
        if binary_feature_mask is None:
            mse_loss = F.mse_loss(recon_x, x, reduction='sum')
            sum_loss = mse_loss
            bce_loss = np.nan
        else:
            binary_feature_mask =  torch.as_tensor(binary_feature_mask, dtype=torch.bool, device=device)
            cont_feat = ~binary_feature_mask
            mse_loss = F.mse_loss(recon_x*cont_feat, x*cont_feat, reduction='sum')
            bce_loss = F.binary_cross_entropy(recon_x*binary_feature_mask, x*binary_feature_mask, reduction = 'sum')
            sum_loss = mse_loss + bce_loss
            if debug:
                print(f"Loss Function Refit: sum_loss {sum_loss} = mse_loss {mse_loss} + bce_loss {bce_loss}\n\n")

    ## KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # print(f"\nloss_function_nomask(): kl_loss{kl_loss} =  -0.5 *{torch.sum(1 + logvar - mu.pow(2) - logvar.exp())} torch.sum(1 + logvar{logvar} - mu.pow(2) - logvar.exp(){mu.pow(2)} - {logvar.exp()}) ")

    total_loss = sum_loss + beta * kl_loss

    # print(f"\ntotal_loss (nomask) {total_loss} = mse_loss {mse_loss} + beta {beta} * kl_loss {kl_loss}")

    if return_components:
        return total_loss, mse_loss, bce_loss
    return total_loss