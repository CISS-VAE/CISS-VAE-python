"""VAE model class."""

import torch
import torch.nn as nn

class CISSVAE(nn.Module):
    def __init__(self, input_dim, hidden_dims, layer_order_enc, layer_order_dec,
                 latent_shared, latent_dim, output_shared, num_clusters, debug=False):
        """
        Variational Autoencoder supporting flexible shared/unshared layers across clusters.

        Parameters:
        - input_dim: int, number of input features
        - hidden_dims: list[int], dimensions of hidden layers
        - layer_order_enc: list[str], "shared"/"unshared" for encoder layers
        - layer_order_dec: list[str], "shared"/"unshared" for decoder layers
        - latent_shared: bool, whether latent representation is shared across clusters
        - latent_dim: int, dimensionality of the latent representation
        - output_shared: bool, whether output layer is shared across clusters
        - num_clusters: int, number of clusters
        - debug: bool, print shape and routing info if True
        """
        super().__init__()
        self.debug = debug
        self.num_clusters = num_clusters
        self.latent_shared = latent_shared
        self.layer_order_enc = layer_order_enc
        self.layer_order_dec = layer_order_dec
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.output_shared = output_shared
        self.hidden_dims = hidden_dims

        # ----------------------------
        # Encoder: shared and unshared
        # ----------------------------
        ## Shared
        self.encoder_layers = nn.ModuleList()
        ## Unshared
        self.cluster_encoder_layers = nn.ModuleDict({str(i): nn.ModuleList() for i in range(num_clusters)})

        in_features = input_dim
        for idx, (out_features, layer_type) in enumerate(zip(hidden_dims, layer_order_enc)):
            layer_type_lower = layer_type.lower()  # Case-insensitive check
            # Prepare factory_kwargs, only including non-None dtype/device
            factory_kwargs = {}
            if layer_type_lower in  ["shared", "s"]: ## if shared add to .encoder_layers
                self.encoder_layers.append(
                    nn.Sequential(
                        nn.Linear(in_features, out_features, **factory_kwargs), 
                        nn.ReLU()))
            elif layer_type_lower in ["unshared", "u"]: ## if unshared add to .cluster_encoder_layers
                for c in range(num_clusters): ## .cluster_encoder_layers has layer list for each cluster
                    self.cluster_encoder_layers[str(c)].append(
                        nn.Sequential(
                            nn.Linear(in_features, out_features, **factory_kwargs), 
                            nn.ReLU()))
            else:
                raise ValueError(f"Invalid encoder layer type at index {idx}: {layer_type}")
            in_features = out_features

        # ----------------------------
        # Latent layers
        # ----------------------------
        if latent_shared:
            self.fc_mu = nn.Linear(in_features, latent_dim)
            self.fc_logvar = nn.Linear(in_features, latent_dim)
        else:
            self.cluster_fc_mu = nn.ModuleDict({str(i): nn.Linear(in_features, latent_dim) for i in range(num_clusters)})
            self.cluster_fc_logvar = nn.ModuleDict({str(i): nn.Linear(in_features, latent_dim) for i in range(num_clusters)})

        # ----------------------------
        # Decoder: shared and unshared
        # ----------------------------
        self.decoder_layers = nn.ModuleList()
        self.cluster_decoder_layers = nn.ModuleDict({str(i): nn.ModuleList() for i in range(num_clusters)})

        ## Set up the layers list w/ correct layers in correct order. 
        in_features = latent_dim
        for idx, (out_features, layer_type) in enumerate(zip(hidden_dims[::-1], layer_order_dec)):
            factory_kwargs = {}
            if layer_type == "shared":
                self.decoder_layers.append(
                    nn.Sequential(
                        nn.Linear(in_features, out_features, **factory_kwargs), 
                        nn.ReLU()))
            elif layer_type == "unshared":
                for c in range(num_clusters):
                    self.cluster_decoder_layers[str(c)].append(
                        nn.Sequential(
                            nn.Linear(in_features, out_features, **factory_kwargs), 
                            nn.ReLU()))
            else:
                raise ValueError(f"Invalid decoder layer type at index {idx}: {layer_type}")
            in_features = out_features

        # ----------------------------
        # Output Layer - needs to be able to switch between shared and unshared
        # ----------------------------
        if self.output_shared:
            self.final_layer = nn.Linear(in_features, input_dim) ## this needs to be able to switch between shared and unshared
        else:
            self.cluster_final_layer = nn.ModuleDict({str(i): nn.Linear(in_features, input_dim) for i in range(num_clusters)})
    def set_final_lr(self, final_lr):
        """Stores final lr from initial training loop in model attributes to be accessed in refit loop."""
        self.final_lr = final_lr

    def get_final_lr(self):
        """Returns the learning rate stored with self.set_final_lr/"""
        return(self.final_lr)

    def route_through_layers(self, x, cluster_labels, layer_type_list, shared_layers, unshared_layers):
        """Routes input through encoder/decoder layers while preserving input row order."""
        shared_idx = 0
        unshared_idx = {str(c): 0 for c in range(self.num_clusters)}

        # Debug: Save original row order
        if self.debug:
            input_hash = torch.arange(x.shape[0], device=x.device)

        for layer_num, layer_type in enumerate(layer_type_list):
            if layer_type == "shared":
                x = shared_layers[shared_idx](x)
                shared_idx += 1
            else:
                outputs = []
                for c in range(self.num_clusters):
                    mask = (cluster_labels == c)
                    if mask.any():
                        x_cluster = x[mask]
                        x_out = unshared_layers[str(c)][unshared_idx[str(c)]](x_cluster)
                        outputs.append((mask, x_out))
                        # if self.debug:
                        #     print(f"[DEBUG] Layer {layer_num}: Cluster {c} through unshared[{unshared_idx[str(c)]}], shape: {x_out.shape}")

                ## Combine all cluster outputs in correct row positions
                out_dim = outputs[0][1].shape[1]
                output = torch.empty(x.shape[0], out_dim, device=x.device, dtype=x.dtype)
                for mask, x_out in outputs: ## Uses mask to 
                    output[mask] = x_out
                x = output

                ## Move to next unshared layer index for all clusters
                for c in range(self.num_clusters):
                    unshared_idx[str(c)] += 1

        if self.debug:
            out_hash = torch.arange(x.shape[0], device=x.device)
            assert torch.equal(input_hash, out_hash), (
                "[ERROR] Row order mismatch: output order doesn't match input order.\n"
                "→ This will break downstream masking or MSE calculations.\n"
                "→ Check unshared routing logic."
            )
            print("[DEBUG] Row order confirmed intact.")

        return x


    def encode(self, x, cluster_labels):
        """Encodes input into latent space using shared/unshared encoder layers."""
        x = self.route_through_layers(x, cluster_labels, self.layer_order_enc,
                                      self.encoder_layers, self.cluster_encoder_layers)
        ## Create latent vector
        if self.latent_shared:
            mu = self.fc_mu(x)
            logvar = self.fc_logvar(x)
        else: # if latent vector is not shared, we create a vector for mu and logvar
            mu = torch.empty(x.size(0), self.latent_dim, device=x.device)
            logvar = torch.empty_like(mu)
            for c in range(self.num_clusters):
                mask = (cluster_labels == c)
                if mask.any():
                    mu[mask] = self.cluster_fc_mu[str(c)](x[mask])
                    logvar[mask] = self.cluster_fc_logvar[str(c)](x[mask])
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """Applies reparameterization trick to sample latent z."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, cluster_labels):
        """Decodes latent representation using shared/unshared decoder layers, with optional unshared final layer."""
        
        # First decode through main shared/unshared layers
        z = self.route_through_layers(z, cluster_labels, self.layer_order_dec,
                                    self.decoder_layers, self.cluster_decoder_layers)

        if self.output_shared:
            # Shared final layer applied to all
            return self.final_layer(z)
        else:
            # Apply final layer per cluster
            outputs = []
            for c in range(self.num_clusters):
                mask = (cluster_labels == c)
                if mask.any():
                    z_cluster = z[mask]
                    z_out = self.cluster_final_layer[str(c)](z_cluster)
                    outputs.append((mask, z_out)) 

            # Combine all cluster outputs in correct row positions
            out_dim = outputs[0][1].shape[1]
            output = torch.empty(z.shape[0], out_dim, device=z.device, dtype=z.dtype)
            for mask, z_out in outputs:
                output[mask] = z_out
            return output


    def forward(self, x, cluster_labels):
        """
        Full forward pass through VAE:
        1. Encode inputs to latent mean/logvar
        2. Sample z using reparameterization trick
        3. Decode z to reconstruct x
        """
        if self.debug:
            print("[DEBUG] Forward pass starting. Input shape:", x.shape)

        mu, logvar = self.encode(x, cluster_labels)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, cluster_labels)

        if self.debug:
            print("[DEBUG] Forward pass complete. Reconstructed shape:", recon.shape)
        return recon, mu, logvar

    def __repr__(self):
        lines = []
        lines.append(f"VAE1(input_dim={self.input_dim}, latent_dim={self.latent_dim}, latent_shared={self.latent_shared}, num_clusters={self.num_clusters})\n")

        lines.append("Encoder Layers:")
        in_dim = self.input_dim
        for i, (out_dim, layer_type) in enumerate(zip(self.hidden_dims, self.layer_order_enc)):
            lines.append(f"  [{i}] {layer_type.upper():<8} {in_dim} → {out_dim}")
            in_dim = out_dim

        lines.append("\nLatent Layer:")
        if self.latent_shared:
            lines.append(f"  SHARED    {in_dim} → {self.latent_dim}")
        else:
            for c in range(self.num_clusters):
                lines.append(f"  UNSHARED (cluster {c}) {in_dim} → {self.latent_dim}")

        lines.append("\nDecoder Layers:")
        hidden_dims_rev = self.hidden_dims[::-1]
        in_dim = self.latent_dim
        for i, (out_dim, layer_type) in enumerate(zip(hidden_dims_rev, self.layer_order_dec)):
            lines.append(f"  [{i}] {layer_type.upper():<8} {in_dim} → {out_dim}")
            in_dim = out_dim

        lines.append(f"\nFinal Output Layer: {in_dim} → {self.input_dim}")
        return "\n".join(lines)
