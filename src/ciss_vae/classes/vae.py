"""VAE model class."""

import torch
import torch.nn as nn

class CISSVAE(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dims,
                 layer_order_enc,
                 layer_order_dec,
                 latent_shared,
                 latent_dim,
                 output_shared,
                 num_clusters,
                 debug=False):
        """
        Variational Autoencoder supporting flexible shared/unshared layers across clusters.

        Parameters
        ----------
        input_dim : int
            Number of input features.
        hidden_dims : list[int]
            Dimensions of hidden layers.
        layer_order_enc : list[str]
            "shared" / "unshared" for encoder layers.
        layer_order_dec : list[str]
            "shared" / "unshared" for decoder layers.
        latent_shared : bool
            Whether latent representation is shared across clusters.
        latent_dim : int
            Dimensionality of the latent space.
        output_shared : bool
            Whether output layer is shared across clusters.
        num_clusters : int
            Number of clusters.
        debug : bool
            If True, print shape/routing info.
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
        self.encoder_layers = nn.ModuleList()
        self.cluster_encoder_layers = nn.ModuleDict({
            str(i): nn.ModuleList() for i in range(num_clusters)
        })

        in_features = input_dim
        for idx, (out_features, layer_type) in enumerate(zip(hidden_dims, layer_order_enc)):
            lt = layer_type.lower()
            if lt in ["shared", "s"]:
                # (A) no dtype/device kwargs → use PyTorch defaults
                self.encoder_layers.append(
                    nn.Sequential(
                        nn.Linear(in_features, out_features),
                        nn.ReLU()
                    )
                )
            elif lt in ["unshared", "u"]:
                for c in range(num_clusters):
                    self.cluster_encoder_layers[str(c)].append(
                        nn.Sequential(
                            nn.Linear(in_features, out_features),
                            nn.ReLU()
                        )
                    )
            else:
                raise ValueError(f"Invalid encoder layer type at index {idx}: {layer_type}")
            in_features = out_features

        # ----------------------------
        # Latent layers
        # ----------------------------
        if latent_shared:
            # These also use defaults
            self.fc_mu = nn.Linear(in_features, latent_dim)
            self.fc_logvar = nn.Linear(in_features, latent_dim)
        else:
            self.cluster_fc_mu = nn.ModuleDict({
                str(i): nn.Linear(in_features, latent_dim)
                for i in range(num_clusters)
            })
            self.cluster_fc_logvar = nn.ModuleDict({
                str(i): nn.Linear(in_features, latent_dim)
                for i in range(num_clusters)
            })

        # ----------------------------
        # Decoder: shared and unshared
        # ----------------------------
        self.decoder_layers = nn.ModuleList()
        self.cluster_decoder_layers = nn.ModuleDict({
            str(i): nn.ModuleList() for i in range(num_clusters)
        })

        in_features = latent_dim
        for idx, (out_features, layer_type) in enumerate(zip(hidden_dims[::-1], layer_order_dec)):
            if layer_type.lower() in ["shared", "s"]:
                self.decoder_layers.append(
                    nn.Sequential(
                        nn.Linear(in_features, out_features),
                        nn.ReLU()
                    )
                )
            elif layer_type.lower() in ["unshared", "u"]:
                for c in range(num_clusters):
                    self.cluster_decoder_layers[str(c)].append(
                        nn.Sequential(
                            nn.Linear(in_features, out_features),
                            nn.ReLU()
                        )
                    )
            else:
                raise ValueError(f"Invalid decoder layer type at index {idx}: {layer_type}")
            in_features = out_features

        # ----------------------------
        # Output Layer
        # ----------------------------
        if output_shared:
            self.final_layer = nn.Linear(in_features, input_dim)
        else:
            self.cluster_final_layer = nn.ModuleDict({
                str(i): nn.Linear(in_features, input_dim)
                for i in range(num_clusters)
            })

    def route_through_layers(self, x, cluster_labels,
                             layer_type_list,
                             shared_layers,
                             unshared_layers):
        shared_idx = 0
        unshared_idx = {str(c): 0 for c in range(self.num_clusters)}

        if self.debug:
            input_hash = torch.arange(x.shape[0], device=x.device)

        for layer_num, layer_type in enumerate(layer_type_list):
            if layer_type.lower() in ["shared", "s"]:
                x = shared_layers[shared_idx](x)
                shared_idx += 1
            else:
                outputs = []
                for c in range(self.num_clusters):
                    mask = (cluster_labels == c)
                    if mask.any():
                        x_c = x[mask]
                        x_out = unshared_layers[str(c)][unshared_idx[str(c)]](x_c)
                        outputs.append((mask, x_out))
                out_dim = outputs[0][1].shape[1]
                # Provide explicit dtype/device from x
                output = torch.empty(x.shape[0], out_dim,
                                     device=x.device,
                                     dtype=x.dtype)
                for mask, x_out in outputs:
                    output[mask] = x_out
                x = output
                for c in range(self.num_clusters):
                    unshared_idx[str(c)] += 1

        if self.debug:
            out_hash = torch.arange(x.shape[0], device=x.device)
            assert torch.equal(input_hash, out_hash), "Row order mismatch!"
        return x

    def encode(self, x, cluster_labels):
        x = self.route_through_layers(
            x, cluster_labels,
            self.layer_order_enc,
            self.encoder_layers,
            self.cluster_encoder_layers
        )
        if self.latent_shared:
            mu = self.fc_mu(x)
            logvar = self.fc_logvar(x)
        else:
            mu = torch.empty(x.size(0), self.latent_dim,
                             device=x.device, dtype=x.dtype)
            logvar = torch.empty_like(mu)
            for c in range(self.num_clusters):
                mask = (cluster_labels == c)
                if mask.any():
                    mu[mask] = self.cluster_fc_mu[str(c)](x[mask])
                    logvar[mask] = self.cluster_fc_logvar[str(c)](x[mask])
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, cluster_labels):
        z = self.route_through_layers(
            z, cluster_labels,
            self.layer_order_dec,
            self.decoder_layers,
            self.cluster_decoder_layers
        )
        if self.output_shared:
            return self.final_layer(z)
        else:
            outputs = []
            for c in range(self.num_clusters):
                mask = (cluster_labels == c)
                if mask.any():
                    z_c = z[mask]
                    z_out = self.cluster_final_layer[str(c)](z_c)
                    outputs.append((mask, z_out))
            out_dim = outputs[0][1].shape[1]
            output = torch.empty(z.shape[0], out_dim,
                                 device=z.device,
                                 dtype=z.dtype)
            for mask, z_out in outputs:
                output[mask] = z_out
            return output

    def forward(self, x, cluster_labels):
        if self.debug:
            print(f"[DEBUG] Forward start: {x.shape}")
        mu, logvar = self.encode(x, cluster_labels)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, cluster_labels)
        if self.debug:
            print(f"[DEBUG] Forward end: {recon.shape}")
        return recon, mu, logvar

    def __repr__(self):
        lines = [f"CISSVAE(input_dim={self.input_dim}, latent_dim={self.latent_dim}, "
                 f"latent_shared={self.latent_shared}, num_clusters={self.num_clusters})"]
        lines.append("Encoder Layers:")
        in_dim = self.input_dim
        for i, (out_dim, lt) in enumerate(zip(self.hidden_dims, self.layer_order_enc)):
            lines.append(f"  [{i}] {lt.upper():<8} {in_dim} → {out_dim}")
            in_dim = out_dim
        lines.append("\nLatent Layer:")
        if self.latent_shared:
            lines.append(f"  SHARED    {in_dim} → {self.latent_dim}")
        else:
            for c in range(self.num_clusters):
                lines.append(f"  UNSHARED (cluster {c}) {in_dim} → {self.latent_dim}")
        lines.append("\nDecoder Layers:")
        hidden_rev = self.hidden_dims[::-1]
        in_dim = self.latent_dim
        for i, (out_dim, lt) in enumerate(zip(hidden_rev, self.layer_order_dec)):
            lines.append(f"  [{i}] {lt.upper():<8} {in_dim} → {out_dim}")
            in_dim = out_dim
        lines.append(f"\nFinal Output Layer: {in_dim} → {self.input_dim}")
        return "\n".join(lines)
