r"""
Variational Autoencoder with cluster‑aware shared/unshared layers.

This module defines :class:`CISSVAE`, a VAE that can route samples through
either **shared** or **cluster‑specific** (unshared) layers in the encoder and
decoder, controlled by per‑layer directives. For binary features, a sigmoid activation function is applied at the end to get probabilities.

"""

import torch
import torch.nn as nn
from typing import Iterable, Optional, Sequence, Union
import numpy as np

class CISSVAE(nn.Module):
    r"""
     Clustering-Informed Shared-Structure Variational Autoencoder (CISSVAE).

    Supports flexible mixtures of **shared** and **unshared** layers across
    clusters in both encoder and decoder. Unshared layers are applied by cluster, shared layers are applied to all samples.

    :param input_dim: Number of input features (columns).
    :type input_dim: int
    :param hidden_dims: Width of each hidden layer (encoder goes forward, decoder uses the reverse).
    :type hidden_dims: list[int]
    :param layer_order_enc: Per‑encoder‑layer directive: ``"shared"`` or ``"unshared"``.
    :type layer_order_enc: list[str]
    :param layer_order_dec: Per‑decoder‑layer directive: ``"shared"`` or ``"unshared"``.
    :type layer_order_dec: list[str]
    :param latent_shared: If ``True``, the latent heads (``mu``, ``logvar``) are shared across clusters; otherwise one head per cluster.
    :type latent_shared: bool
    :param latent_dim: Dimensionality of the latent space.
    :type latent_dim: int
    :param output_shared: If ``True``, the final output layer is shared; otherwise one output layer per cluster.
    :type output_shared: bool
    :param num_clusters: Number of clusters present in the data.
    :type num_clusters: int
    :param debug: If ``True``, prints routing shapes and asserts row order invariants.
    :type debug: bool

    :raises ValueError: If an item of ``layer_order_enc`` or ``layer_order_dec`` is not one of
        ``{"shared","unshared","s","u"}`` (case‑insensitive), or if their lengths do not match
        ``len(hidden_dims)`` for the respective path.

    **Expected shapes**
        * Encoder input ``x``: ``(batch, input_dim)``
        * Cluster labels ``cluster_labels``: ``(batch,)`` (``LongTensor`` with values in ``[0, num_clusters-1]``)
        * Decoder/Output: ``(batch, input_dim)``

    **Notes**
        * The decoder consumes ``hidden_dims[::-1]`` (reverse order).
        * Unshared layers maintain per‑cluster ``ModuleList``/``ModuleDict`` replicas.
        * Routing never reorders rows; masks are used to apply cluster‑specific sublayers in‑place.
    """

    def __init__(self,
                 input_dim,
                 hidden_dims,
                 layer_order_enc,
                 layer_order_dec,
                 latent_shared,
                 latent_dim,
                 output_shared,
                 num_clusters,
                 # new optional inputs to define binary features at init time -> udpdate 14OCT2025
                 activation_groups = None,
                 debug=False,):
        """
        Variational Autoencoder supporting flexible shared/unshared layers across clusters.

        :param input_dim: Number of input features.
        :type input_dim: int
        :param hidden_dims: Dimensions of hidden layers.
        :type hidden_dims: list[int]
        :param layer_order_enc: Layer type for each encoder layer (``"shared"`` or ``"unshared"``).
        :type layer_order_enc: list[str]
        :param layer_order_dec: Layer type for each decoder layer (``"shared"`` or ``"unshared"``).
        :type layer_order_dec: list[str]
        :param latent_shared: Whether latent representation is shared across clusters.
        :type latent_shared: bool
        :param latent_dim: Dimensionality of the latent space.
        :type latent_dim: int
        :param output_shared: Whether output layer is shared across clusters.
        :type output_shared: bool
        :param num_clusters: Number of clusters.
        :type num_clusters: int
        :param activation_groups: Defines type of each column for the purpose of matching columns to the correct activation and loss functions. 
        :type activation_groups: dict 
        :param debug: If ``True``, print shape and routing information.
        :type debug: bool
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

        # -------------------------
        # (NEW) Activation groups metadata
        # - Replaces binary_feature_mask
        # - Stores feature structure for downstream use (loss, imputation, validation)
        # -------------------------
        if activation_groups is not None:
            if self.debug:
                print(f"Activation Groups: {activation_groups}\n")

            if not isinstance(activation_groups, dict):
                raise ValueError("activation_groups must be a dictionary.")

            # Validate and normalize
            validated_groups = {}
            for key, cols in activation_groups.items():
                if not isinstance(cols, (list, tuple)):
                    raise ValueError(f"activation_groups['{key}'] must be a list of column indices.")

                clean_cols = []
                for c in cols:
                    if not isinstance(c, (int, np.integer)):
                        raise ValueError(f"Column index '{c}' in group '{key}' is not an integer.")
                    c = int(c)
                    if not (0 <= c < input_dim):
                        raise ValueError(
                            f"Column index {c} in group '{key}' is out of bounds for input_dim={input_dim}."
                        )
                    clean_cols.append(c)

                validated_groups[key] = clean_cols

            # Store as plain attribute (NOT a buffer)
            self.activation_groups = validated_groups

        else:
            # Default: treat everything as continuous if nothing provided
            if self.debug:
                print("No activation_groups provided; defaulting to all continuous.\n")

            self.activation_groups = {
                "continuous": list(range(input_dim))
            }

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
        r"""
        Apply a sequence of shared/unshared layers according to ``layer_type_list``.

        For each position ``i``:
        * if ``layer_type_list[i]`` is shared → apply ``shared_layers[i_shared]`` to all rows;
        * if unshared → for each cluster ``c``, apply the ``c``‑specific layer
            at that depth to the subset of rows where ``cluster_labels == c``.

        :param x: Input activations to be routed.
        :type x: torch.Tensor, shape ``(batch, d_in)``
        :param cluster_labels: Cluster id per row.
        :type cluster_labels: torch.LongTensor, shape ``(batch,)``
        :param layer_type_list: Sequence of ``"shared"``/``"unshared"`` flags (length = number of layers at this stage).
        :type layer_type_list: list[str]
        :param shared_layers: Layers used when the directive is shared (index increases only when a shared layer is consumed).
        :type shared_layers: torch.nn.ModuleList
        :param unshared_layers: Per‑cluster lists of layers for unshared directives (index per cluster increases only when an unshared layer is consumed).
        :type unshared_layers: dict[str, torch.nn.ModuleList] | torch.nn.ModuleDict

        :returns: Routed activations.
        :rtype: torch.Tensor

        :raises ValueError: If an entry in ``layer_type_list`` is invalid or if per‑cluster
            unshared stacks are inconsistent with the directives.
        """
        shared_idx = 0
        unshared_idx = {str(c): 0 for c in range(self.num_clusters)}

        # if self.debug:
        #     input_hash = torch.arange(x.shape[0], device=x.device)
        #     print(f"layer_type_list: {layer_type_list}")
        #     print(f"num_clusters: {self.num_clusters}")
        #     for c in range(self.num_clusters):
        #         print(f"Cluster {c} unshared layers: {len(unshared_layers[str(c)])}")
        #     print(f"Number of unshared layers needed: {layer_type_list.count('unshared')}")

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

        # if self.debug:
        #     out_hash = torch.arange(x.shape[0], device=x.device)
        #     assert torch.equal(input_hash, out_hash), "Row order mismatch!"
        return x

    def encode(self, x, cluster_labels):
        r"""
        Encoder forward pass producing ``mu`` and ``logvar``.

        :param x: Input batch.
        :type x: torch.Tensor, shape ``(batch, input_dim)``
        :param cluster_labels: Cluster id per row.
        :type cluster_labels: torch.LongTensor, shape ``(batch,)``

        :returns: Tuple ``(mu, logvar)``.
        :rtype: tuple[torch.Tensor, torch.Tensor]
        """
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

    def reparameterize(self, mu, logvar, generator=None):
        r"""
        Reparameterization trick: ``z = mu + eps * exp(0.5 * logvar)``.

        :param mu: Mean of the approximate posterior.
        :type mu: torch.Tensor, shape ``(batch, latent_dim)``
        :param logvar: Log‑variance of the approximate posterior.
        :type logvar: torch.Tensor, shape ``(batch, latent_dim)``
        :param generator: Optionl for RNG control (default None)
        :type generator: torch.Generator

        :returns: Sampled latent codes ``z``.
        :rtype: torch.Tensor
        """
        ## Add the generator -> generator owned by training loop
        std = torch.exp(0.5 * logvar)
        if generator is None:
            eps = torch.randn_like(std)
        else:
            eps = torch.randn(
            std.shape,
            device=std.device,
            dtype=std.dtype,
            generator=generator,
        )
        return mu + eps * std

    def decode(self, z, cluster_labels):
        r"""
        Decoder forward pass from latent ``z`` to reconstruction.

        :param z: Latent codes.
        :type z: torch.Tensor, shape ``(batch, latent_dim)``
        :param cluster_labels: Cluster id per row.
        :type cluster_labels: torch.LongTensor, shape ``(batch,)``

        :returns: Reconstructed inputs.
        :rtype: torch.Tensor, shape ``(batch, input_dim)``
        """
        ## 30 sep 2025 -> changed mask to cluster_mask so I can stop getting confused
        z = self.route_through_layers(
            z, cluster_labels,
            self.layer_order_dec,
            self.decoder_layers,
            self.cluster_decoder_layers
        )
        ## final layer is nn.Linear
        # ----------------------------------------
        # 14 OCT 2025 - Add Logic for handling logit sigmoid thingie
        # - gathers the final layers and applies the output activations according to the mask 
        # ----------------------------------------
        if self.output_shared:
            logits = self.final_layer(z)
        else:
            outputs = []
            for c in range(self.num_clusters):
                cluster_mask = (cluster_labels == c)
                if cluster_mask.any():
                    z_c = z[cluster_mask]
                    z_out = self.cluster_final_layer[str(c)](z_c)
                    outputs.append((cluster_mask, z_out))
            out_dim = outputs[0][1].shape[1]
            logits = torch.empty(z.shape[0], out_dim,
                                 device=z.device,
                                 dtype=z.dtype)
            for cluster_mask, z_out in outputs:
                logits[cluster_mask] = z_out
        return logits

    def forward(self, x, cluster_labels, deterministic=False, *, generator = None):
        r"""
        Full VAE forward pass: encode → reparameterize → decode.

        :param x: Input batch.
        :type x: torch.Tensor, shape ``(batch, input_dim)``
        :param cluster_labels: Cluster id per row.
        :type cluster_labels: torch.LongTensor, shape ``(batch,)``
        :param deterministic: Deterministic Evaluation of Model for Imputation (default False)
        :type deterministic: bool
        :param generator: Optionl for RNG control (default None)
        :type torch.Generator

        :returns: Tuple ``(recon, mu, logvar)``.
        :rtype: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        """
        # if self.debug:
        #     print(f"[DEBUG] Forward start: {x.shape}")
        mu, logvar = self.encode(x, cluster_labels)

        if deterministic:
            z = mu
        else: 
            z = self.reparameterize(mu, logvar, generator = generator)

        recon = self.decode(z, cluster_labels)
        # if self.debug:
        #     print(f"[DEBUG] Forward end: {recon.shape}")
        return recon, mu, logvar

    def __repr__(self):
        r"""
        String summary of the architecture (encoder/latent/decoder layout).

        :returns: Human‑readable multi‑line description.
        :rtype: str
        """
        lines = [f"CISSVAE(input_dim={self.input_dim}, latent_dim={self.latent_dim}, "
                 f"latent_shared={self.latent_shared}, output_shared={self.output_shared},"
                 f"num_clusters={self.num_clusters})"]
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
        lines.append("\nFinal Output Layer:")
        if self.output_shared:
            lines.append(f"   SHARED  {in_dim} → {self.input_dim}")
        else: 
            for c in range(self.num_clusters):
                lines.append(f"  UNSHARED (cluster {c}) {in_dim} → {self.input_dim}")
        return "\n".join(lines)

    def __str__(self):
        """Mimics repr"""
        return self.__repr__()
        
    def set_final_lr(self, final_lr):
        """Stores final lr from initial training loop in model attributes to be accessed in refit loop."""
        self.final_lr = final_lr

    def get_final_lr(self):
        """Returns the learning rate stored with self.set_final_lr/"""
        return(self.final_lr)

    def get_imputed_valdata(self, dataset, device="cpu", deterministic=True):
        self.eval()

        full_x = dataset.data.to(device)
        full_cluster = dataset.cluster_labels.to(device)
        val_data = dataset.val_data.to(device)
        val_mask = torch.isnan(val_data)

        with torch.no_grad():
            logits, _, _ = self.forward(full_x, full_cluster, deterministic=deterministic)

        means = torch.tensor(dataset.feature_means, dtype=torch.float32, device=device)
        stds = torch.tensor(dataset.feature_stds, dtype=torch.float32, device=device)

        # handle zero std
        zero_std_idx = torch.where(stds == 0)[0]
        if zero_std_idx.numel() > 0:
            bad_feats = [dataset.feature_names[i] for i in zero_std_idx.tolist()]
            print(f"[Warning] std == 0 → replaced with 1.0: {bad_feats}")
            stds[zero_std_idx] = 1.0

        recon_out = logits.clone().to(torch.float32)

        # --------------------------------------------------
        # USE activation_groups (NOT binary mask)
        # --------------------------------------------------
        for name, cols in dataset.activation_groups.items():

            cols = torch.tensor(cols, device=device)

            # -------------------------
            # CONTINUOUS
            # -------------------------
            if name == "continuous":
                recon_out[:, cols] = logits[:, cols] * stds[cols] + means[cols]

            # -------------------------
            # BINARY
            # -------------------------
            elif name == "binary":
                recon_out[:, cols] = torch.sigmoid(logits[:, cols])

            # -------------------------
            # CATEGORICAL
            # -------------------------
            else:
                probs = torch.softmax(logits[:, cols], dim=1)
                idx = torch.argmax(probs, dim=1)

                recon_out[:, cols] = 0
                recon_out[torch.arange(recon_out.shape[0]), cols[idx]] = 1

        # only keep imputed values
        recon_out[val_mask] = float("nan")

        return recon_out

    @torch.no_grad()
    def set_activation_groups(
        self,
        activation_groups: dict,
    ) -> None:
        """
        Attach feature activation structure to the model.

        This method stores the resolved `activation_groups` dictionary, which defines how
        each input column should be interpreted during loss computation and imputation.

        The model itself does NOT use this information during the forward pass; it is
        stored purely as metadata to ensure consistency between the model, dataset,
        loss functions, and imputation routines.

        Parameters
        ----------
        activation_groups : dict
            Dictionary mapping feature types to column indices. Expected format:

            {
                "continuous": [col_idx, ...],
                "binary": [col_idx, ...],
                "<categorical_name>": [col_idx, ...],
                ...
            }

            - "continuous": indices of continuous-valued features
            - "binary": indices of binary features
            - Each additional key represents a grouped categorical variable
            (multiple columns corresponding to one variable)

        Raises
        ------
        ValueError
            If activation_groups is not a dictionary or contains invalid column indices.

        Notes
        -----
        - This replaces the old `set_binary_features` functionality.
        - The model outputs raw logits for all features; interpretation is handled
        externally using `activation_groups`.
        - This function is primarily used after loading a model to reattach dataset
        structure if needed.

        Example
        -------
        >>> vae.set_activation_groups(dataset.activation_groups)
        """

        if not isinstance(activation_groups, dict):
            raise ValueError("activation_groups must be a dictionary.")

        # Validate structure
        for key, cols in activation_groups.items():
            if not isinstance(cols, (list, tuple)):
                raise ValueError(f"activation_groups['{key}'] must be a list of column indices.")

            for c in cols:
                if not isinstance(c, (int, np.integer)):
                    raise ValueError(f"Column index '{c}' in group '{key}' is not an integer.")
                if not (0 <= int(c) < self.input_dim):
                    raise ValueError(
                        f"Column index {c} in group '{key}' is out of bounds for input_dim={self.input_dim}."
                    )

        # Store as plain attribute (NOT buffer)
        self.activation_groups = {
            k: list(map(int, v)) for k, v in activation_groups.items()
        }
        
    # @torch.no_grad()
    # def set_binary_features(self,
    #                         mask: Optional[Union[torch.Tensor, Sequence[bool]]] = None,
    #                         feature_names: Optional[Sequence[str]] = None,
    #                         binary_feature_names: Optional[Iterable[str]] = None) -> None:
    #     """
    #     Update which columns are treated as binary at the output. This function should not be necessary for user to touch.

    #     You can pass either:
    #       - mask: 1D bool vector length `input_dim`, or
    #       - feature_names + binary_feature_names: names → mask is computed

    #     This is safe to call after loading a model or dataset schema.

    #     Can set w/ vae.set_binary_features(mask = dataset.binary_feature_mask)

    #     :param binary_feature_mask: Boolean vector of length p for n x p dataset. True for binary columns, False for continuous columns
    #     :type binary_feature_mask: Optional[Union[torch.Tensor, Sequence[bool]]]
    #     :param feature_names: List of all feature names - used with 'binary_feature_names'.
    #     :type feature_names: Optional[Sequence[str]]
    #     :param binary_feature_names: List of all binary features (features must also be included in 'feature_names').
    #     :type binary_feature_names: Optional[Iterable[str]]
    #     """
    #     if mask is not None:
    #         mask = torch.as_tensor(mask, dtype=torch.bool, device=self.binary_mask.device)
    #         if mask.ndim != 1 or mask.numel() != self.input_dim:
    #             raise ValueError("mask must be a 1D boolean vector of length input_dim.")
    #         self.binary_mask.copy_(mask)  # in-place update to keep buffer reference
    #         return

    #     if (feature_names is None) or (binary_feature_names is None):
    #         raise ValueError("Provide either `mask` or (`feature_names` and `binary_feature_names`).")

    #     feat2idx = {name: i for i, name in enumerate(feature_names)}
    #     newmask = torch.zeros(self.input_dim, dtype=torch.bool, device=self.binary_mask.device)
    #     for bname in binary_feature_names:
    #         if bname not in feat2idx:
    #             raise ValueError(f"Binary feature name '{bname}' not found in feature_names.")
    #         newmask[feat2idx[bname]] = True
    #     self.binary_mask.copy_(newmask)



