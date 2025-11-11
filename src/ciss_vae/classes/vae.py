r"""
Variational Autoencoder with cluster‑aware shared/unshared layers.

This module defines :class:`CISSVAE`, a VAE that can route samples through
either **shared** or **cluster‑specific** (unshared) layers in the encoder and
decoder, controlled by per‑layer directives. For binary features, a sigmoid activation function is applied at the end to get probabilities.

"""

import torch
import torch.nn as nn
from typing import Iterable, Optional, Sequence, Union

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
                 binary_feature_mask: Optional[Union[torch.Tensor, Sequence[bool]]] = None,
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
        :param binary_feature_mask: Boolean vector of length p for n x p dataset. True for binary columns, False for continuous columns
        :type binary_feature_mask: Optional[Union[torch.Tensor, Sequence[bool]]]
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
        # (NEW) Binary feature mask
        # - Added 14 OCT 2025
        # - Create a mask that dictates which features are binary and which are not 
        # - allows for sigmoid at end for binary features only
        # -------------------------
        # We store this as a registered buffer so it follows the model to CUDA/CPU and into checkpoints.
        mask = None
        if binary_feature_mask is not None:
            # Accept torch.Tensor or sequence of bools; validate length
            ## Debug statement - Binary feature mask
            if(self.debug):
                print(f"Binary Feature Mask: {binary_feature_mask}\n\n")
            mask = torch.as_tensor(binary_feature_mask, dtype=torch.bool)
            if mask.ndim != 1 or mask.numel() != input_dim:
                raise ValueError("binary_feature_mask must be a 1D boolean vector of length input_dim.")
        else:
            # Default: if nothing provided, treat no columns as binary until user sets it.
            mask = torch.zeros(input_dim, dtype=torch.bool)

        # register as buffer to track with device / state_dict
        self.register_buffer("binary_mask", mask)  # shape: (input_dim,)

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

    def reparameterize(self, mu, logvar):
        r"""
        Reparameterization trick: ``z = mu + eps * exp(0.5 * logvar)``.

        :param mu: Mean of the approximate posterior.
        :type mu: torch.Tensor, shape ``(batch, latent_dim)``
        :param logvar: Log‑variance of the approximate posterior.
        :type logvar: torch.Tensor, shape ``(batch, latent_dim)``

        :returns: Sampled latent codes ``z``.
        :rtype: torch.Tensor
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
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
        return self._apply_output_activations(logits)

    def forward(self, x, cluster_labels):
        r"""
        Full VAE forward pass: encode → reparameterize → decode.

        :param x: Input batch.
        :type x: torch.Tensor, shape ``(batch, input_dim)``
        :param cluster_labels: Cluster id per row.
        :type cluster_labels: torch.LongTensor, shape ``(batch,)``

        :returns: Tuple ``(recon, mu, logvar)``.
        :rtype: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        """
        # if self.debug:
        #     print(f"[DEBUG] Forward start: {x.shape}")
        mu, logvar = self.encode(x, cluster_labels)
        z = self.reparameterize(mu, logvar)
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

    def get_imputed_valdata(self, dataset, device = "cpu"):
        """ Get denormalized imputed data from the trained model.

        [IMPORTANT!] The validation data is also imputed here! This is not the end result dataframe to use in further analyses. This is for calculating MSE per group/cluster.

    Performs a forward pass of the model in evaluation mode to reconstruct 
    validation data. The reconstructed output is then denormalized using the 
    dataset's feature means and standard deviations. Features with zero 
    standard deviation are replaced with 1.0 to ensure numerical stability.

    :param dataset: Dataset object containing normalized data, validation data, and feature statistics.
        Must include:
          - ``data``: Normalized input tensor of shape ``(n_samples, n_features)``
          - ``val_data``: Original validation data tensor of shape ``(n_samples, n_features)``
          - ``cluster_labels``: Cluster assignments for each sample, shape ``(n_samples,)``
          - ``feature_means``: Per-feature means, length ``n_features``
          - ``feature_stds``: Per-feature standard deviations, length ``n_features``
          - ``feature_names``: List of feature names (used for zero-std warnings)
    :type dataset: ClusterDataset

    :param device: Device to perform computations on (e.g., ``"cpu"`` or ``"cuda"``). Default is ``"cpu"``.
    :type device: str, optional

    :returns: Denormalized reconstructed (imputed) validation data of shape ``(n_samples, n_features)``.
    :rtype: torch.Tensor

    .. note::
        - The model is evaluated using ``self.eval()`` (disables dropout, batchnorm updates, etc.).
        - Features with ``std == 0`` are replaced with ``1.0`` and a warning listing affected features is printed.
        - The returned tensor corresponds to the model's best reconstruction of the validation data after denormalization.

    **Example**::

        >>> vae = CISSVAE(...)
        >>> dataset = ClusterDataset(...)
        >>> imputed_val = vae.get_imputed_valdata(dataset, device="cuda")
        >>> imputed_val.shape
        torch.Size([100, 20])
        """
        self.eval()

        # Get normalized inputs and val data
        full_x = dataset.data.to(device)                       # (N, D), normalized
        full_cluster = dataset.cluster_labels.to(device)       # (N,)
        val_data = dataset.val_data.to(device)                 # (N, D), not normalized
        val_mask = torch.isnan(val_data)                      # (N, D)

        with torch.no_grad():
            recon_x, _, _ = self.forward(full_x, full_cluster)        # normalized output

        # Retrieve per-feature stats
        means = torch.tensor(dataset.feature_means, dtype=torch.float32, device=device)
        stds = torch.tensor(dataset.feature_stds, dtype=torch.float32, device=device)

            # check for zero std features
        zero_std_idx = torch.where(stds == 0)[0]
        if zero_std_idx.numel() > 0:
            bad_feats = [dataset.feature_names[i] for i in zero_std_idx.tolist()]
            print(
                f"[Warning] {len(bad_feats)} feature(s) have std == 0. "
                f"Replaced with 1.0. Features: {bad_feats}"
            )
            stds[zero_std_idx] = 1.0  # safe replacement

        # Denormalize model output
        recon_x_denorm = recon_x * stds + means

        # Ensure float dtype to support NaNs
        recon_x_denorm = recon_x_denorm.to(torch.float32)

        # Blank out non-validation (observed) entries ->. keep only validation reconstructions
        recon_x_denorm[val_mask] = float('nan')

        return(recon_x_denorm)

        # -----------------------------
        # NEW -> handles sigmoid
        # -----------------------------

    def _apply_output_activations(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Applies per-feature output activations:
            - Binary columns (self.binary_mask == True) -> sigmoid
            - Continuous columns -> identity (no activation)

        Args:
            logits: Tensor of shape (batch, input_dim) containing final-layer outputs.

        Returns:
            Tensor of same shape with sigmoid applied only to binary columns.
        """
        # if(self.debug):
        #         print(f"From Apply Output Activations: Binary Feature Mask: {self.binary_mask}\n\n")
        if logits.shape[1] != self.input_dim:
            raise RuntimeError("Output dim mismatch; expected last dim == input_dim.")
        if self.binary_mask is None:
            # Should not happen, but be safe
            return logits

        if not torch.any(self.binary_mask):
            # No binary columns
            return logits

        # Clone only if needed; avoids extra mem if no binary cols.
        out = logits.clone()
        out[:, self.binary_mask] = torch.sigmoid(out[:, self.binary_mask])
        return out

        # -----------------------------
        # NEW -> Add mask after initialization
        # -----------------------------
    @torch.no_grad()
    def set_binary_features(self,
                            mask: Optional[Union[torch.Tensor, Sequence[bool]]] = None,
                            feature_names: Optional[Sequence[str]] = None,
                            binary_feature_names: Optional[Iterable[str]] = None) -> None:
        """
        Update which columns are treated as binary at the output. This function should not be necessary for user to touch.

        You can pass either:
          - mask: 1D bool vector length `input_dim`, or
          - feature_names + binary_feature_names: names → mask is computed

        This is safe to call after loading a model or dataset schema.

        Can set w/ vae.set_binary_features(mask = dataset.binary_feature_mask)

        :param binary_feature_mask: Boolean vector of length p for n x p dataset. True for binary columns, False for continuous columns
        :type binary_feature_mask: Optional[Union[torch.Tensor, Sequence[bool]]]
        :param feature_names: List of all feature names - used with 'binary_feature_names'.
        :type feature_names: Optional[Sequence[str]]
        :param binary_feature_names: List of all binary features (features must also be included in 'feature_names').
        :type binary_feature_names: Optional[Iterable[str]]
        """
        if mask is not None:
            mask = torch.as_tensor(mask, dtype=torch.bool, device=self.binary_mask.device)
            if mask.ndim != 1 or mask.numel() != self.input_dim:
                raise ValueError("mask must be a 1D boolean vector of length input_dim.")
            self.binary_mask.copy_(mask)  # in-place update to keep buffer reference
            return

        if (feature_names is None) or (binary_feature_names is None):
            raise ValueError("Provide either `mask` or (`feature_names` and `binary_feature_names`).")

        feat2idx = {name: i for i, name in enumerate(feature_names)}
        newmask = torch.zeros(self.input_dim, dtype=torch.bool, device=self.binary_mask.device)
        for bname in binary_feature_names:
            if bname not in feat2idx:
                raise ValueError(f"Binary feature name '{bname}' not found in feature_names.")
            newmask[feat2idx[bname]] = True
        self.binary_mask.copy_(newmask)



