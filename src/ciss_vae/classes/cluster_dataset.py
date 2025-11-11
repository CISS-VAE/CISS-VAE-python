"""
Dataset utilities for clustering-aware masking and normalization.

This module defines :class:`ClusterDataset`, a PyTorch :class:`torch.utils.data.Dataset`
that (1) optionally holds out a validation subset of *observed* entries on a
per-cluster basis, (2) normalizes features using statistics computed on the
masked training matrix, and (3) exposes tensors required by the CISS-VAE
training loops: normalized data with missing values filled, cluster labels,
and binary observation masks.

Typical usage::

    ds = ClusterDataset(
        data=df,                       # (N, P) with NaNs for missing
        cluster_labels=clusters,       # length-N array-like
        val_proportion=0.1,            # or per-cluster mapping/sequence
        replacement_value=0.0,
        columns_ignore=["id"]          # columns to exclude from validation masking
    )
"""
from torch.utils.data import Dataset
import torch
import numpy as np
import pandas as pd
import copy
from collections.abc import Mapping, Sequence

class ClusterDataset(Dataset):
    r"""
    Dataset that handles cluster-wise masking and normalization for VAE training.

      1. Optionally holds out a validation subset **per cluster** from *observed*
         (non-NaN) entries according to ``val_proportion``.
      2. Combines original missingness with validation-held-out entries.
      3. Normalizes observed values column-wise (mean/std), keeps masks for NaNs,
         and replaces NaNs (incl. held-out) with ``replacement_value``.

    Parameters
    ----------
    data : pandas.DataFrame | numpy.ndarray | torch.Tensor
        Input matrix, shape ``(n_samples, n_features)``. May contain NaNs.
    cluster_labels : array-like or None
        Cluster assignment per sample (length ``n_samples``). If ``None``,
        all rows are assigned to a single cluster ``0``.
    val_proportion : float | collections.abc.Sequence | collections.abc.Mapping | pandas.Series, default=0.1
        Per-cluster fraction of **non-missing** entries to hold out for validation.

        Accepted forms:
          * **float** in ``[0, 1]``: the same fraction for every cluster.
          * **Sequence** (length ``#clusters``): aligned to ``sorted(unique(cluster_labels))``.
          * **Mapping** (e.g. ``{cluster_id: fraction}``) covering **all** clusters.
          * **pandas.Series** with index = cluster IDs covering **all** clusters.
    replacement_value : float, default=0
        Value to fill missing/held-out entries in ``self.data`` after masking.
    columns_ignore : list[str | int] or None, default=None
        Columns to exclude from validation masking (names for DataFrame, indices otherwise).
    imputable : pandas.DataFrame | numpy.ndarray | torch.Tensor
        Matrix showing which data entries to exclude from imputation (0 for impute, 1 for exclude from imputation), shape ``(n_samples, n_features)``.
        Should be same shape as ``data``. 

    Attributes
    ----------
    self.raw_data : torch.FloatTensor
        Original data converted to float tensor (NaNs preserved).
    self.data : torch.FloatTensor
        Normalized data with NaNs replaced by ``replacement_value``.
    self.masks : torch.BoolTensor
        Boolean mask where ``True`` marks observed (non-NaN) entries **before** replacement.
    self.val_data : torch.FloatTensor
        Tensor containing **only** validation-held-out values (others are NaN).
    self.cluster_labels : torch.LongTensor
        Cluster ID for each row, shape ``(n_samples,)``.
    self.indices : torch.LongTensor
        Original row indices (from DataFrame index or ``arange`` for arrays/tensors).
    self.feature_names : list[str]
        Column names (from DataFrame) or synthetic names (``V1``, ``V2``, ...).
    self.n_clusters : int
        Number of unique clusters discovered from ``cluster_labels``.
    self.shape : tuple[int, int]
        Shape of ``self.data`` (``n_samples``, ``n_features``).
    self.binary_feature_mask : np.array(bool)

    Raises
    ------
    TypeError
        If ``data`` or ``cluster_labels`` are of unsupported types; or if
        ``val_proportion`` is not a float/sequence/mapping/Series.
    ValueError
        If any provided proportion is outside ``[0, 1]``; or a sequence/mapping/Series
        omits required clusters; or a sequence length does not match the number
        of clusters.

    Notes
    -----
    * Normalization uses column-wise mean/std on the **current observed** values
      after validation masking; zero stds are set to 1 to avoid division by zero.
    """
    def __init__(self, data, cluster_labels, val_proportion = 0.1, replacement_value = 0, columns_ignore = None, imputable = None, val_seed = 42, binary_feature_mask = None):
        """Build the dataset, apply per-cluster validation masking, and normalize.
        
        Steps:  
        1. Convert inputs to tensors; preserve indices/column names if a DataFrame.  
        2. Resolve per-cluster validation proportions from ``val_proportion``.  
        3. For each cluster and feature, randomly mark the requested fraction of **observed** entries as validation targets.  
        4. Create ``val_data`` (validation targets only) and training ``data`` where validation entries are set to NaN.  
        5. Compute per-feature mean/std over non-NaN entries in ``data`` and apply normalization; then replace remaining NaNs with ``replacement_value``.
        
        :param data: Input matrix, shape ``(n_samples, n_features)``. May contain NaNs
        :type data: pandas.DataFrame or numpy.ndarray or torch.Tensor
        :param cluster_labels: Cluster assignment per sample (length ``n_samples``). If ``None``, all rows are assigned to a single cluster ``0``
        :type cluster_labels: array-like or None
        :param val_proportion: Per-cluster fraction of **non-missing** entries to hold out for validation, defaults to 0.1
        :type val_proportion: float or collections.abc.Sequence or collections.abc.Mapping or pandas.Series, optional
        :param replacement_value: Value to fill missing/held-out entries in ``self.data`` after masking, defaults to 0
        :type replacement_value: float, optional
        :param columns_ignore: Columns to exclude from validation masking (names for DataFrame, indices otherwise), defaults to None
        :type columns_ignore: list[str or int] or None, optional
        :param imputable: Optional Matrix showing which data entries to exclude from imputation (0 for impute, 1 for exclude from imputation), shape ``(n_samples, n_features)``. Should be same shape as ``data``. 
        :type imputable: pandas.DataFrame | numpy.ndarray | torch.Tensor, optional
        :param val_seed: Optional (default 42), seed for random number generator for selecting validation dataset
        :type val_seed: int
        :param binary_feature_mask: 1D bool vector of length 'input_dim' -> true if column is binary.
        :type binary_feature_mask: list[bool]
        """

        ## set seed for selecting valdata
        self._rng = np.random.default_rng(val_seed)

        ## set columns ignore -> no validation data selected from these columns
        if columns_ignore is None:
            self.columns_ignore = []
        else:
            # If columns_ignore is a pandas Index or Series, convert to list
            if hasattr(columns_ignore, "tolist"):
                self.columns_ignore = columns_ignore.tolist()
            else:
                self.columns_ignore = list(columns_ignore)

        if binary_feature_mask is None:
            self.binary_feature_mask = None
        else:
            self.binary_feature_mask = np.array(binary_feature_mask)

        ## set to one cluster as default!!

        # ----------------------------------------
        # Convert input data to numpy
        # ----------------------------------------
        ## Additions -> check if the index column is non-numeric && give error if there are other non-numeric columns
        if hasattr(data, "iloc"):  # pandas DataFrame
            n_rows, n_cols = data.shape
            self.indices = torch.arange(n_rows, dtype=torch.long)  # safe for any index dtype
            self.feature_names = list(data.columns)

            # Build ignore index list by name
            self.ignore_indices = [i for i, col in enumerate(self.feature_names)
                                if col in self.columns_ignore]

            # Build a numeric matrix column-by-column:
            # - ignored columns -> float column filled with NaN (kept in shape, never used)
            # - non-ignored columns -> must be numeric; error if not
            converted_cols = []
            bad_cols = []

            for j, col in enumerate(self.feature_names):
                s = data[col]
                if j in self.ignore_indices:
                    # If column is numeric, keep as-is; if not, replace with NaN float column
                    if pd.api.types.is_numeric_dtype(s):
                        converted_cols.append(s.astype("float32"))
                    else:
                        converted_cols.append(pd.Series(np.nan, index=s.index, dtype="float32"))
                else:
                    # Must be numeric; coerce and detect non-numeric values (not counting real NaNs)
                    sc = pd.to_numeric(s, errors="coerce")
                    introduced_nonnumeric = (~s.isna()) & (sc.isna())
                    if introduced_nonnumeric.any():
                        bad_cols.append(col)
                    converted_cols.append(sc.astype("float32"))

            if bad_cols:
                raise TypeError(
                    "Non-numeric values found in columns not listed in columns_ignore: "
                    f"{bad_cols}. Convert them to numeric or add them to `columns_ignore`."
                )

            # Stack back to (n_rows, n_cols) float32
            raw_data_np = np.column_stack([c.to_numpy(dtype=np.float32) for c in converted_cols])

        elif isinstance(data, np.ndarray):
            self.indices = torch.arange(data.shape[0], dtype=torch.long)
            self.feature_names = [f"V{i+1}" for i in range(data.shape[1])]
            # Ensure numeric array
            if not np.issubdtype(data.dtype, np.number):
                raise TypeError("ndarray input must be numeric. For mixed types, pass a DataFrame and use columns_ignore.")
            raw_data_np = data.astype(np.float32, copy=False)
            # For ndarray, columns_ignore is by index only
            self.ignore_indices = self.columns_ignore if isinstance(self.columns_ignore, list) else []

        elif isinstance(data, torch.Tensor):
            self.indices = torch.arange(data.shape[0], dtype=torch.long)
            self.feature_names = [f"V{i+1}" for i in range(data.shape[1])]
            if not torch.is_floating_point(data) and not torch.is_complex(data) and not data.dtype.is_floating_point:
                data = data.float()
            raw_data_np = data.cpu().numpy().astype(np.float32, copy=False)
            self.ignore_indices = self.columns_ignore if isinstance(self.columns_ignore, list) else []

        else:
            raise TypeError("Unsupported data format. Must be DataFrame, ndarray, or Tensor.")

        self.raw_data = torch.tensor(raw_data_np, dtype=torch.float32)


        # --------------------
        # Added 'imputable' matrix
        # --------------------

        if imputable is not None:
            if hasattr(imputable, 'iloc'):  # pandas DataFrame
                self.imputable = imputable.values.astype(np.float32)
            elif isinstance(imputable, np.ndarray):
                self.imputable = imputable.astype(np.float32)
            elif isinstance(imputable, torch.Tensor):
                self.imputable = imputable.cpu().numpy().astype(np.float32)
            else:
                raise TypeError("Unsupported imputable matrix format. Must be DataFrame, ndarray, or Tensor.")

            self.imputable = torch.tensor(self.imputable, dtype=torch.int64)
            expected_shape = tuple(self.raw_data.shape)  # (n_samples, n_features)
            if self.imputable.shape != expected_shape:
                raise ValueError(
                    f"`imputable` shape {self.imputable.shape} does not match "
                    f"data shape {expected_shape}."
                )

            dni_np = self.imputable.cpu().numpy().astype(bool)
        else:
            self.imputable = None
            dni_np = None
        

        # ----------------------------------------
        # Cluster labels to numpy
        # ----------------------------------------
        if cluster_labels is None:
            # create a LongTensor of zeros, one per sample
            self.cluster_labels = torch.zeros(self.raw_data.shape[0], dtype=torch.long)
            cluster_labels_np = self.cluster_labels.numpy()
        else: 
            if hasattr(cluster_labels, 'iloc'):
                cluster_labels_np = cluster_labels.values
            elif isinstance(cluster_labels, np.ndarray):
                cluster_labels_np = cluster_labels
            elif isinstance(cluster_labels, torch.Tensor):
                cluster_labels_np = cluster_labels.cpu().numpy()
            else:
                raise TypeError("Unsupported cluster_labels format. Must be Series, ndarray, or Tensor.")
        ## cluster labels stored as torch tensor
        self.cluster_labels = torch.tensor(cluster_labels_np, dtype=torch.long)

        self.n_clusters = len(np.unique(cluster_labels_np))
        unique_clusters = np.unique(cluster_labels_np)

        # --------------------------
        # Resolve per-cluster validation proportion
        # --------------------------
        def _as_per_cluster_props(vp):
            # scalar → broadcast
            if isinstance(vp, (int, float, np.floating)):
                p = float(vp)
                if not (0 <= p <= 1):
                    raise ValueError("`val_proportion` scalar must be in [0, 1].")
                return {c: p for c in unique_clusters}

            # pandas Series with labeled index
            if isinstance(vp, pd.Series):
                mapping = {int(k): float(v) for k, v in vp.items()}
                missing = [c for c in unique_clusters if c not in mapping]
                if missing:
                    raise ValueError(f"`val_proportion` Series missing clusters: {missing}")
                return mapping

            # Mapping (e.g., dict)
            if isinstance(vp, Mapping):
                mapping = {int(k): float(v) for k, v in vp.items()}
                missing = [c for c in unique_clusters if c not in mapping]
                if missing:
                    raise ValueError(f"`val_proportion` mapping missing clusters: {missing}")
                return mapping

            # Sequence aligned to sorted unique clusters
            if isinstance(vp, Sequence):
                vals = list(vp)
                if len(vals) != len(unique_clusters):
                    raise ValueError(
                        f"`val_proportion` sequence length ({len(vals)}) must equal number of clusters ({len(unique_clusters)})."
                    )
                return {c: float(v) for c, v in zip(unique_clusters, vals)}

            raise TypeError(
                "`val_proportion` must be float in [0,1], a sequence (len = #clusters), "
                "a pandas Series (index=cluster), or a mapping {cluster: proportion}."
            )

        per_cluster_prop = _as_per_cluster_props(val_proportion)
        for cid, p in per_cluster_prop.items():
            if not (0.0 <= p <= 1.0):
                raise ValueError(f"`val_proportion` for cluster {cid} must be in [0, 1]; got {p}.")

        # ----------------------------------------
        # Validation mask per cluster
        # ----------------------------------------
        val_mask_np = np.zeros_like(raw_data_np, dtype=bool)

        for cluster_id in unique_clusters:
            row_idxs = np.where(cluster_labels_np == cluster_id)[0]
            if row_idxs.size == 0:
                continue

            cluster_data = raw_data_np[row_idxs]      # shape: (n_rows_in_cluster, n_features)
            prop = per_cluster_prop[cluster_id]
            if prop == 0.0:
                continue  # nothing to select for this cluster

            for col in range(cluster_data.shape[1]):
                if col in self.ignore_indices:
                    continue

                # boolean masks (same length as row_idxs)
                mask_non_missing = ~np.isnan(cluster_data[:, col])
                
                candidate_mask = mask_non_missing 
                candidate_rows = np.where(candidate_mask)[0]  # local indices within row_idxs

                if candidate_rows.size == 0:
                    continue

                ## Put back as floor but will mask at least one value if floor is 0
                if prop > 0:
                    n_val = max(1, int(np.floor(candidate_rows.size * prop)))
                else:
                    continue
                
                if n_val <= 0:
                    continue

                chosen_local = self._rng.choice(candidate_rows, size=n_val, replace=False)
                val_mask_np[row_idxs[chosen_local], col] = True

        val_mask_tensor = torch.tensor(val_mask_np, dtype=torch.bool)
        ## val_mask is a tensor
        self.val_mask = val_mask_tensor

        # ----------------------------------------
        # Set aside val_data
        # ----------------------------------------
        self.val_data = self.raw_data.clone()
        self.val_data[~val_mask_tensor] = torch.nan  # keep only validation-masked values

        # ----------------------------------------
        # Combine true + validation-masked missingness
        # ----------------------------------------
        self.data = self.raw_data.clone()
        self.data[val_mask_tensor] = torch.nan  # mask validation entries


        # ----------------------------------------
        # Normalize non-missing entries
        # ----------------------------------------
        ## Compute mean and std on observed (non-NaN) entries
        data_np = self.data.numpy()
        self.feature_means = np.nanmean(data_np, axis=0)
        self.feature_stds = np.nanstd(data_np, axis=0)

        zero_std_idx = np.where(self.feature_stds == 0)[0]
        if zero_std_idx.size > 0:
            bad_feats = [self.feature_names[i] for i in zero_std_idx]
            print(
                f"[Warning] {len(zero_std_idx)} feature(s) had zero std after masking. "
                f"Replaced with 1.0 to avoid div-by-zero. "
                f"Features: {bad_feats}"
    )

        self.feature_stds[self.feature_stds == 0] = 1.0  # avoid division by zero

        

        if self.binary_feature_mask is not None:
            norm_data_cont = (data_np - self.feature_means) / self.feature_stds
            norm_data_np = data_np * self.binary_feature_mask + norm_data_cont * ~self.binary_feature_mask

        else:
            ## Normalize (in-place)
            norm_data_np = (data_np - self.feature_means) / self.feature_stds


        self.data = torch.tensor(norm_data_np, dtype=torch.float32)

        # ----------------------------------------
        # Track missing & replace with value
        # ----------------------------------------
        self.masks = ~torch.isnan(self.data) ## true where value not na
        self.data = torch.where(self.masks, self.data, torch.tensor(replacement_value, dtype=torch.float32))
        self.shape = self.data.shape

    def __len__(self):
        """
        Number of samples in the dataset.

        :return: ``N`` (number of rows).
        """
        return len(self.data)


    def __getitem__(self, index):
        """
        Get a single sample.

        :param index: Row index.
        :return: Tuple ``(x, cluster_id, mask, original_index)`` where:
            * **x** – normalized input row with NaNs replaced (``(P,)``).
            * **cluster_id** – integer cluster label (``()``).
            * **mask** – boolean mask of observed entries before replacement
            (``(P,)``).
            * **original_index** – original row index from the source DataFrame
            (if provided) or the integer position.
        """
        return (
            self.data[index],            # input with missing replaced
            self.cluster_labels[index], # cluster label
            self.masks[index],          # binary mask
            self.indices[index],         # original row index
        )

    def __repr__(self):
        """Displays the number of samples, features, and clusters, the percentage of missing data before masking, and the percentage of non-missing data held out for validation.
        
        :return: String representation of the dataset
        :rtype: str
        """
        n, p = self.data.shape
        total_values = n * (p-len(self.columns_ignore))

        ## Percent originally missing (before validation mask)
        original_missing = torch.isnan(self.raw_data).sum().item()
        original_missing_pct = 100 * original_missing / total_values

        ## Percent used for validation (out of non-missing entries)
        val_entries = torch.sum(~torch.isnan(self.val_data)).item()  # number of validation-held entries
        val_pct_of_nonmissing = 100 * val_entries / (total_values - original_missing)

        ## Count non-imputable entries (where can_impute == 0)
        non_imputable_count = None
        if hasattr(self, "imputable") and self.imputable is not None:
            non_imputable_count = int((self.imputable == 0).sum().item())        

        ## Build string
        out = (
            f"ClusterDataset(n_samples={n}, n_features={p}, n_clusters={len(torch.unique(self.cluster_labels))})\n"
            f"  • Original missing: {original_missing} / {total_values} "
            f"({original_missing_pct:.2f}%)\n"
            f"  • Validation held-out: {val_entries} "
            f"({val_pct_of_nonmissing:.2f}% of non-missing)\n"
            f"  • .data shape:     {tuple(self.data.shape)}\n"
            f"  • .masks shape:    {tuple(self.masks.shape)}\n"
            f"  • .val_data shape: {tuple(self.val_data.shape)}"
        )
        if non_imputable_count is not None:
            out += f"\n  • Non-imputable entries: {non_imputable_count}"
        return out

    # ----------------------------------------
    # Added copy method
    # ----------------------------------------
    def copy(self):
        """Creates a deep copy of the ClusterDataset method containing all attributes.
        
        :return: Deep copy of the dataset
        :rtype: ClusterDataset
        """
        return copy.deepcopy(self)


    def __str__(self):
        """Displays the number of samples, features, and clusters, the percentage of missing data before masking, and the percentage of non-missing data held out for validation.
        
        :return: String representation of the dataset
        :rtype: str
        """
        return self.__repr__()


    


