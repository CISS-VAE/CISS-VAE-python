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
    and replaces NaNs (including held-out values) with ``replacement_value``.

    :param data: Input matrix of shape ``(n_samples, n_features)``. May contain NaNs.
    :type data: pandas.DataFrame | numpy.ndarray | torch.Tensor

    :param cluster_labels: Cluster assignment per sample (length ``n_samples``). If ``None``,
        all rows are assigned to a single cluster ``0``.
    :type cluster_labels: array-like or None

    :param val_proportion: Per-cluster fraction of **non-missing** entries to hold out for validation.
        Accepted forms:

        * float in ``[0, 1]``: same fraction for all clusters
        * sequence (length = number of clusters): aligned to ``sorted(unique(cluster_labels))``
        * mapping (e.g. ``{cluster_id: fraction}``) covering all clusters
        * pandas.Series indexed by cluster IDs covering all clusters
        
    :type val_proportion: float | collections.abc.Sequence | collections.abc.Mapping | pandas.Series

    :param replacement_value: Value used to fill missing and held-out entries after masking.
    :type replacement_value: float

    :param columns_ignore: Columns to exclude from validation masking. Use column names for DataFrame
        and indices otherwise.
    :type columns_ignore: list[str | int] or None

    :param imputable: Matrix indicating which entries should be excluded from imputation
        (1 = impute, 0 = exclude). Must have the same shape as ``data``.
    :type imputable: pandas.DataFrame | numpy.ndarray | torch.Tensor

    :param binary_feature_mask: Boolean vector of length ``n_features`` indicating binary columns.
        Used to construct ``activation_groups``. Categorical dummy columns must also be marked as True.
    :type binary_feature_mask: list[bool] | numpy.ndarray

    :param categorical_column_map: Optional mapping from original categorical variable names to
        their corresponding dummy-variable columns. Example::

            {"C1": ["C1b1", "C1b2"], "C2": ["C2b1", "C2b2"]}

        These columns are grouped together in ``activation_groups`` and treated as categorical variables.
        All listed columns must also be marked as True in ``binary_feature_mask``.
    :type categorical_column_map: dict[str, list[str | int]] or None


    :ivar raw_data: Original data converted to float tensor (NaNs preserved).
    :vartype raw_data: torch.FloatTensor

    :ivar data: Normalized data with NaNs replaced by ``replacement_value``.
    :vartype data: torch.FloatTensor

    :ivar masks: Boolean mask where ``True`` indicates observed (non-NaN) entries before replacement.
    :vartype masks: torch.BoolTensor

    :ivar val_data: Tensor containing only validation-held-out values (others are NaN).
    :vartype val_data: torch.FloatTensor

    :ivar cluster_labels: Cluster ID for each row.
    :vartype cluster_labels: torch.LongTensor

    :ivar indices: Original row indices (from DataFrame index or ``arange`` for arrays/tensors).
    :vartype indices: torch.LongTensor

    :ivar feature_names: Column names (from DataFrame) or synthetic names (``V1``, ``V2``, ...).
    :vartype feature_names: list[str]

    :ivar n_clusters: Number of unique clusters.
    :vartype n_clusters: int

    :ivar shape: Shape of ``self.data`` as ``(n_samples, n_features)``.
    :vartype shape: tuple[int, int]

    :ivar binary_feature_mask: Boolean mask indicating binary features.
    :vartype binary_feature_mask: numpy.ndarray

    :ivar activation_groups: Mapping of feature groups to column indices. Structure::

            {
                "continuous": [int, ...],
                "binary": [int, ...],
                "<categorical_name>": [int, ...],
                ...
            }

        * "continuous": indices of continuous-valued features
        * "binary": indices of binary features
        * Each additional key corresponds to a grouped categorical variable

        This structure is used for loss computation, imputation, and validation logic.
    :vartype activation_groups: dict


    :raises TypeError: If ``data`` or ``cluster_labels`` are invalid types, or if
        ``val_proportion`` is not a supported type.

    :raises ValueError: If any proportion is outside ``[0, 1]``, or if cluster coverage
        is incomplete, or sequence lengths do not match number of clusters.


    .. note::

        * Normalization uses column-wise mean and standard deviation computed from
        observed values after validation masking.
        * Zero standard deviations are replaced with 1 to avoid division by zero.
        * Feature types are resolved into ``activation_groups`` and used throughout
        training, loss computation, and imputation.
    """
    def __init__(
        self, 
        data, 
        cluster_labels, 
        val_proportion = 0.1, 
        replacement_value = 0, 
        columns_ignore = None, 
        imputable = None, 
        val_seed = 42, 
        binary_feature_mask = None,
        categorical_column_map = None,):
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
        :param imputable: Optional Matrix showing which data entries to exclude from imputation (1 for impute, 0 for exclude from imputation), shape ``(n_samples, n_features)``. Should be same shape as ``data``. 
        :type imputable: pandas.DataFrame | numpy.ndarray | torch.Tensor, optional
        :param val_seed: Optional (default 42), seed for random number generator for selecting validation dataset
        :type val_seed: int
        :param binary_feature_mask: 1D bool vector of length 'input_dim' -> true if column is binary.
        :type binary_feature_mask: list[bool]
        :param categorical_column_map: Optional dictionary where keys are original categories and values are resulting dummy variables. Must set binary_feature_mask if using!
        :type categorical_column_map: dict

        """

        ## set seed for selecting valdata
        self.val_seed = val_seed
        self._rng = np.random.default_rng(self.val_seed)

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
            
        ## if categorical_column_map is used and bfm not set, give error 
        if categorical_column_map is not None and binary_feature_mask is None:
            raise RuntimeError("binary_feature_mask required to use categorical_column_map")

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
            # - ignored columns -> if not numeric become float column filled with NaN (kept in shape, never used)
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

        ## added check for binary feature mask matches number of features
        if self.binary_feature_mask is not None:
            if len(self.binary_feature_mask) != raw_data_np.shape[1]:
                raise ValueError("binary_feature_mask must match number of features")

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
        ## Setting unique clusters once in a deterministic way!
        self.unique_clusters = np.sort(np.unique(cluster_labels_np))
        self.cluster_labels = torch.tensor(cluster_labels_np, dtype=torch.long)

        self.n_clusters = len(np.unique(cluster_labels_np))
        # unique_clusters = np.unique(cluster_labels_np)

        # =========================================
        # VALIDATION BEGINS
        # - need to separate columns in categorical_column_map from all the others
        # - for others, do validation extraction normally
        # - for catcols do validation holdout by cat (holdout all columns of that cat in row)
        # =========================================

        # --------------------------
        # Resolve per-cluster validation proportion
        # --------------------------
        def _as_per_cluster_props(vp):
            # scalar → broadcast
            if isinstance(vp, (int, float, np.floating)):
                p = float(vp)
                if not (0 <= p <= 1):
                    raise ValueError("`val_proportion` scalar must be in [0, 1].")
                return {c: p for c in self.unique_clusters}

            # pandas Series with labeled index
            if isinstance(vp, pd.Series):
                mapping = {int(k): float(v) for k, v in vp.items()}
                missing = [c for c in self.unique_clusters if c not in mapping]
                if missing:
                    raise ValueError(f"`val_proportion` Series missing clusters: {missing}")
                return mapping

            # Mapping (e.g., dict)
            if isinstance(vp, Mapping):
                mapping = {int(k): float(v) for k, v in vp.items()}
                missing = [c for c in self.unique_clusters if c not in mapping]
                if missing:
                    raise ValueError(f"`val_proportion` mapping missing clusters: {missing}")
                return mapping

            # Sequence aligned to sorted unique clusters
            if isinstance(vp, Sequence):
                vals = list(vp)
                if len(vals) != len(self.unique_clusters):
                    raise ValueError(
                        f"`val_proportion` sequence length ({len(vals)}) must equal number of clusters ({len(self.unique_clusters)})."
                    )
                return {c: float(v) for c, v in zip(self.unique_clusters, vals)}

            raise TypeError(
                "`val_proportion` must be float in [0,1], a sequence (len = #clusters), "
                "a pandas Series (index=cluster), or a mapping {cluster: proportion}."
            )

        per_cluster_prop = _as_per_cluster_props(val_proportion)
        for cid, p in per_cluster_prop.items():
            if not (0.0 <= p <= 1.0):
                raise ValueError(f"`val_proportion` for cluster {cid} must be in [0, 1]; got {p}.")

        # ------------
        # Build validation Units
        # - each non-categorical feature becomes its own unit
        # - each original categorical becomes one grouped unit
        #
        # Example:
        #   validation_units["C1"] = {"kind": "categorical", "cols": [4, 5]}
        # -------------

        # ============================================================
        # Resolve categorical groups with FULL validation
        # ============================================================
        self.categorical_column_map = categorical_column_map
        self.categorical_group_indices = {}

        def _resolve_col_to_index(col_id):
            """
            Convert column identifier → integer index
            Supports:
            - column names (str)
            - integer indices
            """
            if isinstance(col_id, str):
                if col_id not in self.feature_names:
                    raise ValueError(
                        f"Column '{col_id}' from categorical_column_map not found in data."
                    )
                return self.feature_names.index(col_id)

            elif isinstance(col_id, (int, np.integer)):
                col_id = int(col_id)
                if not (0 <= col_id < len(self.feature_names)):
                    raise ValueError(
                        f"Column index {col_id} from categorical_column_map is out of bounds."
                    )
                return col_id

            else:
                raise TypeError(
                    "categorical_column_map entries must be column names (str) or integer indices."
                )

        if categorical_column_map is not None:

            if not isinstance(categorical_column_map, dict):
                raise TypeError(
                    "`categorical_column_map` must be a dictionary like "
                    "{'C1': ['C1b1','C1b2'], ...}"
                )

            used_cols = set()

            for cat_name, dummy_cols in categorical_column_map.items():

                # ---- must be non-empty sequence ----
                if not isinstance(dummy_cols, (list, tuple)):
                    raise TypeError(
                        f"Value for category '{cat_name}' must be a list/tuple of dummy columns."
                    )

                if len(dummy_cols) == 0:
                    raise ValueError(
                        f"Category '{cat_name}' cannot have empty dummy column list."
                    )

                # ---- resolve to indices ----
                resolved = [_resolve_col_to_index(c) for c in dummy_cols]

                # ---- check duplicates within category ----
                if len(set(resolved)) != len(resolved):
                    raise ValueError(
                        f"Duplicate dummy columns found within category '{cat_name}'."
                    )

                # ---- check overlap across categories ----
                overlap = used_cols.intersection(resolved)
                if overlap:
                    overlap_names = [self.feature_names[i] for i in sorted(overlap)]
                    raise ValueError(
                        f"Dummy columns {overlap_names} appear in more than one category."
                    )

                # ---- check binary mask correctness ----
                for idx in resolved:
                    if self.binary_feature_mask is None or not self.binary_feature_mask[idx]:
                        raise ValueError(
                            f"Dummy column '{self.feature_names[idx]}' is listed in categorical_column_map but is not marked True in binary_feature_mask." 
                            f"'{self.feature_names[idx]}' must be binary and marked True"
                            "in binary_feature_mask."
                        )

                # ---- IMPORTANT CHANGE ----
                # DO NOT block overlap with columns_ignore anymore

                self.categorical_group_indices[cat_name] = resolved
                used_cols.update(resolved)

        # ============================================================
        # Build activation_groups (INCLUDES ignored categorical columns)
        # ============================================================
        dummy_cols = set()
        for g in self.categorical_group_indices.values():
            dummy_cols.update(g)

        self.activation_groups = {"binary": [], "continuous": []}

        for i in range(len(self.feature_names)):
            if i in dummy_cols:
                continue

            is_binary = (
                False if self.binary_feature_mask is None
                else self.binary_feature_mask[i]
            )

            if is_binary:
                self.activation_groups["binary"].append(i)
            else:
                self.activation_groups["continuous"].append(i)

        for k, cols in self.categorical_group_indices.items():
            self.activation_groups[k] = list(cols)

        # ==========================================================
        # Ensure ALL columns are covered exactly once
        # ==========================================================
        all_grouped_cols = set()
        for cols in self.activation_groups.values():
            all_grouped_cols.update(cols)

        expected_cols = set(range(len(self.feature_names)))

        missing = expected_cols - all_grouped_cols
        extra = all_grouped_cols - expected_cols

        if missing:
            raise RuntimeError(
                f"The following columns are missing from activation_groups: "
                f"{[self.feature_names[i] for i in sorted(missing)]}"
            )

        if extra:
            raise RuntimeError(
                f"Invalid column indices found in activation_groups: {sorted(extra)}"
            )

        seen = set()
        for name, cols in self.activation_groups.items():
            overlap = seen.intersection(cols)
            if overlap:
                raise RuntimeError(
                    f"Columns appear in multiple activation groups: {overlap}"
                )
            seen.update(cols)

        # ============================================================
        # Build validation_units (EXCLUDES ignored columns)
        # ============================================================
        self.validation_units = {}

        for i, name in enumerate(self.feature_names):
            if i in self.ignore_indices:
                continue
            if i in dummy_cols:
                continue

            is_binary = (
                False if self.binary_feature_mask is None
                else self.binary_feature_mask[i]
            )

            if is_binary:
                self.validation_units[name] = {"kind": "binary", "cols": [i]}
            else:
                self.validation_units[name] = {"kind": "continuous", "cols": [i]}

        ignore_set = set(self.ignore_indices)

        for k, cols in self.categorical_group_indices.items():
            cols_set = set(cols)
            ignored_in_group = cols_set.intersection(ignore_set)

            if len(ignored_in_group) == 0:
                self.validation_units[k] = {"kind": "categorical", "cols": list(cols)}

            elif len(ignored_in_group) == len(cols_set):
                # all ignored → skip entirely
                continue

            else:
                bad_cols = [self.feature_names[i] for i in sorted(ignored_in_group)]
                raise ValueError(
                    f"Categorical group '{k}' has partially ignored columns: {bad_cols}. "
                    "Either ignore ALL dummy columns for this category or NONE."
                )
            

        # ============================================================
        # Validation masking 
        # ============================================================
        val_mask_np = np.zeros_like(raw_data_np, dtype=bool)

        for cid in self.unique_clusters:
            rows = np.where(cluster_labels_np == cid)[0]
            if rows.size == 0:
                continue

            prop = per_cluster_prop[cid]# prop = val_proportion if isinstance(val_proportion, float) else val_proportion[cid]

            cluster_data = raw_data_np[rows]

            for unit_name, info in self.validation_units.items():
                cols = info["cols"]
                kind = info["kind"]

                if kind in ["binary", "continuous"]:
                    col = cols[0]
                    valid = ~np.isnan(cluster_data[:, col])

                    # ----------------------------------------
                    # Exclude DNI entries (DO NOT IMPUTE)
                    # ----------------------------------------
                    if dni_np is not None: 
                        valid = valid & (dni_np[rows, col] == 1)

                    idxs = np.where(valid)[0]

                    if len(idxs) == 0:
                        continue

                    n_val = int(len(idxs) * prop)
                    if n_val == 0 and prop > 0:
                        n_val = 1
                    chosen = self._rng.choice(idxs, size=n_val, replace=False)

                    val_mask_np[rows[chosen], col] = True

                elif kind == "categorical":
                    group = np.array(cols)
                    valid = np.all(~np.isnan(cluster_data[:, group]), axis=1)

                    if dni_np is not None:
                        valid = valid & (dni_np[rows][:, group].all(axis=1))
                    idxs = np.where(valid)[0]

                    if len(idxs) == 0:
                        continue

                    n_val = int(len(idxs) * prop)
                    if n_val == 0 and prop > 0:
                        n_val = 1
                    chosen = self._rng.choice(idxs, size=n_val, replace=False)

                    val_mask_np[np.ix_(rows[chosen], group)] = True

        # =========================================
        # END VALIDATION CHANGES 
        # =========================================

        val_mask_tensor = torch.tensor(val_mask_np, dtype=torch.bool)
        ## val_mask is a tensor
        self.val_mask = val_mask_tensor

        # ----------------------------------------
        # Set aside val_data
        # ----------------------------------------
        self.val_data = self.raw_data.clone()
        self.val_data[~val_mask_tensor] = torch.nan  # keep only validation-masked values

        if len(self.ignore_indices) > 0:
            ignore_idx = torch.tensor(self.ignore_indices, dtype=torch.long)
            self.val_data[:, ignore_idx] = torch.nan

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
        self.feature_means = np.nan_to_num(self.feature_means, nan=0.0)
        self.feature_stds = np.nan_to_num(self.feature_stds, nan=1.0)

        zero_std_idx = np.where(self.feature_stds == 0)[0]
        if zero_std_idx.size > 0:
            bad_feats = [self.feature_names[i] for i in zero_std_idx]
            print(
                f"[Warning] {len(zero_std_idx)} feature(s) had zero std after masking. "
                f"Replaced with 1.0 to avoid div-by-zero. "
                f"Features: {bad_feats}"
    )

        self.feature_stds[self.feature_stds == 0] = 1.0  # avoid division by zero

        
        ## improved handling of bfm
        if self.binary_feature_mask is not None:
            norm_data_cont = (data_np - self.feature_means) / self.feature_stds
            bfm_mask = self.binary_feature_mask.astype(bool)
            norm_data_np = data_np * bfm_mask + norm_data_cont * (~bfm_mask)

        else:
            ## Normalize (in-place)
            norm_data_np = (data_np - self.feature_means) / self.feature_stds


        self.data = torch.tensor(norm_data_np, dtype=torch.float32)

        # ----------------------------------------
        # Track missing & replace with value
        # ----------------------------------------
        self.masks = ~torch.isnan(self.data) ## true where value not na
        self.data = torch.where(
            self.masks, 
            self.data, 
            torch.tensor(replacement_value, dtype=torch.float32)
            )
        self.shape = self.data.shape

    def get_activation_groups(self, exclude_ignored: bool = False):
        """
        Return activation groups, optionally excluding ignored columns.

        Parameters
        ----------
        exclude_ignored : bool
            If True, removes columns listed in columns_ignore.

        Returns
        -------
        dict
            Filtered activation groups with ignored columns removed.
        """

        # --------------------------------------------------
        # 1. Fast path: no filtering needed
        # --------------------------------------------------
        if not exclude_ignored:
            return self.activation_groups

        # --------------------------------------------------
        # 2. Convert ignore list to set (O(1) lookup)
        # --------------------------------------------------
        ignore_set = set(self.ignore_indices)

        filtered = {}

        # --------------------------------------------------
        # 3. Filter each group
        # --------------------------------------------------
        for name, cols in self.activation_groups.items():

            # Remove ignored columns
            kept = [c for c in cols if c not in ignore_set]

            # Only keep non-empty groups
            if len(kept) > 0:
                filtered[name] = kept

        # --------------------------------------------------
        # 4. SAFETY CHECK (prevents silent bugs)
        # --------------------------------------------------
        # Ensure no ignored columns leaked through
        for name, cols in filtered.items():
            overlap = set(cols).intersection(ignore_set)
            if overlap:
                raise RuntimeError(
                    f"BUG: ignored columns still present in activation group '{name}': {overlap}"
                )

        return filtered

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

        if hasattr(self, "validation_units"):
            unit_summary = {
                k: {"kind": v["kind"], "cols": [self.feature_names[i] for i in v["cols"]]}
                for k, v in self.validation_units.items()
            }
            out += f"\n  • Validation units: {unit_summary}"

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


    


