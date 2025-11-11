from __future__ import annotations
from typing import Dict, Optional, List, Any, Union, Set
import re
import numpy as np
import pandas as pd


class MissingnessMatrix:
    """A matrix with missingness proportions and metadata."""

    def __init__(
        self,
        data: np.ndarray,
        feature_columns_map: Dict[str, List[str]],
        feature_names: List[str],
        sample_names: Optional[List[str]] = None,
    ):
        self.data = data
        self.feature_columns_map = feature_columns_map
        self.feature_names = feature_names
        self.sample_names = sample_names or list(range(len(data)))

    @property
    def shape(self):
        """Return (n_samples, n_features)."""
        return self.data.shape

    def __getitem__(self, key):
        """Index into the underlying array."""
        return self.data[key]

    def __array__(self):
        """Allow NumPy ops directly on this object."""
        return self.data

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame with preserved names."""
        return pd.DataFrame(self.data, columns=self.feature_names, index=self.sample_names)

    def to_numpy(self, dtype=None, copy: bool = False) -> np.ndarray:
        """Return the underlying NumPy array (optionally cast/copied)."""
        arr = self.data
        if dtype is not None and arr.dtype != dtype:
            arr = arr.astype(dtype, copy=False)
        if copy:
            arr = arr.copy()
        return arr

    def __repr__(self) -> str:
        """Full string representation (no preview/truncation)."""
        return str(self.to_dataframe())

    def __str__(self) -> str:
        """Full string representation (no preview/truncation)."""
        return str(self.to_dataframe())

    def head(self):
        return(self.to_dataframe().head())


def create_missingness_prop_matrix(
    data: Union[pd.DataFrame, np.ndarray],
    index_col: Optional[str] = None,
    cols_ignore: Optional[List[str]] = None,
    na_values: Optional[List[Any]] = None,
    repeat_feature_names: Optional[List[str]] = None,
    timepoint_prefix: Optional[str] = None,
    nonint_timepoint: bool = False,
    column_mapping: Optional[Dict[str, List[str]]] = None,
    loose = False,
) -> MissingnessMatrix:
    """
    Create a missingness proportion matrix summarizing feature-level missingness per sample.

    Computes the proportion of missing values for each feature within each sample,
    optionally aggregating repeated measurements (e.g., ``feature_t1``, ``feature_t2``).
    Can also accept an explicit ``column_mapping`` from base feature → list of columns.

    :param data: Input dataset (coercible to DataFrame).
    :type data: pandas.DataFrame or numpy.ndarray
    :param index_col: Optional column to use as sample index in the output metadata (not scored).
    :type index_col: str or None, optional
    :param cols_ignore: Columns to exclude from scoring (e.g., IDs, non-features).
    :type cols_ignore: list[str] or None, optional
    :param na_values: Extra values to treat as missing (in addition to NaN/None/±Inf).
    :type na_values: list[Any] or None, optional
    :param repeat_feature_names: Base feature names that have repeated timepoints to be aggregated.
                                 Columns matched by regex pattern:
                                 - if ``timepoint_prefix`` is provided: ``^<feat>_<prefix>\\d+$``
                                 - else: ``^<feat>_\\d+$``
    :type repeat_feature_names: list[str] or None, optional
    :param timepoint_prefix: Optional prefix that appears before the timepoint integer, e.g., ``t`` to match ``feat_t1``.
    :type timepoint_prefix: str or None, optional
    :param nonint_timepoint: If true, any text after '_' will count as timepoint (eg Baseline).
    :type nonint_timepoint: bool, optional
    :param column_mapping: Explicit mapping { base_feature: [col1, col2, ...] } to aggregate. Takes precedence.
    :type column_mapping: dict[str, list[str]] or None, optional
    :param loose: If true, will match any text starting with the base feature names in `repeat_feature_names`.
    :type loose: bool

    :returns: MissingnessMatrix with:
              - ``data``: (n_samples, n_features) matrix of missingness proportions
              - ``feature_columns_map``: mapping of base features → contributing columns
              - ``to_dataframe()`` to view as DataFrame
    :rtype: MissingnessMatrix
    """
    # -------------------------------
    # 1) Validate & normalize inputs
    # -------------------------------
    if not isinstance(data, (pd.DataFrame, np.ndarray)):
        raise ValueError("`data` must be a pandas DataFrame or numpy.ndarray.")

    # Coerce to DataFrame (no in-place mutation of original)
    df = pd.DataFrame(data).copy() if isinstance(data, np.ndarray) else data.copy()

    # Normalize column names to strings (prevents regex/type issues)
    df.columns = df.columns.astype(str)

    # Defaults
    cols_ignore = [] if cols_ignore is None else list(cols_ignore)
    repeat_feature_names = [] if repeat_feature_names is None else list(repeat_feature_names)
    # pandas.isna already covers None/NaN; we also treat ±Inf as missing
    na_values = [np.inf, -np.inf] if na_values is None else list(na_values)

    # Validate basic types
    if index_col is not None and not isinstance(index_col, str):
        raise ValueError("`index_col` must be None or a string.")
    if not isinstance(cols_ignore, list):
        raise ValueError("`cols_ignore` must be a list or None.")
    if not isinstance(repeat_feature_names, list):
        raise ValueError("`repeat_feature_names` must be a list or None.")
    if column_mapping is not None and not isinstance(column_mapping, dict):
        raise ValueError("`column_mapping` must be a dict or None.")

    # -------------------------------
    # 2) Sample names & drop columns
    # -------------------------------
    # Determine sample names (prefer explicit index_col if present)
    if index_col is not None and index_col in df.columns:
        sample_names = df[index_col].astype(str).tolist()
    else:
        # fallback: use DataFrame index if useful; otherwise None → class will auto-range
        sample_names = df.index.astype(str).tolist() if hasattr(df, "index") else None

    # Columns excluded from scoring
    cols_to_drop: Set[str] = set()
    if index_col is not None and index_col in df.columns:
        cols_to_drop.add(index_col)
    for c in cols_ignore:
        if c in df.columns:
            cols_to_drop.add(c)

    # Candidate feature columns (post-exclusions)
    all_cols = list(df.columns)
    feature_candidate_cols = [c for c in all_cols if c not in cols_to_drop]
    if not feature_candidate_cols:
        raise ValueError("After excluding `index_col` and `cols_ignore`, no feature columns remain.")

    # --------------------------------------------------
    # 3) Build feature → columns mapping
    #     precedence: column_mapping (explicit) > repeat_feature_names (regex) > singletons
    # --------------------------------------------------
    feature_to_cols: Dict[str, List[str]] = {}
    consumed_cols: Set[str] = set()

    # (A) explicit mapping takes precedence
    if column_mapping:
        # validate columns exist
        missing = {base: [c for c in cols if c not in df.columns] for base, cols in column_mapping.items()}
        missing = {k: v for k, v in missing.items() if v}
        if missing:
            raise ValueError(f"`column_mapping` refers to missing columns: {missing}")
        # adopt mapping (preserve key order)
        for base, cols in column_mapping.items():
            cols_list = [str(c) for c in cols]
            feature_to_cols[base] = cols_list
            consumed_cols.update(cols_list)

    # (B) repeated features by regex (only those not already covered by mapping)
    if repeat_feature_names:
        for feat in repeat_feature_names:
            if feat in feature_to_cols:
                # already defined via mapping; skip regex collection for this base
                continue
            feat_escaped = re.escape(feat)
            if nonint_timepoint:
                pattern = rf"^{feat_escaped}_[A-Za-z0-9]+$"
            elif timepoint_prefix:
                pattern = rf"^{feat_escaped}_{re.escape(timepoint_prefix)}\d+$"
            elif loose:
                pattern = rf"^{feat_escaped}"
            else:
                pattern = rf"^{feat_escaped}_\d+$"
            matching_cols = [c for c in feature_candidate_cols if re.match(pattern, c)]
            if not matching_cols:
                raise ValueError(
                    f"No columns found for repeated feature '{feat}' using pattern '{pattern}'. "
                    f"Ensure columns look like '{feat}_1', '{feat}_2', ... (or '{feat}_{timepoint_prefix}1', ...)."
                )
            feature_to_cols[feat] = matching_cols
            consumed_cols.update(matching_cols)

    # (C) remaining single columns become their own features
    for c in feature_candidate_cols:
        if c not in consumed_cols:
            feature_to_cols[c] = [c]

    # Final feature order:
    # - preserve dict insertion order (mapping keys first, then repeats, then singletons)
    out_features: List[str] = list(feature_to_cols.keys())
    if not out_features:
        raise ValueError("No features to score after processing mapping and repeats.")

    # -------------------------------------------
    # 4) Missingness checker (vectorized friendly)
    # -------------------------------------------
    def is_missing(arr_like: Union[pd.Series, np.ndarray]) -> np.ndarray:
        """
        Return boolean mask where values are considered missing:
        - pandas.isna (NaN/None)
        - ±Inf
        - any user-specified values in `na_values`
        """
        x = arr_like.values if isinstance(arr_like, pd.Series) else np.asarray(arr_like)
        miss = pd.isna(x)

        # include ±Inf as missing
        with np.errstate(all="ignore"):
            miss |= np.isinf(pd.to_numeric(x, errors="coerce"))

        # include any explicit values
        for na in na_values:
            try:
                miss |= (x == na)
            except Exception:
                # Comparisons may fail for mixed dtypes; ignore safely
                pass
        return miss

    # -------------------------------------------
    # 5) Compute per-sample missingness proportion
    # -------------------------------------------
    n_samples = len(df)
    n_features = len(out_features)
    out = np.full((n_samples, n_features), np.nan, dtype=float)

    for j, feat in enumerate(out_features):
        cols = feature_to_cols[feat]
        subdf = df[cols]
        # boolean matrix: True where missing
        miss_matrix = subdf.apply(is_missing, axis=0)
        # mean across timepoints/columns → per-row proportion missing for this feature
        prop_missing = miss_matrix.mean(axis=1).to_numpy(dtype=float)
        out[:, j] = prop_missing

    # -------------------------------------------
    # 6) Package result
    # -------------------------------------------
    return MissingnessMatrix(
        data=out,
        feature_columns_map=feature_to_cols,
        feature_names=out_features,
        sample_names=sample_names,
    )
