from __future__ import annotations
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union, Callable
import re
import numpy as np
import pandas as pd

def make_missingness_prop_matrix(
    data: pd.DataFrame,
    *,
    # ---- Common ----
    format: str = "wide",  # {"wide", "long"}
    sample_col: Optional[str] = None,  # required for long; optional for wide
    # ---- Long format parameters ----
    feature_col: Optional[str] = None,
    time_col: Optional[str] = None,
    value_col: Optional[str] = None,
    expected_timepoints: Optional[Sequence] = None,
    # ---- Wide format parameters ----
    # Option A: MultiIndex columns (levels named or positions)
    wide_multiindex_feature_level: Optional[Union[str, int]] = None,
    # Option B: Single string columns parsed via regex: must expose a named group "feature"
    wide_regex: str = r"^(?P<feature>.+?)_(?P<time>[^_]+)$",
    # Optional: explicit mapping column -> feature (overrides regex if provided)
    wide_col_to_feature: Optional[Mapping[str, str]] = None,
    # ---- General controls ----
    min_timepoints_per_feature: int = 1,
    drop_features_below_min: bool = True,
    sort_features: bool = True,
) -> pd.DataFrame:
    """Build a matrix of missingness proportions across timepoints for each sample-feature combination.
    
    Constructs a (n_samples × n_features) matrix where each entry represents the proportion
    of missing values across all timepoints for that specific sample and feature. Handles
    both wide format (columns represent feature×timepoint combinations) and long format
    (rows represent individual sample/feature/timepoint observations) data structures.
    
    :param data: Source dataset in either wide or long format
    :type data: pandas.DataFrame
    :param format: Data layout format, defaults to "wide"
    :type format: str, optional
    :param sample_col: Sample identifier column (required for long format, optional for wide), defaults to None
    :type sample_col: str, optional
    :param feature_col: Column name identifying feature (long format only), defaults to None
    :type feature_col: str, optional
    :param time_col: Column name identifying timepoint (long format only), defaults to None
    :type time_col: str, optional
    :param value_col: Column name containing measured values (long format only), defaults to None
    :type value_col: str, optional
    :param expected_timepoints: Complete set of expected timepoints for proportion calculation (long format), defaults to None
    :type expected_timepoints: Sequence, optional
    :param wide_multiindex_feature_level: MultiIndex level containing feature names (wide format), defaults to None
    :type wide_multiindex_feature_level: str or int, optional
    :param wide_regex: Regex pattern to extract feature from column names, must contain named group "feature" (wide format), defaults to r"^(?P<feature>.+?)_(?P<time>[^_]+)$"
    :type wide_regex: str, optional
    :param wide_col_to_feature: Explicit mapping from column names to feature labels (wide format), defaults to None
    :type wide_col_to_feature: Mapping[str, str], optional
    :param min_timepoints_per_feature: Minimum number of timepoints required per feature, defaults to 1
    :type min_timepoints_per_feature: int, optional
    :param drop_features_below_min: Whether to exclude features with insufficient timepoints, defaults to True
    :type drop_features_below_min: bool, optional
    :param sort_features: Whether to sort output columns alphabetically by feature name, defaults to True
    :type sort_features: bool, optional
    :return: Matrix with samples as rows, features as columns, and missingness proportions as values
    :rtype: pandas.DataFrame
    :raises ValueError: If format is not "wide" or "long", or if required parameters for chosen format are missing
    :raises ValueError: If wide_regex does not contain a named group "feature"
    :raises ValueError: If no feature groups can be formed from the input data
    """
    if format not in {"wide", "long"}:
        raise ValueError("`format` must be 'wide' or 'long'.")

    # ------------------------------
    # Helper: finalize output layout
    # ------------------------------
    def _finalize(df_props: pd.DataFrame) -> pd.DataFrame:
        # Clip to [0,1] for safety; ensure numeric dtype
        df_props = df_props.astype(float).clip(lower=0.0, upper=1.0)
        if sort_features:
            df_props = df_props.reindex(sorted(df_props.columns), axis=1)
        return df_props

    # ======================================================================================
    # WIDE FORMAT
    # ======================================================================================
    if format == "wide":
        df = data.copy()

        # Optionally set index to samples if the user provided `sample_col`
        if sample_col is not None and sample_col in df.columns:
            df = df.set_index(sample_col)

        # Determine feature for each column
        if isinstance(df.columns, pd.MultiIndex):
            # MultiIndex case: pick the specified level for feature names
            if wide_multiindex_feature_level is None:
                # Try to infer by name
                if "feature" in df.columns.names:
                    lvl = "feature"
                else:
                    lvl = 0  # fall back to level 0
            else:
                lvl = wide_multiindex_feature_level
            feature_labels = df.columns.get_level_values(lvl).to_list()
            col_to_bm = dict(zip(df.columns, feature_labels))
        else:
            # String columns: use explicit mapping or regex
            if wide_col_to_feature is not None:
                col_to_bm = dict(wide_col_to_feature)
            else:
                # Compile regex once; must have named group "feature"
                try:
                    rx = re.compile(wide_regex)
                    _ = rx.groupindex["feature"]
                except Exception as e:
                    raise ValueError(
                        "wide_regex must be a valid regex with a named group 'feature'."
                    ) from e

                col_to_bm = {}
                for c in df.columns:
                    m = rx.match(str(c))
                    if m and ("feature" in m.groupdict()):
                        col_to_bm[c] = m.group("feature")
                    else:
                        # Columns that don't match are ignored for the proportion calc
                        # (but kept in df; they just won't contribute to any feature)
                        continue

        # Group columns by feature
        groups: Dict[str, List] = {}
        for col, bm in col_to_bm.items():
            groups.setdefault(bm, []).append(col)

        # Optionally drop features with too few timepoints
        if drop_features_below_min:
            groups = {bm: cols for bm, cols in groups.items()
                      if len(cols) >= min_timepoints_per_feature}

        if len(groups) == 0:
            raise ValueError("No feature groups could be formed from wide data.")

        # Compute per-row proportion missing per feature:
        #   mean of isna() across the group's columns.
        out_parts = []
        for bm, cols in groups.items():
            # Skip empty groups defensively
            if len(cols) == 0:
                continue
            # mean of boolean NA mask across the timepoint columns for this feature
            prop_missing = df[cols].isna().mean(axis=1)
            out_parts.append(prop_missing.rename(bm))

        prop_matrix = pd.concat(out_parts, axis=1)
        return _finalize(prop_matrix)

    # ======================================================================================
    # LONG FORMAT
    # ======================================================================================
    # Validate required columns
    required = {"sample_col": sample_col, "feature_col": feature_col,
                "time_col": time_col, "value_col": value_col}
    missing = [k for k, v in required.items() if v is None]
    if missing:
        raise ValueError(f"Long format requires: {', '.join(missing)}")

    df = data[[sample_col, feature_col, time_col, value_col]].copy()

    # If the user provides the expected time grid, expand to count absent rows as missing
    if expected_timepoints is not None:
        expected_timepoints = list(expected_timepoints)

        # Build the full index of (sample, feature, time)
        samples = df[sample_col].unique()
        features = df[feature_col].unique()
        full_index = (
            pd.MultiIndex.from_product(
                [samples, features, expected_timepoints],
                names=[sample_col, feature_col, time_col]
            )
        )
        # Reindex the long table; absent rows become NaN in value_col
        df_full = (
            df.set_index([sample_col, feature_col, time_col])
              .reindex(full_index)
              .reset_index()
        )
        vcol = value_col
    else:
        # Use observed rows only (denominator = number of rows actually present)
        df_full = df
        vcol = value_col

    # Compute per (sample, feature) proportion missing across its time rows
    grp = df_full.groupby([sample_col, feature_col], dropna=False)
    # count of rows per cell in the grid (denominator), missing count (numerator)
    denom = grp[vcol].size()
    num_missing = grp[vcol].apply(lambda s: s.isna().sum())

    # If dropping features with too few timepoints, enforce per (sample, feature)
    if drop_features_below_min:
        valid_mask = denom >= min_timepoints_per_feature
        denom = denom.where(valid_mask)
        num_missing = num_missing.where(valid_mask)

    # Compute proportion (safe division)
    prop = (num_missing / denom).rename("prop_missing")

    # Pivot to (n_samples × n_features)
    prop_matrix = prop.unstack(feature_col)

    # Ensure stable row order by sample_col (optional; here we keep pandas default)
    return prop_matrix.pipe(_finalize)