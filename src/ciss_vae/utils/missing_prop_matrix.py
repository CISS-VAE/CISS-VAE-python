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
    biomarker_col: Optional[str] = None,
    time_col: Optional[str] = None,
    value_col: Optional[str] = None,
    expected_timepoints: Optional[Sequence] = None,
    # ---- Wide format parameters ----
    # Option A: MultiIndex columns (levels named or positions)
    wide_multiindex_biomarker_level: Optional[Union[str, int]] = None,
    # Option B: Single string columns parsed via regex: must expose a named group "biomarker"
    wide_regex: str = r"^(?P<biomarker>.+?)_(?P<time>[^_]+)$",
    # Optional: explicit mapping column -> biomarker (overrides regex if provided)
    wide_col_to_biomarker: Optional[Mapping[str, str]] = None,
    # ---- General controls ----
    min_timepoints_per_biomarker: int = 1,
    drop_biomarkers_below_min: bool = True,
    sort_biomarkers: bool = True,
) -> pd.DataFrame:
    """
    Build a (n_samples × n_biomarkers) matrix where each entry is the *proportion of
    missingness across all timepoints* for that (sample, biomarker).

    Parameters
    ----------
    data : pd.DataFrame
        Source data, either wide (columns represent biomarker×timepoint) or long
        (one row per sample/biomarker/timepoint).
    format : {"wide", "long"}, default="wide"
        - "wide": columns encode biomarker and timepoint (via MultiIndex, regex, or a
          provided mapping).
        - "long": provide `sample_col`, `biomarker_col`, `time_col`, `value_col`.
    sample_col : str or None
        Long: **required** — sample identifier column.
        Wide: optional — if provided, will be set as index (otherwise RangeIndex used).

    Long-format parameters
    ----------------------
    biomarker_col, time_col, value_col : str
        Column names identifying biomarker, timepoint, and the measured value.
    expected_timepoints : Sequence or None, default=None
        If provided, the denominator for the proportion will be `len(expected_timepoints)`
        for every (sample, biomarker). Missing *rows* for absent timepoints are counted
        as missing. If None, the denominator is the number of *observed* rows for that
        (sample, biomarker) — use only if you truly don't know the expected grid.

    Wide-format parameters
    ----------------------
    wide_multiindex_biomarker_level : str or int or None
        If `data.columns` is a MultiIndex, indicate which level holds the biomarker
        name (level name or index).
    wide_regex : str, default=r"^(?P<biomarker>.+?)_(?P<time>[^_]+)$"
        Regex used to parse biomarker (and optionally time) from string column names.
        Must contain a named group `(?P<biomarker>...)`.
    wide_col_to_biomarker : Mapping[str, str] or None
        Optional explicit mapping from column name -> biomarker label. If provided,
        this overrides `wide_regex`.

    General controls
    ----------------
    min_timepoints_per_biomarker : int, default=1
        Minimum number of timepoint columns/rows required for a biomarker to be kept.
    drop_biomarkers_below_min : bool, default=True
        If True, biomarkers with fewer than `min_timepoints_per_biomarker` timepoints
        are dropped; otherwise they are kept and their proportions are computed over
        whatever timepoints exist.
    sort_biomarkers : bool, default=True
        If True, columns in the output are sorted alphabetically by biomarker name.

    Returns
    -------
    prop_matrix : pd.DataFrame, shape (n_samples, n_biomarkers)
        Each cell is in [0, 1] and equals:
            (# missing timepoints for that biomarker & sample) / (total expected timepoints)

    Notes
    -----
    - In **wide** mode, the proportion for biomarker B in a row is the mean of `isna()`
      across all columns that belong to B.
    - In **long** mode, if `expected_timepoints` is given, we first expand to the full
      sample×biomarker×time grid so absent rows are counted as missing.
    - The output is designed to be passed directly to `cluster_on_missingness_prop`.
    """
    if format not in {"wide", "long"}:
        raise ValueError("`format` must be 'wide' or 'long'.")

    # ------------------------------
    # Helper: finalize output layout
    # ------------------------------
    def _finalize(df_props: pd.DataFrame) -> pd.DataFrame:
        # Clip to [0,1] for safety; ensure numeric dtype
        df_props = df_props.astype(float).clip(lower=0.0, upper=1.0)
        if sort_biomarkers:
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

        # Determine biomarker for each column
        if isinstance(df.columns, pd.MultiIndex):
            # MultiIndex case: pick the specified level for biomarker names
            if wide_multiindex_biomarker_level is None:
                # Try to infer by name
                if "biomarker" in df.columns.names:
                    lvl = "biomarker"
                else:
                    lvl = 0  # fall back to level 0
            else:
                lvl = wide_multiindex_biomarker_level
            biomarker_labels = df.columns.get_level_values(lvl).to_list()
            col_to_bm = dict(zip(df.columns, biomarker_labels))
        else:
            # String columns: use explicit mapping or regex
            if wide_col_to_biomarker is not None:
                col_to_bm = dict(wide_col_to_biomarker)
            else:
                # Compile regex once; must have named group "biomarker"
                try:
                    rx = re.compile(wide_regex)
                    _ = rx.groupindex["biomarker"]
                except Exception as e:
                    raise ValueError(
                        "wide_regex must be a valid regex with a named group 'biomarker'."
                    ) from e

                col_to_bm = {}
                for c in df.columns:
                    m = rx.match(str(c))
                    if m and ("biomarker" in m.groupdict()):
                        col_to_bm[c] = m.group("biomarker")
                    else:
                        # Columns that don't match are ignored for the proportion calc
                        # (but kept in df; they just won't contribute to any biomarker)
                        continue

        # Group columns by biomarker
        groups: Dict[str, List] = {}
        for col, bm in col_to_bm.items():
            groups.setdefault(bm, []).append(col)

        # Optionally drop biomarkers with too few timepoints
        if drop_biomarkers_below_min:
            groups = {bm: cols for bm, cols in groups.items()
                      if len(cols) >= min_timepoints_per_biomarker}

        if len(groups) == 0:
            raise ValueError("No biomarker groups could be formed from wide data.")

        # Compute per-row proportion missing per biomarker:
        #   mean of isna() across the group's columns.
        out_parts = []
        for bm, cols in groups.items():
            # Skip empty groups defensively
            if len(cols) == 0:
                continue
            # mean of boolean NA mask across the timepoint columns for this biomarker
            prop_missing = df[cols].isna().mean(axis=1)
            out_parts.append(prop_missing.rename(bm))

        prop_matrix = pd.concat(out_parts, axis=1)
        return _finalize(prop_matrix)

    # ======================================================================================
    # LONG FORMAT
    # ======================================================================================
    # Validate required columns
    required = {"sample_col": sample_col, "biomarker_col": biomarker_col,
                "time_col": time_col, "value_col": value_col}
    missing = [k for k, v in required.items() if v is None]
    if missing:
        raise ValueError(f"Long format requires: {', '.join(missing)}")

    df = data[[sample_col, biomarker_col, time_col, value_col]].copy()

    # If the user provides the expected time grid, expand to count absent rows as missing
    if expected_timepoints is not None:
        expected_timepoints = list(expected_timepoints)

        # Build the full index of (sample, biomarker, time)
        samples = df[sample_col].unique()
        biomarkers = df[biomarker_col].unique()
        full_index = (
            pd.MultiIndex.from_product(
                [samples, biomarkers, expected_timepoints],
                names=[sample_col, biomarker_col, time_col]
            )
        )
        # Reindex the long table; absent rows become NaN in value_col
        df_full = (
            df.set_index([sample_col, biomarker_col, time_col])
              .reindex(full_index)
              .reset_index()
        )
        vcol = value_col
    else:
        # Use observed rows only (denominator = number of rows actually present)
        df_full = df
        vcol = value_col

    # Compute per (sample, biomarker) proportion missing across its time rows
    grp = df_full.groupby([sample_col, biomarker_col], dropna=False)
    # count of rows per cell in the grid (denominator), missing count (numerator)
    denom = grp[vcol].size()
    num_missing = grp[vcol].apply(lambda s: s.isna().sum())

    # If dropping biomarkers with too few timepoints, enforce per (sample, biomarker)
    if drop_biomarkers_below_min:
        valid_mask = denom >= min_timepoints_per_biomarker
        denom = denom.where(valid_mask)
        num_missing = num_missing.where(valid_mask)

    # Compute proportion (safe division)
    prop = (num_missing / denom).rename("prop_missing")

    # Pivot to (n_samples × n_biomarkers)
    prop_matrix = prop.unstack(biomarker_col)

    # Ensure stable row order by sample_col (optional; here we keep pandas default)
    return prop_matrix.pipe(_finalize)