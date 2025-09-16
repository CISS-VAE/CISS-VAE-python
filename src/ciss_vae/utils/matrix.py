from __future__ import annotations
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union, Callable
import re
import numpy as np
import pandas as pd

import pandas as pd
import numpy as np
import re
from typing import Optional, List, Union, Dict, Any
import pandas as pd
import numpy as np
import re
from typing import Optional, List, Union, Dict, Any

class MissingnessMatrix:
    """A matrix with missingness proportions and metadata."""
    
    def __init__(self, data: np.ndarray, feature_columns_map: Dict[str, List[str]], 
                 feature_names: List[str], sample_names: Optional[List[str]] = None):
        self.data = data
        self.feature_columns_map = feature_columns_map
        self.feature_names = feature_names
        self.sample_names = sample_names or list(range(len(data)))
    
    @property
    def shape(self):
        return self.data.shape
    
    def __getitem__(self, key):
        return self.data[key]
    
    def __array__(self):
        """Allow numpy operations on this object."""
        return self.data
    
    def to_dataframe(self):
        """Convert to pandas DataFrame."""
        return pd.DataFrame(
            self.data, 
            columns=self.feature_names,
            index=self.sample_names
        )

    def to_numpy(self, dtype=None, copy: bool = False) -> np.ndarray:
        """Return the underlying numpy array."""
        arr = self.data
        if dtype is not None and arr.dtype != dtype:
            arr = arr.astype(dtype, copy=False)
        if copy:
            arr = arr.copy()
        return arr

def create_missingness_prop_matrix(
    data: Union[pd.DataFrame, np.ndarray],
    index_col: Optional[str] = None,
    cols_ignore: Optional[List[str]] = None,
    na_values: Optional[List[Any]] = None,
    repeat_feature_names: Optional[List[str]] = None
) -> MissingnessMatrix:
    """
    Creates a matrix where each entry represents the proportion of missing values
    for each sample–feature combination across multiple timepoints.
    
    [Previous docstring content remains the same]
    
    Returns
    -------
    MissingnessMatrix
        A custom matrix object with missingness proportions and metadata.
        Access the numpy array with .data, feature mapping with .feature_columns_map,
        and convert to DataFrame with .to_dataframe().
    """
    
    # [All the previous validation and processing code remains the same until the end]
    
    # ------------------------------- #
    # 1) Validate & normalize inputs  #
    # ------------------------------- #
    if not isinstance(data, (pd.DataFrame, np.ndarray)):
        raise ValueError("`data` must be a pandas DataFrame or numpy array.")
    
    # Convert to DataFrame for uniform handling
    if isinstance(data, np.ndarray):
        df = pd.DataFrame(data)
    else:
        df = data.copy()
    
    # Set default values
    if na_values is None:
        na_values = [np.nan, np.inf, -np.inf]
    if repeat_feature_names is None:
        repeat_feature_names = []
    if cols_ignore is None:
        cols_ignore = []
    
    # Validate inputs
    if index_col is not None and not isinstance(index_col, str):
        raise ValueError("`index_col` must be None or a string.")
    
    if not isinstance(cols_ignore, list):
        raise ValueError("`cols_ignore` must be None or a list of column names.")
    
    if not isinstance(repeat_feature_names, list):
        raise ValueError("`repeat_feature_names` must be a list of strings.")
    
    # Determine columns to drop
    cols_to_drop = []
    if index_col is not None and index_col in df.columns:
        cols_to_drop.append(index_col)
    if cols_ignore:
        cols_to_drop.extend([col for col in cols_ignore if col in df.columns])
    
    cols_to_drop = list(set(cols_to_drop))  # Remove duplicates
    
    # --------------------------------------- #
    # 2) Build helper for missingness checks  #
    # --------------------------------------- #
    def is_missing(x):
        """Check if values are missing according to our criteria."""
        if isinstance(x, pd.Series):
            x = x.values
        
        # Start with standard missing flags
        miss = pd.isna(x)
        
        # Check for infinite values
        try:
            numeric_x = pd.to_numeric(x, errors='coerce')
            miss = miss | np.isinf(numeric_x)
        except:
            pass
        
        # Check user-provided values
        if na_values:
            for na_val in na_values:
                try:
                    miss = miss | (x == na_val)
                except:
                    # Handle type mismatches gracefully
                    pass
        
        return miss
    
    # --------------------------------------------------- #
    # 3) Identify columns for repeated vs single features #
    # --------------------------------------------------- #
    all_cols = list(df.columns)
    if len(all_cols) == 0:
        raise ValueError("Input `data` has no columns.")
    
    # Remove drop columns from consideration
    feature_candidate_cols = [col for col in all_cols if col not in cols_to_drop]
    
    # For each base feature in repeat_feature_names, collect its timepoint columns
    feature_to_cols = {}
    consumed_cols = []
    
    if repeat_feature_names:
        for feat in repeat_feature_names:
            # Escape special regex characters and create pattern
            feat_escaped = re.escape(feat)
            pattern = f"^{feat_escaped}_\\d+$"
            
            # Find matching columns
            matching_cols = [col for col in feature_candidate_cols 
                           if re.match(pattern, col)]
            
            if len(matching_cols) == 0:
                raise ValueError(
                    f"No columns found for repeated feature '{feat}' using pattern '{pattern}'. "
                    f"Ensure columns are named like '{feat}_1', '{feat}_2', ..."
                )
            
            feature_to_cols[feat] = matching_cols
            consumed_cols.extend(matching_cols)
    
    # Remaining feature columns (not part of repeated features and not dropped)
    remaining_cols = [col for col in feature_candidate_cols if col not in consumed_cols]
    
    # Treat each remaining column as a single-timepoint feature
    for col in remaining_cols:
        feature_to_cols[col] = [col]
    
    # Determine output feature order
    out_features = list(repeat_feature_names) + remaining_cols
    
    if len(out_features) == 0:
        raise ValueError("After excluding `index_col` and `cols_ignore`, no feature columns remain.")
    
    # ------------------------------------------- #
    # 4) Compute per-sample missingness proportion #
    # ------------------------------------------- #
    n_samples = len(df)
    n_features = len(out_features)
    
    # Initialize output matrix
    out = np.full((n_samples, n_features), np.nan, dtype=float)
    
    # Fill the matrix column-by-column
    for j, feat in enumerate(out_features):
        cols = feature_to_cols[feat]
        
        # Get data for this feature's columns
        subdf = df[cols]
        
        # Compute missingness for each column
        miss_matrix = subdf.apply(is_missing, axis=0)
        
        # Compute proportion missing per row (sample)
        prop_missing = miss_matrix.mean(axis=1).values
        
        out[:, j] = prop_missing
    
    # Get sample names if available
    sample_names = list(data.index) if hasattr(data, 'index') else None
    
    # Return custom object with metadata
    return MissingnessMatrix(
        data=out,
        feature_columns_map=feature_to_cols,
        feature_names=out_features,
        sample_names=sample_names
    )
