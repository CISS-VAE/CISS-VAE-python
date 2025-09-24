"""Data loading utilities for CISS-VAE example dataset."""
import pandas as pd
try:
    from importlib import resources
except ImportError:
    # Python < 3.9 fallback
    import importlib_resources as resources

def load_example_dataset():
    """
    Load the complete CISS-VAE example dataset.
    
    Returns
    -------
    tuple
        A tuple containing (df_missing, df_complete, clusters) where:
        - df_missing : pd.DataFrame
            DataFrame with missing values
        - df_complete : pd.DataFrame  
            Complete DataFrame without missing values
        - clusters : np.ndarray
            Cluster labels as 1D array
    
    Examples
    --------
    >>> from ciss_vae.data import load_example_dataset
    >>> df_missing, df_complete, clusters = load_example_dataset()
    """
    # Load df_missing.csv with index_col=[0]
    with resources.open_text(__package__, "df_missing.csv") as f:
        df_missing = pd.read_csv(f, index_col=[0])
    
    # Load df_complete.csv with index_col=[0] 
    with resources.open_text(__package__, "df_complete.csv") as f:
        df_complete = pd.read_csv(f, index_col=[0])
    
    # Load clusters.csv with index_col=[0] and squeeze to 1D array
    with resources.open_text(__package__, "clusters.csv") as f:
        clusters_df = pd.read_csv(f, index_col=[0])
        clusters = clusters_df.values.squeeze()
    
    return df_missing, df_complete, clusters

def load_missing_data():
    """Load only the missing data DataFrame."""
    with resources.open_text(__package__, "df_missing.csv") as f:
        return pd.read_csv(f, index_col=[0])

def load_complete_data():
    """Load only the complete data DataFrame."""
    with resources.open_text(__package__, "df_complete.csv") as f:
        return pd.read_csv(f, index_col=[0])

def load_clusters():
    """Load only the cluster labels as 1D array."""
    with resources.open_text(__package__, "clusters.csv") as f:
        clusters_df = pd.read_csv(f, index_col=[0])
        return clusters_df.values.squeeze()

def load_dni():
    """Load dni matrix for df_missing"""
    with resources.open_text(__package__, "dni.csv") as f:
        return pd.read_csv(f, index_col=[0])

# For convenience, make the main function available at package level
__all__ = ["load_example_dataset", "load_missing_data", "load_complete_data", "load_clusters"]