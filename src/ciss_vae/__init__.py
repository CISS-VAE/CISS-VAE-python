## This needs to be updated since we've added and changed things. 
from .classes.vae import CISSVAE
from .classes.cluster_dataset import ClusterDataset
from .utils.loss import loss_function
from .training.autotune import autotune
from .training.train_initial import train_vae_initial
from .training.train_refit import impute_and_refit_loop
from .utils.helpers import plot_vae_architecture, get_imputed_df, evaluate_imputation, get_val_comp_df
from .training.run_cissvae import run_cissvae
from .utils.clustering import  cluster_on_missing, cluster_on_missing_prop
from .utils.matrix import create_missingness_prop_matrix

# ciss_vae/__init__.py
# Expose __version__ based on the installed package metadata.
# Falls back to a local placeholder when the package isn't installed.

from importlib.metadata import PackageNotFoundError, version as _pkg_version  # Python 3.8+
# If you support 3.7, use: from importlib_metadata import PackageNotFoundError, version as _pkg_version

try:
    # IMPORTANT: use the *distribution name* exactly as in pyproject.toml ([project].name)
    __version__ = _pkg_version("CISS-VAE")   # e.g., "ciss-vae" not "ciss_vae"
except PackageNotFoundError:
    __version__ = "0.0.0+local"
