import pytest
import numpy as np
import pandas as pd
import torch
from ciss_vae.training.run_cissvae import run_cissvae
from ciss_vae.classes.vae import CISSVAE
from ciss_vae.classes.cluster_dataset import ClusterDataset
from ciss_vae.utils.matrix import create_missingness_prop_matrix
from ciss_vae.utils.helpers import get_imputed_df

from torch.utils.data import DataLoader
from types import SimpleNamespace


class TestRunCissVAE:
    
    def test_default_returns(self, sample_data, minimal_params):
        """Test default return configuration (imputed_dataset, vae)"""
        result = run_cissvae(sample_data, **minimal_params)
        
        # Default: return_model=True, others=False
        # Should return: (imputed_dataset, vae)
        assert isinstance(result, tuple)
        assert len(result) == 2
        
        imputed_dataset, vae = result
        
        # Check types
        assert isinstance(imputed_dataset, pd.DataFrame)
        assert isinstance(vae, CISSVAE)
        
        # Check dimensions
        assert imputed_dataset.shape == sample_data.shape
    
    def test_single_return_imputed_dataset_only(self, sample_data, minimal_params):
        """Test returning only imputed dataset"""
        result = run_cissvae(
            sample_data,
            return_model=False,
            return_clusters=False,
            return_silhouettes=False,
            return_history=False,
            return_dataset=False,
            **minimal_params
        )
        
        # Should return single DataFrame, not wrapped in tuple
        assert isinstance(result, pd.DataFrame)
        assert result.shape == sample_data.shape
    
    @pytest.mark.parametrize("return_flags,expected_types", [
    # Single returns
    (
        {'return_model': True, 'return_clusters': False, 'return_silhouettes': False, 'return_history': False, 'return_dataset': False},
        [pd.DataFrame, CISSVAE]
    ),
    (
        {'return_model': False, 'return_clusters': True, 'return_silhouettes': False, 'return_history': False, 'return_dataset': False},
        [pd.DataFrame, np.ndarray]
    ),
    (
        # silhouettes returns ONE item (float | None | dict), no extra DataFrame
        {'return_model': False, 'return_clusters': False, 'return_silhouettes': True, 'return_history': False, 'return_dataset': False},
        [pd.DataFrame, (float, type(None), dict)]
    ),
    (
        {'return_model': False, 'return_clusters': False, 'return_silhouettes': False, 'return_history': True, 'return_dataset': False},
        [pd.DataFrame, (pd.DataFrame, type(None))]
    ),
    (
        {'return_model': False, 'return_clusters': False, 'return_silhouettes': False, 'return_history': False, 'return_dataset': True},
        [pd.DataFrame, ClusterDataset]
    ),

    # Multiple returns - test order
    (
        {'return_model': True, 'return_clusters': True, 'return_silhouettes': False, 'return_history': False, 'return_dataset': False},
        [pd.DataFrame, CISSVAE, np.ndarray]
    ),
    (
        # model + silhouettes ⇒ three items total
        {'return_model': True, 'return_clusters': False, 'return_silhouettes': True, 'return_history': False, 'return_dataset': False},
        [pd.DataFrame, CISSVAE, (float, type(None), dict)]
    ),
    (
        # clusters + silhouettes ⇒ three items total
        {'return_model': False, 'return_clusters': True, 'return_silhouettes': True, 'return_history': False, 'return_dataset': False},
        [pd.DataFrame, np.ndarray, (float, type(None), dict)]
    ),
    (
        {'return_model': True, 'return_dataset': True, 'return_clusters': False, 'return_silhouettes': False, 'return_history': False},
        [pd.DataFrame, CISSVAE, ClusterDataset]
    ),

    # All returns (no extra DF for silhouettes)
    (
        {'return_model': True, 'return_dataset': True, 'return_clusters': True, 'return_silhouettes': True, 'return_history': True},
        [pd.DataFrame, CISSVAE, ClusterDataset, np.ndarray, (float, type(None), dict), (pd.DataFrame, type(None))]
    ),
    ])

    def test_return_combinations(self, sample_data, minimal_params, return_flags, expected_types):
        """Test various return flag combinations and verify types and order"""
        result = run_cissvae(sample_data, **return_flags, **minimal_params)
        
        if len(expected_types) == 1:
            # Single item might not be wrapped in tuple
            if not isinstance(result, tuple):
                result = (result,)
        
        assert isinstance(result, tuple)
        assert len(result) == len(expected_types)
        
        for i, (item, expected_type) in enumerate(zip(result, expected_types)):
            if isinstance(expected_type, tuple):
                # Multiple acceptable types (e.g., float or None for silhouette)
                assert isinstance(item, expected_type), f"Item {i}: expected {expected_type}, got {type(item)}"
            else:
                assert isinstance(item, expected_type), f"Item {i}: expected {expected_type}, got {type(item)}"
    
    def test_return_order_consistency(self, sample_data, minimal_params):
        """Test that return order is always consistent regardless of which flags are set"""
        
        # Test: imputed_dataset, vae, dataset, clusters
        result1 = run_cissvae(
            sample_data,
            return_model=True,
            return_dataset=True,
            return_clusters=True,
            return_silhouettes=False,
            return_history=False,
            **minimal_params
        )
        
        assert len(result1) == 4
        imputed_dataset, vae, dataset, clusters = result1
        assert isinstance(imputed_dataset, pd.DataFrame)
        assert isinstance(vae, CISSVAE)
        assert isinstance(dataset, ClusterDataset)
        assert isinstance(clusters, np.ndarray)
        
        # Test: imputed_dataset, vae, clusters, silhouettes, history
        result2 = run_cissvae(
            sample_data,
            return_model=True,
            return_dataset=False,
            return_clusters=True,
            return_silhouettes=True,
            return_history=True,
            **minimal_params
        )
        
        assert len(result2) == 5
        imputed_dataset2, vae2, clusters2, silhouettes, history = result2
        assert isinstance(imputed_dataset2, pd.DataFrame)
        assert isinstance(vae2, CISSVAE)
        assert isinstance(clusters2, np.ndarray)
        assert (
            silhouettes is None
            or isinstance(silhouettes, float)
            or (isinstance(silhouettes, dict) and "mean_silhouette_width" in silhouettes)
        )

        assert isinstance(history, (pd.DataFrame, type(None)))
    
    def test_data_integrity(self, sample_data, minimal_params):
        """Test that returned data has correct properties"""
        result = run_cissvae(
            sample_data,
            return_model=True,
            return_clusters=True,
            return_dataset=True,
            **minimal_params
        )
        
        imputed_dataset, vae, dataset, clusters = result
        
        # Data shape consistency
        assert imputed_dataset.shape == sample_data.shape
        assert len(clusters) == len(sample_data)
        assert dataset.shape == sample_data.shape
        
        # No missing values in imputed dataset
        assert not imputed_dataset.isna().any().any()
        
        # Clusters should be integers starting from 0
        unique_clusters = np.unique(clusters)
        assert np.all(unique_clusters >= 0)
        assert np.all(unique_clusters == np.arange(len(unique_clusters)))
        
        # VAE should have correct architecture
        assert vae.input_dim == sample_data.shape[1]
        assert vae.num_clusters == len(unique_clusters)
    
    def test_model_architecture_parameters(self, sample_data, minimal_params):
        """Test that model architecture parameters are respected"""
        custom_params = {
            **minimal_params,
            'hidden_dims': [64, 32],
            'latent_dim': 10,
            'layer_order_enc': ['unshared', 'shared'],
            'layer_order_dec': ['shared', 'unshared'],
            'latent_shared': True,
            'output_shared': False,
        }
        
        result = run_cissvae(
            sample_data,
            return_model=True,
            return_clusters=False,
            **custom_params
        )
        
        imputed_dataset, vae = result
        
        # Check VAE architecture matches parameters
        assert vae.hidden_dims == [64, 32]
        assert vae.latent_dim == 10
        assert vae.latent_shared == True
        assert vae.output_shared == False
    
    def test_clustering_parameters(self, sample_data, minimal_params):
        """Test that clustering parameters work correctly"""
        # Test with fixed number of clusters
        result1 = run_cissvae(
            sample_data,
            return_clusters=True,
            return_silhouettes=True,
            return_model=False,
            **minimal_params
        )
        
        imputed_dataset, clusters, silhouettes = result1
        
        # Should have exactly 2 clusters when n_clusters=2
        unique_clusters = np.unique(clusters)
        assert len(unique_clusters) == 2
        
        # Test with Leiden clustering (no n_clusters specified)
        params = minimal_params.copy()
        params["n_clusters"] = None
        params["leiden_resolution"] = 0.1
        result2 = run_cissvae(
            sample_data,
            return_clusters=True,
            return_model=False,
            **params
        )
        
        imputed_dataset2, clusters2 = result2
        
        # Should have some reasonable number of clusters
        unique_clusters2 = np.unique(clusters2)
        assert len(unique_clusters2) >= 1
        assert len(unique_clusters2) <= len(sample_data) // 2  # Sanity check

    def test_prop_clustering(self, longitudinal_data, minimal_params):
        """Test that create_missngness_prop_matrix and prop matrix stuff works correctly"""

        ## should make array w/ 3 columns and same number of rows as longitudinal_data
        prop_matrix = create_missingness_prop_matrix(longitudinal_data, repeat_feature_names=["y1", "y2", "y3"])

        result1 = prop_matrix.data

        assert result1.shape[1] == 3
        assert result1.shape[0] == longitudinal_data.shape[0]

        result2 = run_cissvae(
            longitudinal_data,
            return_clusters=True,
            return_silhouettes=True,
            return_model=False,
            missingness_proportion_matrix=prop_matrix,
            **minimal_params
        )
        
        imputed_dataset, clusters, silhouettes = result2

        # Should have exactly 2 clusters when n_clusters=2
        unique_clusters = np.unique(clusters)
        assert len(unique_clusters) == 2
        assert len(clusters) == longitudinal_data.shape[0]
    
    def test_training_parameters(self, sample_data, minimal_params):
        """Test that training parameters don't break the pipeline"""
        custom_params = {
            **minimal_params,
            'epochs': 1,  # Very short for speed
            'max_loops': 1,
            'epochs_per_loop': 1,
            'initial_lr': 0.1,
            'decay_factor': 0.9,
            'beta': 0.01,
        }
        
        result = run_cissvae(
            sample_data,
            return_model=True,
            return_history=True,
            **custom_params
        )
        
        imputed_dataset, vae, history = result
        
        assert isinstance(imputed_dataset, pd.DataFrame)
        assert isinstance(vae, CISSVAE)
        # History might be None or DataFrame depending on implementation
        assert isinstance(history, (pd.DataFrame, type(None)))

    def test_activation_groups_present_in_dataset(self, sample_data, minimal_params):
        result = run_cissvae(
            sample_data,
            return_dataset=True,
            return_model=False,
            return_clusters=False,
            return_silhouettes=False,
            return_history=False,
            **minimal_params
        )

        imputed_dataset, dataset = result

        assert isinstance(dataset, ClusterDataset)

        ag = dataset.activation_groups

        # structure
        assert isinstance(ag, dict)
        assert "continuous" in ag

        # coverage
        all_cols = set()
        for cols in ag.values():
            all_cols.update(cols)

        assert all_cols == set(range(dataset.shape[1]))
    
    @pytest.mark.slow
    def test_full_pipeline_integration(self, large_sample_data):
        """Test full pipeline with larger data and more realistic parameters"""
        result = run_cissvae(
            large_sample_data,
            hidden_dims=[100, 50, 25],
            latent_dim=15,
            epochs=5,
            max_loops=3,
            epochs_per_loop=2,
            batch_size=128,
            return_model=True,
            return_clusters=True,
            return_silhouettes=True,
            return_history=True,
            return_dataset=True,
            verbose=False
        )
        
        imputed_dataset, vae, dataset, clusters, silhouettes, history = result
        
        # All return types should be correct
        assert isinstance(imputed_dataset, pd.DataFrame)
        assert isinstance(vae, CISSVAE)
        assert isinstance(dataset, ClusterDataset)
        assert isinstance(clusters, np.ndarray)
        assert (
            silhouettes is None
            or isinstance(silhouettes, float)
            or (isinstance(silhouettes, dict) and "mean_silhouette_width" in silhouettes)
        )

        assert isinstance(history, (pd.DataFrame, type(None)))
        
        # Data integrity
        assert imputed_dataset.shape == large_sample_data.shape
        assert not imputed_dataset.isna().any().any()


class DummyModel(torch.nn.Module):
    """
    Minimal model that returns its input as reconstruction.
    This lets us control behavior without mocking.
    """
    def forward(self, X, C, deterministic=True):
        return X, None, None


def test_get_imputed_df_unscales_ignored_continuous_columns():
    """
    Ensure ignored continuous columns are properly unscaled in get_imputed_df.
    """

    # -------------------------
    # Simple dataset
    # -------------------------
    df = pd.DataFrame({
        "ignored_col": [10.0, 20.0, 30.0, 40.0],
        "x1": [1.0, 2.0, 3.0, 4.0],
    })

    dataset = ClusterDataset(
        data=df,
        cluster_labels=np.zeros(len(df)),
        val_proportion=0.0,
        columns_ignore=["ignored_col"],
        binary_feature_mask=[False, False],
        categorical_column_map=None,
    )

    # -------------------------
    # DataLoader
    # -------------------------
    loader = DataLoader(dataset, batch_size=2, shuffle=False)

    # -------------------------
    # Run imputation
    # -------------------------
    model = DummyModel()
    out_df = get_imputed_df(model, loader)

    # -------------------------
    # Check both columns match original scale
    # -------------------------
    np.testing.assert_allclose(
        out_df["ignored_col"].values,
        df["ignored_col"].values,
        atol=1e-6
    )

    np.testing.assert_allclose(
        out_df["x1"].values,
        df["x1"].values,
        atol=1e-6
    )

import torch
import numpy as np

from ciss_vae.utils.loss import loss_function_nomask


def test_loss_includes_ignored_columns():
    """
    Ensure ignored columns contribute to loss.

    Strategy:
    - Build simple tensors
    - Inject error ONLY in ignored columns
    - Verify loss > 0
    """

    # -------------------------
    # Fake data (2 samples, 2 features)
    # col 0 = ignored
    # col 1 = active
    # -------------------------
    x = torch.tensor([
        [10.0, 1.0],
        [20.0, 2.0],
    ])

    recon = x.clone()

    # inject error ONLY in ignored column (col 0)
    recon[:, 0] = 9999.0

    # mask = all observed
    mask = torch.ones_like(x)

    # activation groups include BOTH columns
    activation_groups = {
        "continuous": [0, 1]
    }

    # dummy latent variables
    mu = torch.zeros((2, 1))
    logvar = torch.zeros((2, 1))

    loss, mse, bce, ce = loss_function_nomask(
        cluster=None,
        recon_x=recon,
        x=x,
        activation_groups=activation_groups,
        mu=mu,
        logvar=logvar,
        return_components=True,
    )

    assert loss.item() > 0, "Ignored column did not contribute to loss"
    assert mse.item() > 0, "Ignored column did not contribute to MSE"

def test_loss_changes_when_ignored_column_removed():
    x = torch.tensor([[10.0, 1.0]])
    recon = x.clone()

    # error in col 0
    recon[:, 0] = 9999.0

    mu = torch.zeros((1, 1))
    logvar = torch.zeros((1, 1))

    # case 1: include ignored column
    groups_full = {"continuous": [0, 1]}

    loss_full, *_ = loss_function_nomask(
        None, recon, x, groups_full, mu, logvar, return_components=True
    )

    # case 2: exclude ignored column
    groups_filtered = {"continuous": [1]}

    loss_filtered, *_ = loss_function_nomask(
        None, recon, x, groups_filtered, mu, logvar, return_components=True
    )

    assert loss_full.item() > loss_filtered.item(), (
        "Ignored column is not contributing to loss as expected"
    )