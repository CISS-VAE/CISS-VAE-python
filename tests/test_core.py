import pytest
import numpy as np
import pandas as pd
import torch
from ciss_vae.training.run_cissvae import run_cissvae
from ciss_vae.classes.vae import CISSVAE
from ciss_vae.classes.cluster_dataset import ClusterDataset
from ciss_vae.utils.matrix import create_missingness_prop_matrix


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