import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock, call
import optuna
import torch

from ciss_vae.training.autotune import autotune as run_autotune, SearchSpace
from ciss_vae.classes.cluster_dataset import ClusterDataset
from ciss_vae.classes.vae import CISSVAE

## Added mock model maker
def _make_mock_model():
    """Create mock CISSVAE model w/ attributes/methods used by autotune/training."""
    m = MagicMock(spec = CISSVAE)
    # add history df
    m.training_history_ = pd.DataFrame({
        "imputation_error": []
    })
    ## Add lr helpers called in refit loop
    m.set_final_lr = MagicMock()
    m.get_final_lr = MagicMock(return_value = 0.001)
    return m

class TestSearchSpace:
    """Test SearchSpace class initialization and parameter handling"""
    
    def test_default_search_space(self):
        """Test SearchSpace with default parameters"""
        search_space = SearchSpace()
        
        # Check that all default attributes exist
        assert hasattr(search_space, 'num_hidden_layers')
        assert hasattr(search_space, 'hidden_dims')
        assert hasattr(search_space, 'latent_dim')
        assert hasattr(search_space, 'latent_shared')
        assert hasattr(search_space, 'output_shared')
        assert hasattr(search_space, 'lr')
        assert hasattr(search_space, 'decay_factor')
        assert hasattr(search_space, 'beta')
        assert hasattr(search_space, 'num_epochs')
        assert hasattr(search_space, 'batch_size')
        assert hasattr(search_space, 'num_shared_encode')
        assert hasattr(search_space, 'num_shared_decode')
        assert hasattr(search_space, 'encoder_shared_placement')
        assert hasattr(search_space, 'decoder_shared_placement')
        assert hasattr(search_space, 'refit_patience')
        assert hasattr(search_space, 'refit_loops')
        assert hasattr(search_space, 'epochs_per_loop')
        assert hasattr(search_space, 'reset_lr_refit')
    
    def test_custom_search_space_fixed_params(self):
        """Test SearchSpace with all fixed parameters"""
        search_space = SearchSpace(
            num_hidden_layers=2,
            hidden_dims=128,
            latent_dim=20,
            latent_shared=True,
            output_shared=False,
            lr=0.001,
            decay_factor=0.95,
            beta=0.01,
            num_epochs=50,
            batch_size=64,
            num_shared_encode=1,
            num_shared_decode=1,
            encoder_shared_placement="at_end",
            decoder_shared_placement="at_start",
            refit_patience=2,
            refit_loops=20,
            epochs_per_loop=50,
            reset_lr_refit=True
        )
        
        # Verify all parameters are set correctly
        assert search_space.num_hidden_layers == 2
        assert search_space.hidden_dims == 128
        assert search_space.latent_dim == 20
        assert search_space.latent_shared == True
        assert search_space.output_shared == False
        assert search_space.lr == 0.001
        assert search_space.decay_factor == 0.95
        assert search_space.beta == 0.01
        assert search_space.num_epochs == 50
        assert search_space.batch_size == 64
        assert search_space.num_shared_encode == 1
        assert search_space.num_shared_decode == 1
        assert search_space.encoder_shared_placement == "at_end"
        assert search_space.decoder_shared_placement == "at_start"
        assert search_space.refit_patience == 2
        assert search_space.refit_loops == 20
        assert search_space.epochs_per_loop == 50
        assert search_space.reset_lr_refit == True
    
    def test_custom_search_space_tunable_params(self):
        """Test SearchSpace with tunable parameters"""
        search_space = SearchSpace(
            num_hidden_layers=(1, 4),
            hidden_dims=[64, 128, 256, 512],
            latent_dim=(10, 100),
            latent_shared=[True, False],
            output_shared=[True, False],
            lr=(1e-4, 1e-2),
            decay_factor=(0.9, 0.99),
            beta=[0.001, 0.01, 0.1],
            num_epochs=[10, 50, 100],
            batch_size=[32, 64, 128],
            num_shared_encode=[0, 1, 3],
            num_shared_decode=[0, 1, 3],
            encoder_shared_placement=["at_end", "at_start", "alternating", "random"],
            decoder_shared_placement=["at_end", "at_start", "alternating", "random"],
            refit_patience=[1, 2, 5],
            refit_loops=[10, 50, 100],
            epochs_per_loop=[10, 100, 1000],
            reset_lr_refit=[True, False]
        )
        
        # Verify tunable parameters are lists/tuples
        assert isinstance(search_space.num_hidden_layers, tuple)
        assert isinstance(search_space.hidden_dims, list)
        assert isinstance(search_space.latent_dim, tuple)
        assert isinstance(search_space.latent_shared, list)
        assert isinstance(search_space.lr, tuple)
        assert len(search_space.encoder_shared_placement) == 4


class TestAutoTune:
    """Test autotune function"""
    
    # REMOVE THE mock_cluster_dataset FIXTURE FROM HERE - it's now in conftest.py
    
    @pytest.fixture
    def basic_search_space(self):
        """Basic search space for testing"""
        return SearchSpace(
            num_hidden_layers=(1, 3),
            hidden_dims=[32, 64],
            latent_dim=[8, 16],
            latent_shared=[True, False],
            output_shared=[True, False],
            lr=(1e-4, 1e-3),
            decay_factor=(0.9, 0.999),
            beta=0.01,  # Fixed parameter
            num_epochs=5,  # Small for testing
            batch_size=32,  # Fixed parameter
            num_shared_encode=[0, 1],
            num_shared_decode=[0, 1],
            encoder_shared_placement=["at_end", "at_start"],
            decoder_shared_placement=["at_end", "at_start"],
            refit_patience=2,
            refit_loops=2,  # Small for testing
            epochs_per_loop=2,  # Small for testing
            reset_lr_refit=[True, False]
        )
    
    @pytest.fixture
    def all_fixed_search_space(self):
        """Search space with all fixed parameters"""
        return SearchSpace(
            num_hidden_layers=2,
            hidden_dims=64,
            latent_dim=16,
            latent_shared=True,
            output_shared=False,
            lr=0.001,
            decay_factor=0.95,
            beta=0.01,
            num_epochs=3,
            batch_size=32,
            num_shared_encode=[1],  # ← CHANGE: wrap in list
            num_shared_decode=[1],  # ← CHANGE: wrap in list
            encoder_shared_placement="at_end",
            decoder_shared_placement="at_start",
            refit_patience=2,
            refit_loops=2,
            epochs_per_loop=2,
            reset_lr_refit=True
        )


    @patch('ciss_vae.training.autotune.train_vae_initial')
    @patch('ciss_vae.training.autotune.impute_and_refit_loop')
    @patch('ciss_vae.training.autotune.compute_val_mse')
    def test_fixed_parameters(self, mock_compute_val_mse, mock_refit, mock_initial, 
                             mock_cluster_dataset, all_fixed_search_space):
        """Test autotune with fixed parameters"""
        # Mock return values
        mock_model = _make_mock_model()
        mock_initial.return_value = mock_model
        mock_refit.return_value = (pd.DataFrame(), mock_model, None)
        mock_compute_val_mse.return_value = 0.5, 0.45, 0.05
        
        result = run_autotune(
            search_space=all_fixed_search_space,
            train_dataset=mock_cluster_dataset,
            n_trials=2,
            evaluate_all_orders=False,
            verbose=False,
            show_progress=False
        )
        
        assert result is not None
        assert isinstance(result, tuple)
        assert len(result) == 4  # best_imputed_df, best_model, study, results_df
        
        best_imputed_df, best_model, study, results_df = result
        assert not results_df["imputation_error"].isna().any(), "imputation_error column contains NaN"
        assert isinstance(best_imputed_df, pd.DataFrame)
        assert best_model is not None
        assert isinstance(study, optuna.study.Study)
        assert isinstance(results_df, pd.DataFrame)

    @patch('ciss_vae.training.autotune.train_vae_initial')
    @patch('ciss_vae.training.autotune.impute_and_refit_loop')
    @patch('ciss_vae.training.autotune.compute_val_mse')
    def test_tunable_parameters(self, mock_compute_val_mse, mock_refit, mock_initial,
                               mock_cluster_dataset, basic_search_space):
        """Test autotune with tunable parameters"""
        # Mock return values
        mock_model = _make_mock_model()
        mock_initial.return_value = mock_model
        mock_refit.return_value = (pd.DataFrame(), mock_model, None)
        mock_compute_val_mse.return_value = 0.3, 0.25, 0.05
        
        result = run_autotune(
            search_space=basic_search_space,
            train_dataset=mock_cluster_dataset,
            n_trials=3,
            evaluate_all_orders=False,
            verbose=False,
            show_progress=False
        )
        
        assert result is not None
        assert isinstance(result, tuple)
        assert len(result) == 4

    @pytest.mark.parametrize("n_trials", [1, 3, 5])
    @patch('ciss_vae.training.autotune.train_vae_initial')
    @patch('ciss_vae.training.autotune.impute_and_refit_loop') 
    @patch('ciss_vae.training.autotune.compute_val_mse')
    def test_n_trials_parameter(self, mock_compute_val_mse, mock_refit, mock_initial,
                               mock_cluster_dataset, basic_search_space, n_trials):
        """Test that n_trials parameter controls number of trials"""
        # Mock return values
        mock_model = _make_mock_model()
        mock_initial.return_value = mock_model
        mock_refit.return_value = (pd.DataFrame(), mock_model, None)
        mock_compute_val_mse.return_value = 0.4, 0.35, 0.05
        
        result = run_autotune(
            search_space=basic_search_space,
            train_dataset=mock_cluster_dataset,
            n_trials=n_trials,
            evaluate_all_orders=False,
            verbose=False,
            show_progress=False
        )
        
        # Check that the correct number of trials were run
        _, _, study, results_df = result
        assert len(study.trials) == n_trials
        assert len(results_df) == n_trials

    @patch('ciss_vae.training.autotune.train_vae_initial')
    @patch('ciss_vae.training.autotune.impute_and_refit_loop')
    @patch('ciss_vae.training.autotune.compute_val_mse')
    def test_evaluate_all_orders_true(self, mock_compute_val_mse, mock_refit, mock_initial,
                                     mock_cluster_dataset, basic_search_space):
        """Test evaluate_all_orders=True functionality"""
        # Mock return values
        mock_model = _make_mock_model()
        mock_initial.return_value = mock_model
        mock_refit.return_value = (pd.DataFrame(), mock_model, None)
        mock_compute_val_mse.return_value = 0.2, 0.15, 0.05
        
        result = run_autotune(
            search_space=basic_search_space,
            train_dataset=mock_cluster_dataset,
            n_trials=2,
            evaluate_all_orders=True,
            max_exhaustive_orders=50,
            verbose=False,
            show_progress=False
        )
        
        assert result is not None
        # When evaluate_all_orders=True, should potentially run more model evaluations
        # The exact behavior depends on the number of layer combinations

    @patch('ciss_vae.training.autotune.train_vae_initial')
    @patch('ciss_vae.training.autotune.impute_and_refit_loop')
    @patch('ciss_vae.training.autotune.compute_val_mse')
    def test_evaluate_all_orders_false(self, mock_compute_val_mse, mock_refit, mock_initial,
                                      mock_cluster_dataset, basic_search_space):
        """Test evaluate_all_orders=False functionality"""
        # Mock return values
        mock_model = _make_mock_model()
        mock_initial.return_value = mock_model
        mock_refit.return_value = (pd.DataFrame(), mock_model, None)
        mock_compute_val_mse.return_value = 0.35, 0.30, 0.05
        
        result = run_autotune(
            search_space=basic_search_space,
            train_dataset=mock_cluster_dataset,
            n_trials=2,
            evaluate_all_orders=False,
            verbose=False,
            show_progress=False
        )
        
        assert result is not None

    @patch('ciss_vae.training.autotune.train_vae_initial')
    @patch('ciss_vae.training.autotune.impute_and_refit_loop')
    @patch('ciss_vae.training.autotune.compute_val_mse')
    def test_return_format_without_history(self, mock_compute_val_mse, mock_refit, mock_initial,
                                          mock_cluster_dataset, basic_search_space):
        """Test return format when return_history=False"""
        # Mock return values
        mock_model = _make_mock_model()
        mock_initial.return_value = mock_model
        mock_refit.return_value = (pd.DataFrame({'col1': [1, 2, 3]}), mock_model, None)
        mock_compute_val_mse.return_value = 0.25, 0.2, 0.05
        
        result = run_autotune(
            search_space=basic_search_space,
            train_dataset=mock_cluster_dataset,
            n_trials=2,
            return_history=False,
            verbose=False,
            show_progress=False
        )
        
        assert isinstance(result, tuple)
        assert len(result) == 4  # Should not include history
        
        best_imputed_df, best_model, study, results_df = result
        assert isinstance(best_imputed_df, pd.DataFrame)
        assert best_model is not None
        assert isinstance(study, optuna.study.Study)
        assert isinstance(results_df, pd.DataFrame)
        
        # Check that study has the expected number of trials
        assert len(study.trials) == 2

    @patch('ciss_vae.training.autotune.train_vae_initial')
    @patch('ciss_vae.training.autotune.impute_and_refit_loop')
    @patch('ciss_vae.training.autotune.compute_val_mse')
    def test_return_format_with_history(self, mock_compute_val_mse, mock_refit, mock_initial,
                                       mock_cluster_dataset, basic_search_space):
        """Test return format when return_history=True"""
        # Mock return values
        mock_model = _make_mock_model()
        mock_history_df = pd.DataFrame({'epoch': [1, 2, 3], 'loss': [0.5, 0.4, 0.3]})
        mock_model.training_history_ = mock_history_df

        mock_initial.return_value = (mock_model, mock_history_df)
        mock_refit.return_value = (pd.DataFrame({'col1': [1, 2, 3]}), mock_model, None)
        mock_compute_val_mse.return_value = 0.15, 0.1, 0.05
        
        result = run_autotune(
            search_space=basic_search_space,
            train_dataset=mock_cluster_dataset,
            n_trials=2,
            return_history=True,
            verbose=False,
            show_progress=False
        )
        
        assert isinstance(result, tuple)
        assert len(result) == 5  # Should include history
        
        best_imputed_df, best_model, study, results_df, history_df = result
        assert isinstance(best_imputed_df, pd.DataFrame)
        assert best_model is not None
        assert isinstance(study, optuna.study.Study)
        assert isinstance(results_df, pd.DataFrame)
        assert isinstance(history_df, pd.DataFrame)

    @patch('ciss_vae.training.autotune.train_vae_initial')
    @patch('ciss_vae.training.autotune.impute_and_refit_loop')
    @patch('ciss_vae.training.autotune.compute_val_mse')
    def test_constant_layer_size_true(self, mock_compute_val_mse, mock_refit, mock_initial,
                                     mock_cluster_dataset):
        """Test constant_layer_size=True functionality"""
        search_space = SearchSpace(
            num_hidden_layers=3,
            hidden_dims=[64, 128],  # Should use same dim for all layers
            latent_dim=16,
            num_epochs=2,
            refit_loops=1,
            epochs_per_loop=1
        )
        
        # Mock return values
        mock_model = _make_mock_model()
        mock_initial.return_value = mock_model
        mock_refit.return_value = (pd.DataFrame(), mock_model, None)
        mock_compute_val_mse.return_value = 0.45, 0.4, 0.05
        
        result = run_autotune(
            search_space=search_space,
            train_dataset=mock_cluster_dataset,
            n_trials=2,
            constant_layer_size=True,
            verbose=False,
            show_progress=False
        )
        
        assert result is not None

    @patch('ciss_vae.training.autotune.train_vae_initial')
    @patch('ciss_vae.training.autotune.impute_and_refit_loop')
    @patch('ciss_vae.training.autotune.compute_val_mse')
    def test_max_exhaustive_orders_limit(self, mock_compute_val_mse, mock_refit, mock_initial,
                                        mock_cluster_dataset):
        """Test that max_exhaustive_orders limits the number of combinations tested"""
        search_space = SearchSpace(
            num_hidden_layers=4,  # This will create many combinations
            hidden_dims=64,
            latent_dim=16,
            num_shared_encode=[0, 1, 2, 3],  # Many options
            num_shared_decode=[0, 1, 2, 3],  # Many options
            num_epochs=1,
            refit_loops=1,
            epochs_per_loop=1
        )
        
        # Mock return values
        mock_model = _make_mock_model()
        mock_initial.return_value = mock_model
        mock_refit.return_value = (pd.DataFrame(), mock_model, None)
        mock_compute_val_mse.return_value = 0.3, 0.25, 0.05
        
        result = run_autotune(
            search_space=search_space,
            train_dataset=mock_cluster_dataset,
            n_trials=1,
            evaluate_all_orders=True,
            max_exhaustive_orders=5,  # Limit combinations
            verbose=False,
            show_progress=False
        )
        
        assert result is not None

    @patch('ciss_vae.training.autotune.train_vae_initial')
    @patch('ciss_vae.training.autotune.impute_and_refit_loop')
    @patch('ciss_vae.training.autotune.compute_val_mse')
    def test_seed_reproducibility(self, mock_compute_val_mse, mock_refit, mock_initial,
                                 mock_cluster_dataset, basic_search_space):
        """Test that same seed produces consistent results"""
        # Mock return values - make them deterministic
        mock_model = _make_mock_model()
        mock_initial.return_value = mock_model
        mock_refit.return_value = (pd.DataFrame(), mock_model, None)
        mock_compute_val_mse.return_value = 0.333, 0.3, 0.033
        
        # Run with same seed twice
        result1 = run_autotune(
            search_space=basic_search_space,
            train_dataset=mock_cluster_dataset,
            n_trials=2,
            seed=42,
            verbose=False,
            show_progress=False
        )
        
        result2 = run_autotune(
            search_space=basic_search_space,
            train_dataset=mock_cluster_dataset,
            n_trials=2,
            seed=42,
            verbose=False,
            show_progress=False
        )
        
        # Should get same number of trials
        _, _, study1, results_df1 = result1
        _, _, study2, results_df2 = result2
        
        assert len(study1.trials) == len(study2.trials)
        assert len(results_df1) == len(results_df2)

    def test_parameter_validation(self, mock_cluster_dataset):
        """Test that invalid parameters raise appropriate errors"""
        # Test with invalid search space parameter types
        with pytest.raises((ValueError, TypeError)):
            invalid_search_space = SearchSpace(
                latent_dim="invalid_string",  # Should be int, tuple, or list
            )
            run_autotune(
                search_space=invalid_search_space,
                train_dataset=mock_cluster_dataset,
                n_trials=1
            )


    def test_actual_optimization_minimal(self, mock_cluster_dataset):
        """Test actual optimization with minimal real parameters"""
        # This test runs actual autotune with minimal settings
        minimal_search_space = SearchSpace(
            num_hidden_layers=1,
            hidden_dims=16,  # Small fixed size
            latent_dim=[4, 8],  # Small range
            lr=0.01,  # Fixed
            num_epochs=1,  # Very small
            batch_size=4000,  # Small
            refit_loops=1,  # Minimal
            epochs_per_loop=1,  # Minimal
            num_shared_encode=0,  # Fixed
            num_shared_decode=1,  # Fixed
        )
        
        result = run_autotune(
            search_space=minimal_search_space,
            train_dataset=mock_cluster_dataset,
            n_trials=2,
            evaluate_all_orders=False,
            study_name="vae_autotune",
            verbose=False,
            show_progress=False,
            device_preference="cpu",
            load_if_exists = False,
        )
        
        assert result is not None
        best_imputed_df, best_model, study, results_df = result
        assert best_model is not None
        assert len(study.trials) == 2

