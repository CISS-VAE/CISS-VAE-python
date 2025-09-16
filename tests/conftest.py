import pytest
import numpy as np
import pandas as pd
import tempfile
import os
import torch
from unittest.mock import MagicMock

# Add this import
from ciss_vae.classes.cluster_dataset import ClusterDataset

@pytest.fixture
def sample_data():
    """Create sample data for testing with better clustering structure"""
    np.random.seed(42)
    
    # Create 2 well-separated clusters
    cluster1 = np.random.multivariate_normal([0, 0], np.eye(2) * 0.3, 50)
    cluster2 = np.random.multivariate_normal([3, 3], np.eye(2) * 0.3, 50)
    
    # Add noise features
    noise_features = np.random.randn(100, 18) * 0.5
    structured_data = np.vstack([cluster1, cluster2])
    data = np.hstack([structured_data, noise_features])
    
    df = pd.DataFrame(data, columns=[f'feature_{i}' for i in range(20)])
    
    # Add some missing values (not too many)
    mask = np.random.random((100, 20)) < 0.05
    df = df.mask(mask)
    
    return df

@pytest.fixture
def large_sample_data():
    """Create larger sample data for performance tests"""
    np.random.seed(42)
    data = pd.DataFrame(
        np.random.randn(1000, 50),
        columns=[f'feature_{i}' for i in range(50)]
    )
    mask = np.random.random((1000, 50)) < 0.05
    data = data.mask(mask)
    return data

@pytest.fixture
def minimal_params():
    """Minimal parameters for fast testing"""
    return {
        'hidden_dims': [32, 16],
        'latent_dim': 8,
        'epochs': 2,
        'batch_size': 32,
        'max_loops': 2,
        'patience': 1,
        'epochs_per_loop': 1,
        'verbose': False,
        'n_clusters': 2,  # Force 2 clusters for predictability
        'layer_order_enc': ['unshared', 'shared'],  # 3 layers  
        'layer_order_dec': ['shared', 'unshared'],
    }

@pytest.fixture
def temp_dir():
    """Create temporary directory for test files"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir

## added mock_cluster_datast
@pytest.fixture
def mock_cluster_dataset(sample_data):
    """Create a mock ClusterDataset for testing"""
    # Create mock cluster labels
    n_samples = len(sample_data)
    cluster_labels = torch.randint(0, 3, (n_samples,))  # 3 clusters
    
    # Create mock dataset
    dataset = ClusterDataset(sample_data, cluster_labels)
    return dataset