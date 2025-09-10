"""
Pytest configuration and fixtures for GOTO-SWAP tests.
"""

import pytest
import numpy as np


@pytest.fixture
def test_data():
    """Create test data for backend testing."""
    batch_size, height, width = 2, 32, 32
    x = np.random.randn(batch_size, height, width).astype(np.float32)
    x_true = np.random.randn(batch_size, height, width).astype(np.float32)
    return x, x_true


@pytest.fixture
def test_data_small():
    """Create smaller test data for faster testing."""
    batch_size, height, width = 1, 16, 16
    x = np.random.randn(batch_size, height, width).astype(np.float32)
    x_true = np.random.randn(batch_size, height, width).astype(np.float32)
    return x, x_true


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "jax: mark test as requiring JAX backend"
    )
    config.addinivalue_line(
        "markers", "pytorch: mark test as requiring PyTorch backend"
    )
    config.addinivalue_line(
        "markers", "backend: mark test as backend-specific"
    )
