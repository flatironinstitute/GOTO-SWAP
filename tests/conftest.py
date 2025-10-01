"""
Pytest configuration and fixtures for GOTO-SWAP tests.
"""

import pytest
import numpy as np


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--backend",
        action="store",
        default=None,
        choices=["jax", "pytorch"],
        help="Run tests for specific backend only"
    )


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
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection based on backend option."""
    backend = config.getoption("--backend")
    if backend is None:
        return
    
    # Filter tests based on backend
    selected_items = []
    deselected_items = []
    
    for item in items:
        # Check if test is marked for the specific backend
        if backend == "jax" and item.get_closest_marker("jax"):
            selected_items.append(item)
        elif backend == "pytorch" and item.get_closest_marker("pytorch"):
            selected_items.append(item)
        elif backend == "jax" and not item.get_closest_marker("pytorch"):
            # Include non-backend-specific tests for JAX
            selected_items.append(item)
        elif backend == "pytorch" and not item.get_closest_marker("jax"):
            # Include non-backend-specific tests for PyTorch
            selected_items.append(item)
        else:
            deselected_items.append(item)
    
    # Update the items list
    config.hook.pytest_deselected(items=deselected_items)
    items[:] = selected_items


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


@pytest.fixture
def test_emd_data():
    """Create test data for EMD testing."""
    n = 30
    u = np.sort(np.random.randn(n))
    v = np.sort(np.random.randn(n))
    u_weights = np.random.rand(n)
    u_weights = u_weights / u_weights.sum()
    v_weights = np.random.rand(n)
    v_weights = v_weights / v_weights.sum()
    return u, v, u_weights, v_weights