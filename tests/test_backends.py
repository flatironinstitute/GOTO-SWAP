"""
Tests for GOTO-SWAP backend system.
"""

import pytest
import numpy as np
from goto_swap import get_backend, wasserstein_distance_loss_images, emd_1d_sorted_differentiable


class TestBackendSystem:
    """Test the backend system functionality."""
    
    def test_backend_factory(self):
        """Test backend factory function."""
        # Test invalid backend
        with pytest.raises(ValueError, match="Unsupported backend"):
            get_backend("invalid_backend")
    
    def test_backend_interface(self, test_data_small):
        """Test that backends implement the required interface."""
        x, x_true = test_data_small
        
        # Test available backends
        for backend_name in ["jax", "pytorch"]:
            try:
                backend = get_backend(backend_name)
                
                # Test interface methods
                assert hasattr(backend, 'get_backend_name')
                assert hasattr(backend, 'wasserstein_distance_loss_images')
                assert backend.get_backend_name() == backend_name
                
            except ImportError:
                # Backend not available, skip test
                pytest.skip(f"{backend_name} backend not available")


@pytest.mark.jax
@pytest.mark.backend
class TestJAXBackend:
    """Test JAX backend functionality."""
    
    def test_jax_backend_creation(self):
        """Test JAX backend can be created."""
        try:
            backend = get_backend("jax")
            assert backend.get_backend_name() == "jax"
        except ImportError:
            pytest.skip("JAX backend not available")
    
    def test_jax_wasserstein_loss(self, test_data_small):
        """Test JAX backend Wasserstein loss computation."""
        try:
            import jax.numpy as jnp
            backend = get_backend("jax")
            
            x, x_true = test_data_small
            x_jax = jnp.array(x)
            x_true_jax = jnp.array(x_true)
            
            loss = backend.wasserstein_distance_loss_images(x_jax, x_true_jax, num_rotations=5)
            
            # Check output properties
            assert isinstance(loss, jnp.ndarray)
            assert loss.shape == ()  # Scalar
            assert not np.isnan(float(loss))
            assert not np.isinf(float(loss))
            assert float(loss) >= 0  # Loss should be non-negative
            
        except ImportError:
            pytest.skip("JAX backend not available")
    
    def test_jax_convenience_function(self, test_data_small):
        """Test convenience function with JAX backend."""
        try:
            import jax.numpy as jnp
            x, x_true = test_data_small
            x_jax = jnp.array(x)
            x_true_jax = jnp.array(x_true)
            
            loss = wasserstein_distance_loss_images(x_jax, x_true_jax, num_rotations=5)
            
            assert isinstance(loss, jnp.ndarray)
            assert not np.isnan(float(loss))
            
        except ImportError:
            pytest.skip("JAX backend not available")
    
    def test_jax_emd_1d(self):
        """Test JAX backend EMD 1D computation."""
        try:
            import jax.numpy as jnp
            backend = get_backend("jax")
            
            # Create simple test data for EMD
            u = jnp.array([1.0, 2.0, 3.0])
            v = jnp.array([1.5, 2.5, 3.5])
            u_weights = jnp.array([0.3, 0.4, 0.3])
            v_weights = jnp.array([0.2, 0.5, 0.3])
            
            loss = backend.emd_1d_sorted_differentiable(u, v, u_weights, v_weights)
            
            # Check output properties
            assert isinstance(loss, jnp.ndarray)
            assert loss.shape == ()  # Scalar
            assert not np.isnan(float(loss))
            assert not np.isinf(float(loss))
            assert float(loss) >= 0  # Loss should be non-negative
            
        except ImportError:
            pytest.skip("JAX backend not available")
    
    def test_jax_emd_convenience_function(self):
        """Test EMD convenience function with JAX backend."""
        try:
            import jax.numpy as jnp
            u = jnp.array([1.0, 2.0, 3.0])
            v = jnp.array([1.5, 2.5, 3.5])
            u_weights = jnp.array([0.3, 0.4, 0.3])
            v_weights = jnp.array([0.2, 0.5, 0.3])
            
            loss = emd_1d_sorted_differentiable(u, v, u_weights, v_weights)
            
            assert isinstance(loss, jnp.ndarray)
            assert not np.isnan(float(loss))
            
        except ImportError:
            pytest.skip("JAX backend not available")


@pytest.mark.pytorch
@pytest.mark.backend
class TestPyTorchBackend:
    """Test PyTorch backend functionality."""
    
    def test_pytorch_backend_creation(self):
        """Test PyTorch backend can be created."""
        try:
            backend = get_backend("pytorch")
            assert backend.get_backend_name() == "pytorch"
        except ImportError:
            pytest.skip("PyTorch backend not available")
    
    def test_pytorch_wasserstein_loss(self, test_data_small):
        """Test PyTorch backend Wasserstein loss computation."""
        try:
            import torch
            backend = get_backend("pytorch")
            
            x, x_true = test_data_small
            x_torch = torch.from_numpy(x)
            x_true_torch = torch.from_numpy(x_true)
            
            loss = backend.wasserstein_distance_loss_images(x_torch, x_true_torch, num_rotations=5)
            
            # Check output properties
            assert isinstance(loss, torch.Tensor)
            assert loss.shape == ()  # Scalar
            assert not torch.isnan(loss)
            assert not torch.isinf(loss)
            assert float(loss) >= 0  # Loss should be non-negative
            
        except ImportError:
            pytest.skip("PyTorch backend not available")
    
    def test_pytorch_convenience_function(self, test_data_small):
        """Test convenience function with PyTorch backend."""
        try:
            import torch
            x, x_true = test_data_small
            x_torch = torch.from_numpy(x)
            x_true_torch = torch.from_numpy(x_true)
            
            loss = wasserstein_distance_loss_images(x_torch, x_true_torch, backend="pytorch", num_rotations=5)
            
            assert isinstance(loss, torch.Tensor)
            assert not torch.isnan(loss)
            
        except ImportError:
            pytest.skip("PyTorch backend not available")
    
    def test_pytorch_emd_1d(self):
        """Test PyTorch backend EMD 1D computation."""
        try:
            import torch
            backend = get_backend("pytorch")
            
            # Create simple test data for EMD
            u = torch.tensor([1.0, 2.0, 3.0])
            v = torch.tensor([1.5, 2.5, 3.5])
            u_weights = torch.tensor([0.3, 0.4, 0.3])
            v_weights = torch.tensor([0.2, 0.5, 0.3])
            
            loss = backend.emd_1d_sorted_differentiable(u, v, u_weights, v_weights)
            
            # Check output properties
            assert isinstance(loss, torch.Tensor)
            assert loss.shape == ()  # Scalar
            assert not torch.isnan(loss)
            assert not torch.isinf(loss)
            assert float(loss) >= 0  # Loss should be non-negative
            
        except ImportError:
            pytest.skip("PyTorch backend not available")
    
    def test_pytorch_emd_convenience_function(self):
        """Test EMD convenience function with PyTorch backend."""
        try:
            import torch
            u = torch.tensor([1.0, 2.0, 3.0])
            v = torch.tensor([1.5, 2.5, 3.5])
            u_weights = torch.tensor([0.3, 0.4, 0.3])
            v_weights = torch.tensor([0.2, 0.5, 0.3])
            
            loss = emd_1d_sorted_differentiable(u, v, u_weights, v_weights, backend="pytorch")
            
            assert isinstance(loss, torch.Tensor)
            assert not torch.isnan(loss)
            
        except ImportError:
            pytest.skip("PyTorch backend not available")


class TestDirectEMDFunctions:
    """Test direct EMD function imports."""
    
    def test_jax_direct_emd_import(self):
        """Test direct import of JAX EMD function."""
        try:
            from goto_swap.backends.jax import emd_1d_sorted_jax_differentiable
            import jax.numpy as jnp
            
            u = jnp.array([1.0, 2.0, 3.0])
            v = jnp.array([1.5, 2.5, 3.5])
            u_weights = jnp.array([0.3, 0.4, 0.3])
            v_weights = jnp.array([0.2, 0.5, 0.3])
            
            loss = emd_1d_sorted_jax_differentiable(u, v, u_weights, v_weights)
            
            assert isinstance(loss, jnp.ndarray)
            assert not np.isnan(float(loss))
            assert float(loss) >= 0
            
        except ImportError:
            pytest.skip("JAX backend not available")
    
    def test_pytorch_direct_emd_import(self):
        """Test direct import of PyTorch EMD function."""
        try:
            from goto_swap.backends.pytorch import emd_1d_sorted_pytorch_differentiable
            import torch
            
            u = torch.tensor([1.0, 2.0, 3.0])
            v = torch.tensor([1.5, 2.5, 3.5])
            u_weights = torch.tensor([0.3, 0.4, 0.3])
            v_weights = torch.tensor([0.2, 0.5, 0.3])
            
            loss = emd_1d_sorted_pytorch_differentiable(u, v, u_weights, v_weights)
            
            assert isinstance(loss, torch.Tensor)
            assert not torch.isnan(loss)
            assert float(loss) >= 0
            
        except ImportError:
            pytest.skip("PyTorch backend not available")
    
    def test_direct_emd_consistency(self):
        """Test that direct EMD functions produce consistent results."""
        try:
            from goto_swap.backends.jax import emd_1d_sorted_jax_differentiable
            from goto_swap.backends.pytorch import emd_1d_sorted_pytorch_differentiable
            import jax.numpy as jnp
            import torch
            
            # Create test data
            u_np = np.array([1.0, 2.0, 3.0])
            v_np = np.array([1.5, 2.5, 3.5])
            u_weights_np = np.array([0.3, 0.4, 0.3])
            v_weights_np = np.array([0.2, 0.5, 0.3])
            
            # JAX version
            u_jax = jnp.array(u_np)
            v_jax = jnp.array(v_np)
            u_weights_jax = jnp.array(u_weights_np)
            v_weights_jax = jnp.array(v_weights_np)
            
            loss_jax = emd_1d_sorted_jax_differentiable(u_jax, v_jax, u_weights_jax, v_weights_jax)
            
            # PyTorch version
            u_torch = torch.from_numpy(u_np)
            v_torch = torch.from_numpy(v_np)
            u_weights_torch = torch.from_numpy(u_weights_np)
            v_weights_torch = torch.from_numpy(v_weights_np)
            
            loss_pytorch = emd_1d_sorted_pytorch_differentiable(u_torch, v_torch, u_weights_torch, v_weights_torch)
            
            # Results should be numerically close
            diff = abs(float(loss_jax) - float(loss_pytorch))
            assert diff < 1e-6, f"JAX and PyTorch results differ by {diff}"
            
        except ImportError as e:
            pytest.skip(f"Backend comparison not available: {e}")


class TestBackendCompatibility:
    """Test backend compatibility and consistency."""
    
    def test_legacy_compatibility(self, test_data_small):
        """Test legacy compatibility with deprecation warning."""
        try:
            import jax.numpy as jnp
            from goto_swap.base import wasserstein_distance_loss_images as legacy_func
            
            x, x_true = test_data_small
            x_jax = jnp.array(x)
            x_true_jax = jnp.array(x_true)
            
            # Should show deprecation warning
            with pytest.warns(DeprecationWarning):
                loss = legacy_func(x_jax, x_true_jax, num_rotations=5)
            
            assert not np.isnan(float(loss))
            
        except ImportError:
            pytest.skip("JAX backend not available for legacy test")
    
    @pytest.mark.parametrize("backend_name", ["jax", "pytorch"])
    def test_backend_consistency(self, test_data_small, backend_name):
        """Test that backends produce consistent results for the same input."""
        try:
            backend = get_backend(backend_name)
            x, x_true = test_data_small
            
            if backend_name == "jax":
                import jax.numpy as jnp
                x_backend = jnp.array(x)
                x_true_backend = jnp.array(x_true)
            else:  # pytorch
                import torch
                x_backend = torch.from_numpy(x)
                x_true_backend = torch.from_numpy(x_true)
            
            # Test multiple runs produce consistent results
            loss1 = backend.wasserstein_distance_loss_images(x_backend, x_true_backend, num_rotations=5)
            loss2 = backend.wasserstein_distance_loss_images(x_backend, x_true_backend, num_rotations=5)
            
            # Results should be identical (deterministic)
            if backend_name == "jax":
                assert float(loss1) == float(loss2)
            else:  # pytorch
                assert torch.allclose(loss1, loss2)
                
        except ImportError:
            pytest.skip(f"{backend_name} backend not available")


class TestBackendErrors:
    """Test error handling in backend system."""
    
    def test_invalid_backend_error(self):
        """Test error handling for invalid backend names."""
        with pytest.raises(ValueError, match="Unsupported backend"):
            get_backend("invalid")
    
    def test_missing_backend_import_error(self):
        """Test import error when backend dependencies are missing."""
        # This test assumes at least one backend is missing
        # In practice, you'd mock the import to test this
        pass
    
    def test_invalid_input_shapes(self, test_data_small):
        """Test error handling for invalid input shapes."""
        x, x_true = test_data_small
        
        # Test with mismatched shapes
        x_wrong = np.random.randn(1, 16, 16).astype(np.float32)
        
        try:
            import jax.numpy as jnp
            backend = get_backend("jax")
            
            x_jax = jnp.array(x)
            x_wrong_jax = jnp.array(x_wrong)
            
            # This should work (broadcasting) or raise an appropriate error
            # The exact behavior depends on implementation
            loss = backend.wasserstein_distance_loss_images(x_jax, x_wrong_jax, num_rotations=5)
            assert not np.isnan(float(loss))
            
        except ImportError:
            pytest.skip("JAX backend not available")
        except Exception as e:
            # If it raises an error, it should be a meaningful one
            assert "shape" in str(e).lower() or "dimension" in str(e).lower()
