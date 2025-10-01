"""
Systematic tests for emd_1d_sorted_differentiable functionality.
Based on test patterns from GOTO repository.
"""

import pytest
import numpy as np
from goto_swap import emd_1d_sorted_differentiable, get_backend


@pytest.mark.backend
class TestEMD1D:
    """Test emd_1d_sorted_differentiable functionality."""
    
    @pytest.fixture
    def test_data_uniform(self):
        """Create test data with uniform weights."""
        n = 50
        u = np.sort(np.random.randn(n))
        v = np.sort(np.random.randn(n))
        u_weights = np.ones(n) / n
        v_weights = np.ones(n) / n
        return u, v, u_weights, v_weights
    
    @pytest.fixture
    def test_data_non_uniform(self):
        """Create test data with non-uniform weights."""
        n = 30
        u = np.sort(np.random.randn(n))
        v = np.sort(np.random.randn(n))
        u_weights = np.random.rand(n)
        u_weights = u_weights / u_weights.sum()
        v_weights = np.random.rand(n)
        v_weights = v_weights / v_weights.sum()
        return u, v, u_weights, v_weights
    
    def test_emd_basic_functionality(self, test_data_uniform):
        """Test basic EMD computation."""
        u, v, u_weights, v_weights = test_data_uniform
        
        try:
            import jax.numpy as jnp
            u_jax = jnp.array(u)
            v_jax = jnp.array(v)
            u_weights_jax = jnp.array(u_weights)
            v_weights_jax = jnp.array(v_weights)
            
            loss = emd_1d_sorted_differentiable(u_jax, v_jax, u_weights_jax, v_weights_jax)
            
            assert isinstance(loss, jnp.ndarray)
            assert loss.shape == ()  # Scalar
            assert not np.isnan(float(loss))
            assert not np.isinf(float(loss))
            assert float(loss) >= 0  # Non-negative loss
            
        except ImportError:
            pytest.skip("JAX backend not available")
    
    def test_emd_pytorch_functionality(self, test_data_uniform):
        """Test PyTorch EMD computation."""
        u, v, u_weights, v_weights = test_data_uniform
        
        try:
            import torch
            u_torch = torch.from_numpy(u)
            v_torch = torch.from_numpy(v)
            u_weights_torch = torch.from_numpy(u_weights)
            v_weights_torch = torch.from_numpy(v_weights)
            
            loss = emd_1d_sorted_differentiable(u_torch, v_torch, u_weights_torch, v_weights_torch, backend="pytorch")
            
            assert isinstance(loss, torch.Tensor)
            assert loss.shape == ()  # Scalar
            assert not torch.isnan(loss)
            assert not torch.isinf(loss)
            assert float(loss) >= 0  # Non-negative loss
            
        except ImportError:
            pytest.skip("PyTorch backend not available")
    
    def test_emd_identical_distributions(self, test_data_uniform):
        """Test that identical distributions produce zero loss."""
        u, _, u_weights, _ = test_data_uniform
        
        try:
            import jax.numpy as jnp
            u_jax = jnp.array(u)
            u_weights_jax = jnp.array(u_weights)
            
            # Test with identical distributions
            loss = emd_1d_sorted_differentiable(u_jax, u_jax, u_weights_jax, u_weights_jax)
            assert float(loss) < 1e-6, f"Identical distributions should have near-zero loss, got {float(loss)}"
            
        except ImportError:
            pytest.skip("JAX backend not available")
    
    def test_emd_different_metrics(self, test_data_uniform):
        """Test EMD with different distance metrics."""
        u, v, u_weights, v_weights = test_data_uniform
        
        try:
            import jax.numpy as jnp
            u_jax = jnp.array(u)
            v_jax = jnp.array(v)
            u_weights_jax = jnp.array(u_weights)
            v_weights_jax = jnp.array(v_weights)
            
            # Test different metrics
            loss_sqeuclidean = emd_1d_sorted_differentiable(u_jax, v_jax, u_weights_jax, v_weights_jax, metric='sqeuclidean')
            loss_euclidean = emd_1d_sorted_differentiable(u_jax, v_jax, u_weights_jax, v_weights_jax, metric='euclidean')
            loss_cityblock = emd_1d_sorted_differentiable(u_jax, v_jax, u_weights_jax, v_weights_jax, metric='cityblock')
            
            # All should be non-negative
            assert float(loss_sqeuclidean) >= 0
            assert float(loss_euclidean) >= 0
            assert float(loss_cityblock) >= 0
            
            # Squared euclidean should be larger than euclidean for distances > 1
            # This is a basic sanity check
            
        except ImportError:
            pytest.skip("JAX backend not available")
    
    def test_emd_backend_consistency(self, test_data_uniform):
        """Test that JAX and PyTorch backends produce consistent results."""
        u, v, u_weights, v_weights = test_data_uniform
        
        try:
            import jax.numpy as jnp
            import torch
            
            # JAX computation
            u_jax = jnp.array(u)
            v_jax = jnp.array(v)
            u_weights_jax = jnp.array(u_weights)
            v_weights_jax = jnp.array(v_weights)
            loss_jax = emd_1d_sorted_differentiable(u_jax, v_jax, u_weights_jax, v_weights_jax)
            
            # PyTorch computation
            u_torch = torch.from_numpy(u)
            v_torch = torch.from_numpy(v)
            u_weights_torch = torch.from_numpy(u_weights)
            v_weights_torch = torch.from_numpy(v_weights)
            loss_pytorch = emd_1d_sorted_differentiable(u_torch, v_torch, u_weights_torch, v_weights_torch, backend="pytorch")
            
            # Results should be numerically close
            diff = abs(float(loss_jax) - float(loss_pytorch))
            assert diff < 1e-6, f"JAX and PyTorch results differ by {diff}"
            
        except ImportError as e:
            pytest.skip(f"Backend comparison not available: {e}")
    
    def test_emd_gradient_computation(self, test_data_uniform):
        """Test that gradients can be computed."""
        u, v, u_weights, v_weights = test_data_uniform
        
        try:
            import jax
            import jax.numpy as jnp
            
            u_jax = jnp.array(u)
            v_jax = jnp.array(v)
            u_weights_jax = jnp.array(u_weights)
            v_weights_jax = jnp.array(v_weights)
            
            # Test gradient computation w.r.t. u_weights
            grad_fn = jax.grad(emd_1d_sorted_differentiable, argnums=2)
            gradients = grad_fn(u_jax, v_jax, u_weights_jax, v_weights_jax)
            
            assert gradients.shape == u_weights_jax.shape
            assert not np.any(np.isnan(gradients))
            
        except ImportError:
            pytest.skip("JAX backend not available")
    
    def test_emd_non_uniform_weights(self, test_data_non_uniform):
        """Test EMD with non-uniform weights."""
        u, v, u_weights, v_weights = test_data_non_uniform
        
        try:
            import jax.numpy as jnp
            u_jax = jnp.array(u)
            v_jax = jnp.array(v)
            u_weights_jax = jnp.array(u_weights)
            v_weights_jax = jnp.array(v_weights)
            
            loss = emd_1d_sorted_differentiable(u_jax, v_jax, u_weights_jax, v_weights_jax)
            
            assert not np.isnan(float(loss))
            assert float(loss) >= 0
            
        except ImportError:
            pytest.skip("JAX backend not available")
    
    def test_emd_different_sizes(self):
        """Test EMD with different array sizes."""
        try:
            import jax.numpy as jnp
            
            # Test with different sizes
            n1, n2 = 20, 30
            u = np.sort(np.random.randn(n1))
            v = np.sort(np.random.randn(n2))
            u_weights = np.random.rand(n1)
            u_weights = u_weights / u_weights.sum()
            v_weights = np.random.rand(n2)
            v_weights = v_weights / v_weights.sum()
            
            u_jax = jnp.array(u)
            v_jax = jnp.array(v)
            u_weights_jax = jnp.array(u_weights)
            v_weights_jax = jnp.array(v_weights)
            
            loss = emd_1d_sorted_differentiable(u_jax, v_jax, u_weights_jax, v_weights_jax)
            
            assert not np.isnan(float(loss))
            assert float(loss) >= 0
            
        except ImportError:
            pytest.skip("JAX backend not available")
    
    def test_emd_edge_cases(self):
        """Test edge cases for EMD computation."""
        try:
            import jax.numpy as jnp
            
            # Test with single points
            u_single = jnp.array([1.0])
            v_single = jnp.array([2.0])
            u_weights_single = jnp.array([1.0])
            v_weights_single = jnp.array([1.0])
            
            loss_single = emd_1d_sorted_differentiable(u_single, v_single, u_weights_single, v_weights_single)
            assert not np.isnan(float(loss_single))
            assert float(loss_single) >= 0
            
            # Test with identical points
            loss_identical = emd_1d_sorted_differentiable(u_single, u_single, u_weights_single, u_weights_single)
            assert float(loss_identical) < 1e-6
            
        except ImportError:
            pytest.skip("JAX backend not available")
    
    def test_emd_tolerance_parameter(self, test_data_uniform):
        """Test EMD with different tolerance parameters."""
        u, v, u_weights, v_weights = test_data_uniform
        
        try:
            import jax.numpy as jnp
            u_jax = jnp.array(u)
            v_jax = jnp.array(v)
            u_weights_jax = jnp.array(u_weights)
            v_weights_jax = jnp.array(v_weights)
            
            # Test with different tolerances
            loss_tol_1e_6 = emd_1d_sorted_differentiable(u_jax, v_jax, u_weights_jax, v_weights_jax, tol=1e-6)
            loss_tol_1e_12 = emd_1d_sorted_differentiable(u_jax, v_jax, u_weights_jax, v_weights_jax, tol=1e-12)
            
            # Results should be very close
            diff = abs(float(loss_tol_1e_6) - float(loss_tol_1e_12))
            assert diff < 1e-10, f"Results with different tolerances differ by {diff}"
            
        except ImportError:
            pytest.skip("JAX backend not available")
    
    def test_emd_jit_compilation(self, test_data_uniform):
        """Test JIT compilation of EMD function."""
        u, v, u_weights, v_weights = test_data_uniform
        
        try:
            import jax
            import jax.numpy as jnp
            
            u_jax = jnp.array(u)
            v_jax = jnp.array(v)
            u_weights_jax = jnp.array(u_weights)
            v_weights_jax = jnp.array(v_weights)
            
            # JIT compile the function
            jit_emd = jax.jit(emd_1d_sorted_differentiable, static_argnames=['metric', 'tol'])
            
            # Test JIT compiled version
            loss_jit = jit_emd(u_jax, v_jax, u_weights_jax, v_weights_jax)
            
            assert not np.isnan(float(loss_jit))
            assert float(loss_jit) >= 0
            
        except ImportError:
            pytest.skip("JAX backend not available")
    
    def test_emd_consistency_across_runs(self, test_data_uniform):
        """Test that EMD produces consistent results across multiple runs."""
        u, v, u_weights, v_weights = test_data_uniform
        
        try:
            import jax.numpy as jnp
            u_jax = jnp.array(u)
            v_jax = jnp.array(v)
            u_weights_jax = jnp.array(u_weights)
            v_weights_jax = jnp.array(v_weights)
            
            # Run multiple times
            losses = []
            for _ in range(5):
                loss = emd_1d_sorted_differentiable(u_jax, v_jax, u_weights_jax, v_weights_jax)
                losses.append(float(loss))
            
            # All results should be identical (deterministic)
            assert all(abs(losses[0] - loss) < 1e-10 for loss in losses), "EMD should be deterministic"
            
        except ImportError:
            pytest.skip("JAX backend not available")
    
    def test_emd_vectorization(self, test_data_uniform):
        """Test vectorized EMD computation."""
        u, v, u_weights, v_weights = test_data_uniform
        
        try:
            import jax
            import jax.numpy as jnp
            
            # Create batch of data
            batch_size = 3
            u_batch = jnp.array([u] * batch_size)
            v_batch = jnp.array([v] * batch_size)
            u_weights_batch = jnp.array([u_weights] * batch_size)
            v_weights_batch = jnp.array([v_weights] * batch_size)
            
            # Vectorized computation
            vmap_emd = jax.vmap(emd_1d_sorted_differentiable, in_axes=(0, 0, 0, 0))
            losses = vmap_emd(u_batch, v_batch, u_weights_batch, v_weights_batch)
            
            assert losses.shape == (batch_size,)
            assert not np.any(np.isnan(losses))
            assert np.all(losses >= 0)
            
        except ImportError:
            pytest.skip("JAX backend not available")
