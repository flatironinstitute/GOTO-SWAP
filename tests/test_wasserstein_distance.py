"""
Systematic tests for wasserstein_distance_loss_images functionality.
"""

import pytest
import numpy as np
from goto_swap import wasserstein_distance_loss_images


@pytest.mark.backend
class TestWassersteinDistanceImages:
    """Test wasserstein_distance_loss_images functionality."""
    
    @pytest.fixture
    def test_images(self):
        """Create test images for Wasserstein distance computation."""
        batch_size, height, width = 2, 32, 32
        x = np.random.randn(batch_size, height, width).astype(np.float32)
        x_true = np.random.randn(batch_size, height, width).astype(np.float32)
        return x, x_true
    
    @pytest.fixture
    def test_images_large(self):
        """Create larger test images."""
        batch_size, height, width = 4, 64, 64
        x = np.random.randn(batch_size, height, width).astype(np.float32)
        x_true = np.random.randn(batch_size, height, width).astype(np.float32)
        return x, x_true
    
    def test_wasserstein_basic_functionality(self, test_images):
        """Test basic Wasserstein distance computation."""
        x, x_true = test_images
        
        # Test JAX backend
        try:
            import jax.numpy as jnp
            x_jax = jnp.array(x)
            x_true_jax = jnp.array(x_true)

            for projection_method in ['real_rotate', 'fourier_slice']:
                for method in ['cdf', 'icdf', 'dsort']:
                    loss = wasserstein_distance_loss_images(x_jax, x_true_jax, projection_method=projection_method, method=method, num_rotations=10).sum()
                    
                    assert isinstance(loss, jnp.ndarray)
                    assert loss.shape == ()  # Scalar
                    assert not np.isnan(float(loss))
                    assert not np.isinf(float(loss))
                    assert float(loss) >= 0  # Non-negative loss

            
        except ImportError:
            pytest.skip("JAX backend not available")
    
    def test_wasserstein_pytorch_functionality(self, test_images):
        """Test PyTorch Wasserstein distance computation."""
        x, x_true = test_images
        
        try:
            import torch
            x_torch = torch.from_numpy(x)
            x_true_torch = torch.from_numpy(x_true)
            
            loss = wasserstein_distance_loss_images(x_torch, x_true_torch, backend="pytorch", num_rotations=10)
            
            assert isinstance(loss, torch.Tensor)
            assert loss.shape == ()  # Scalar
            assert not torch.isnan(loss)
            assert not torch.isinf(loss)
            assert float(loss) >= 0  # Non-negative loss
            
        except ImportError:
            pytest.skip("PyTorch backend not available")
    
    def test_wasserstein_identical_images(self, test_images):
        """Test that identical images produce zero loss."""
        x, _ = test_images
        
        try:
            import jax.numpy as jnp
            x_jax = jnp.array(x)
            
            # Test with identical images
            for projection_method in ['real_rotate', 'fourier_slice']:
                for method in ['cdf', 'icdf', 'dsort']:
                    loss = wasserstein_distance_loss_images(x_jax, x_jax, projection_method=projection_method, method=method, num_rotations=10).sum()
                    assert float(loss) < 1e-6, f"Identical images should have near-zero loss, got {float(loss)}"
            
        except ImportError:
            pytest.skip("JAX backend not available")
    
    def test_wasserstein_different_rotations(self, test_images):
        """Test that different rotation counts produce consistent results."""
        x, x_true = test_images
        
        try:
            import jax.numpy as jnp
            x_jax = jnp.array(x)
            x_true_jax = jnp.array(x_true)
            
            # Test with different rotation counts
            for projection_method in ['real_rotate', 'fourier_slice']:
                for method in ['cdf', 'icdf', 'dsort']:
                    loss_10 = wasserstein_distance_loss_images(x_jax, x_true_jax, projection_method=projection_method, method=method, num_rotations=10).sum()
                    loss_20 = wasserstein_distance_loss_images(x_jax, x_true_jax, projection_method=projection_method, method=method, num_rotations=20).sum()
                    loss_50 = wasserstein_distance_loss_images(x_jax, x_true_jax, projection_method=projection_method, method=method, num_rotations=50).sum()
            
                # More rotations should generally give more accurate results
                # but the results should be in the same ballpark
                assert not np.isnan(float(loss_10))
                assert not np.isnan(float(loss_20))
                assert not np.isnan(float(loss_50))
                
                # All should be non-negative
                assert float(loss_10) >= 0
                assert float(loss_20) >= 0
                assert float(loss_50) >= 0
            
        except ImportError:
            pytest.skip("JAX backend not available")
    
    def test_wasserstein_backend_consistency(self, test_images):
        """Test that JAX and PyTorch backends produce consistent results."""
        x, x_true = test_images
        
        try:
            import jax.numpy as jnp
            import torch
            
            # JAX computation
            x_jax = jnp.array(x)
            x_true_jax = jnp.array(x_true)
            
            # PyTorch computation
            x_torch = torch.from_numpy(x)
            x_true_torch = torch.from_numpy(x_true)
            loss_pytorch = wasserstein_distance_loss_images(x_torch, x_true_torch, backend="pytorch", num_rotations=10).sum()


            for projection_method in ['real_rotate', 'fourier_slice']:
                for method in ['cdf', 'icdf', 'dsort']:
                    loss_jax = wasserstein_distance_loss_images(x_jax, x_true_jax, projection_method=projection_method, method=method, num_rotations=10).sum()
            
                # Results should be numerically close
                diff = abs(float(loss_jax) - float(loss_pytorch))
                assert diff < 1e-5, f"JAX and PyTorch results differ by {diff}"
            
        except ImportError as e:
            pytest.skip(f"Backend comparison not available: {e}")
    
    def test_wasserstein_gradient_computation(self, test_images):
        """Test that gradients can be computed."""
        x, x_true = test_images
        
        try:
            import jax
            import jax.numpy as jnp
            
            x_jax = jnp.array(x)
            x_true_jax = jnp.array(x_true)
            
            # Test gradient computation
            grad_fn = jax.grad(lambda x_jax, x_true_jax, projection_method, method, num_rotations: jnp.sum(wasserstein_distance_loss_images(x_jax, x_true_jax, projection_method, method=method, num_rotations=10)), argnums=0)

            for projection_method in ['real_rotate', 'fourier_slice']:
                for method in ['cdf', 'icdf', 'dsort']:
                    gradients = grad_fn(x_jax, x_true_jax, projection_method=projection_method, method=method, num_rotations=10)
                    
                    assert gradients.shape == x_jax.shape
                    assert not np.any(np.isnan(gradients))

            
        except ImportError:
            pytest.skip("JAX backend not available")
    
    def test_wasserstein_different_image_sizes(self, test_images):
        """Test with different image sizes."""
        x, x_true = test_images
        
        try:
            import jax.numpy as jnp
            
            # Test with different sizes
            x_small = x[:, :16, :16]  # Smaller images
            x_true_small = x_true[:, :16, :16]
            
            x_jax_small = jnp.array(x_small)
            x_true_jax_small = jnp.array(x_true_small)

            for projection_method in ['real_rotate', 'fourier_slice']:
                for method in ['cdf', 'icdf', 'dsort']:
                    loss_small = wasserstein_distance_loss_images(x_jax_small, x_true_jax_small, projection_method=projection_method, method=method, num_rotations=10).sum()
                    
                    assert not np.isnan(float(loss_small))
                    assert float(loss_small) >= 0

            
        except ImportError:
            pytest.skip("JAX backend not available")
    
    def test_wasserstein_batch_processing(self, test_images_large):
        """Test batch processing with multiple images."""
        x, x_true = test_images_large
        
        try:
            import jax
            import jax.numpy as jnp
            from goto_swap import get_backend
            
            x_jax = jnp.array(x)
            x_true_jax = jnp.array(x_true)
            
            # Get backend and create a wrapper that adds batch dimension
            backend = get_backend("jax")
            
            def single_image_loss(x_single, x_true_single, projection_method, method, num_rotations):
                # Add batch dimension for single images
                x_batch = x_single[None, ...]  # (H, W) -> (1, H, W)
                x_true_batch = x_true_single[None, ...]
                return backend.wasserstein_distance_loss_images(x_batch, x_true_batch, projection_method, method, num_rotations).sum()
            
            # Vectorize over batch dimension
            vmap_loss = jax.vmap(single_image_loss, in_axes=(0, 0, None, None, None))
            num_rotations=10
            for projection_method in ['real_rotate', 'fourier_slice']:
                for method in ['cdf', 'icdf', 'dsort']:
                    losses = vmap_loss(x_jax, x_true_jax, projection_method, method, num_rotations)  # 
                    
                    assert losses.shape == (x_jax.shape[0],)  # One loss per image pair
                    assert not np.any(np.isnan(losses))
                    assert np.all(losses >= 0)

            
        except ImportError:
            pytest.skip("JAX backend not available")
    
    def test_wasserstein_edge_cases(self):
        """Test edge cases for Wasserstein distance."""
        try:
            import jax
            import jax.numpy as jnp
            
            # Test with very small images
            key = jax.random.PRNGKey(0)
            x_tiny = jax.random.normal(key, (1, 4, 4))
            key, subkey = jax.random.split(key)
            x_true_tiny = jax.random.normal(subkey, (1, 4, 4))

            for projection_method in ['real_rotate', 'fourier_slice']:
                for method in ['cdf', 'icdf', 'dsort']:
                    loss_tiny = wasserstein_distance_loss_images(x_tiny, x_true_tiny, projection_method=projection_method, method=method, num_rotations=5).sum()
                    assert not np.isnan(float(loss_tiny))
                    assert float(loss_tiny) >= 0

                # Test with constant images
                x_constant = jnp.ones((1, 8, 8))
                x_true_constant = jnp.zeros((1, 8, 8))
                
                loss_constant = wasserstein_distance_loss_images(x_constant, x_true_constant, projection_method=projection_method, method=method, num_rotations=5).sum()
                assert not np.isnan(float(loss_constant))
                assert float(loss_constant) >= 0
            
        except ImportError:
            pytest.skip("JAX backend not available")
    
    def test_wasserstein_jit_compilation(self, test_images):
        """Test JIT compilation of Wasserstein distance."""
        try:
            import jax
            import jax.numpy as jnp
            from goto_swap import get_backend
            
            x_jax = jnp.array(test_images[0])
            x_true_jax = jnp.array(test_images[1])
            
            # Get backend and JIT compile the method
            backend = get_backend("jax")
            jit_loss = jax.jit(backend.wasserstein_distance_loss_images, static_argnames=['projection_method', 'method', 'num_rotations'])

            for projection_method in ['real_rotate', 'fourier_slice']:
                for method in ['cdf', 'icdf', 'dsort']:
                    # Test JIT compiled version
                    loss_jit = jit_loss(x_jax, x_true_jax, projection_method=projection_method, method=method, num_rotations=10).sum()
                    
                    assert not np.isnan(float(loss_jit))
                    assert float(loss_jit) >= 0
            
        except ImportError:
            pytest.skip("JAX backend not available")
