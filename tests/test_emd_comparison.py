"""
Comparison tests between GOTO-SWAP EMD implementation and reference implementations.
Based on test patterns from GOTO repository.
"""

import pytest
import numpy as np


@pytest.mark.backend
class TestEMDComparison:
    """Test EMD implementation against reference implementations."""
    
    def test_emd_against_ot_emd2_1d(self):
        """Test GOTO-SWAP EMD against ot.emd2_1d for uniform weights."""
        try:
            import ot
            import jax.numpy as jnp
            from goto_swap import emd_1d_sorted_differentiable
            
            n = 50
            u = np.sort(np.random.randn(n))
            v = np.sort(np.random.randn(n))
            u_weights = np.ones(n) / n
            v_weights = np.ones(n) / n
            
            # Reference implementation
            ot_result = ot.emd2_1d(u, v, u_weights, v_weights)
            
            # GOTO-SWAP implementation
            u_jax = jnp.array(u)
            v_jax = jnp.array(v)
            u_weights_jax = jnp.array(u_weights)
            v_weights_jax = jnp.array(v_weights)
            goto_swap_result = emd_1d_sorted_differentiable(u_jax, v_jax, u_weights_jax, v_weights_jax)
            
            # Results should be close
            diff = abs(float(ot_result) - float(goto_swap_result))
            assert diff < 1e-5, f"GOTO-SWAP differs from ot.emd2_1d by {diff}"
            
        except ImportError as e:
            pytest.skip(f"Required dependencies not available: {e}")
    
    def test_emd_against_ot_emd2_1d_non_uniform(self):
        """Test GOTO-SWAP EMD against ot.emd2_1d for non-uniform weights."""
        try:
            import ot
            import jax.numpy as jnp
            from goto_swap import emd_1d_sorted_differentiable
            
            n = 30
            u = np.sort(np.random.randn(n))
            v = np.sort(np.random.randn(n))
            u_weights = np.random.rand(n)
            u_weights = u_weights / u_weights.sum()
            v_weights = np.random.rand(n)
            v_weights = v_weights / v_weights.sum()
            
            # Reference implementation
            ot_result = ot.emd2_1d(u, v, u_weights, v_weights)
            
            # GOTO-SWAP implementation
            u_jax = jnp.array(u)
            v_jax = jnp.array(v)
            u_weights_jax = jnp.array(u_weights)
            v_weights_jax = jnp.array(v_weights)
            goto_swap_result = emd_1d_sorted_differentiable(u_jax, v_jax, u_weights_jax, v_weights_jax)
            
            # Results should be close
            diff = abs(float(ot_result) - float(goto_swap_result))
            assert diff < 1e-5, f"GOTO-SWAP differs from ot.emd2_1d by {diff}"
            
        except ImportError as e:
            pytest.skip(f"Required dependencies not available: {e}")
    
    def test_emd_consistency_across_implementations(self):
        """Test consistency across multiple EMD implementations."""
        try:
            import ot
            import jax.numpy as jnp
            from goto_swap import emd_1d_sorted_differentiable
            import torch
            
            n = 40
            u = np.sort(np.random.randn(n))
            v = np.sort(np.random.randn(n))
            u_weights = np.random.rand(n)
            u_weights = u_weights / u_weights.sum()
            v_weights = np.random.rand(n)
            v_weights = v_weights / v_weights.sum()
            
            # Multiple implementations
            ot_result = ot.emd2_1d(u, v, u_weights, v_weights)
            
            # GOTO-SWAP JAX
            u_jax = jnp.array(u)
            v_jax = jnp.array(v)
            u_weights_jax = jnp.array(u_weights)
            v_weights_jax = jnp.array(v_weights)
            goto_swap_jax = emd_1d_sorted_differentiable(u_jax, v_jax, u_weights_jax, v_weights_jax)
            
            # GOTO-SWAP PyTorch
            u_torch = torch.from_numpy(u)
            v_torch = torch.from_numpy(v)
            u_weights_torch = torch.from_numpy(u_weights)
            v_weights_torch = torch.from_numpy(v_weights)
            goto_swap_pytorch = emd_1d_sorted_differentiable(u_torch, v_torch, u_weights_torch, v_weights_torch, backend="pytorch")
            
            # All results should be close
            results = [float(ot_result), float(goto_swap_jax), float(goto_swap_pytorch)]
            for i, result in enumerate(results):
                for j, other_result in enumerate(results):
                    if i != j:
                        diff = abs(result - other_result)
                        assert diff < 1e-5, f"Implementation {i} differs from {j} by {diff}"
            
        except ImportError as e:
            pytest.skip(f"Required dependencies not available: {e}")
    
    def test_emd_quantitative_consistency(self):
        """Test quantitative consistency across multiple trials."""
        try:
            import ot
            import jax.numpy as jnp
            from goto_swap import emd_1d_sorted_differentiable
            from sklearn.linear_model import HuberRegressor
            
            n_trials = 50
            ot_results = []
            goto_swap_results = []
            
            for _ in range(n_trials):
                n = np.random.randint(20, 60)
                u = np.sort(np.random.randn(n))
                v = np.sort(np.random.randn(n))
                u_weights = np.random.rand(n)
                u_weights = u_weights / u_weights.sum()
                v_weights = np.random.rand(n)
                v_weights = v_weights / v_weights.sum()
                
                # Reference
                ot_result = ot.emd2_1d(u, v, u_weights, v_weights)
                ot_results.append(float(ot_result))
                
                # GOTO-SWAP
                u_jax = jnp.array(u)
                v_jax = jnp.array(v)
                u_weights_jax = jnp.array(u_weights)
                v_weights_jax = jnp.array(v_weights)
                goto_swap_result = emd_1d_sorted_differentiable(u_jax, v_jax, u_weights_jax, v_weights_jax)
                goto_swap_results.append(float(goto_swap_result))
            
            # Fit robust linear regression
            ot_results = np.array(ot_results)
            goto_swap_results = np.array(goto_swap_results)
            
            huber = HuberRegressor().fit(goto_swap_results.reshape(-1, 1), ot_results)
            slope = huber.coef_[0]
            intercept = huber.intercept_
            
            # Should have slope close to 1 and intercept close to 0
            assert abs(slope - 1.0) < 0.1, f"Slope should be close to 1, got {slope}"
            assert abs(intercept) < 0.1, f"Intercept should be close to 0, got {intercept}"
            
        except ImportError as e:
            pytest.skip(f"Required dependencies not available: {e}")
    
    def test_emd_different_metrics_consistency(self):
        """Test consistency of different metrics with reference implementation."""
        try:
            import ot
            import jax.numpy as jnp
            from goto_swap import emd_1d_sorted_differentiable
            
            n = 30
            u = np.sort(np.random.randn(n))
            v = np.sort(np.random.randn(n))
            u_weights = np.ones(n) / n
            v_weights = np.ones(n) / n
            
            # Test different metrics
            metrics = ['sqeuclidean', 'euclidean', 'cityblock']
            
            for metric in metrics:
                # Reference implementation
                ot_result = ot.emd2_1d(u, v, u_weights, v_weights, metric=metric)
                
                # GOTO-SWAP implementation
                u_jax = jnp.array(u)
                v_jax = jnp.array(v)
                u_weights_jax = jnp.array(u_weights)
                v_weights_jax = jnp.array(v_weights)
                goto_swap_result = emd_1d_sorted_differentiable(u_jax, v_jax, u_weights_jax, v_weights_jax, metric=metric)
                
                # Results should be close
                diff = abs(float(ot_result) - float(goto_swap_result))
                assert diff < 1e-5, f"GOTO-SWAP differs from ot.emd2_1d for metric {metric} by {diff}"
            
        except ImportError as e:
            pytest.skip(f"Required dependencies not available: {e}")
    
    def test_emd_edge_cases_consistency(self):
        """Test edge cases consistency with reference implementation."""
        try:
            import ot
            import jax.numpy as jnp
            from goto_swap import emd_1d_sorted_differentiable
            
            # Test with identical distributions
            u = np.array([1.0, 2.0, 3.0])
            v = np.array([1.0, 2.0, 3.0])
            u_weights = np.array([0.3, 0.4, 0.3])
            v_weights = np.array([0.3, 0.4, 0.3])
            
            # Both should give zero or near-zero result
            ot_result = ot.emd2_1d(u, v, u_weights, v_weights)
            u_jax = jnp.array(u)
            v_jax = jnp.array(v)
            u_weights_jax = jnp.array(u_weights)
            v_weights_jax = jnp.array(v_weights)
            goto_swap_result = emd_1d_sorted_differentiable(u_jax, v_jax, u_weights_jax, v_weights_jax)
            
            assert float(ot_result) < 1e-6, f"Identical distributions should give near-zero result, got {float(ot_result)}"
            assert float(goto_swap_result) < 1e-6, f"Identical distributions should give near-zero result, got {float(goto_swap_result)}"
            
        except ImportError as e:
            pytest.skip(f"Required dependencies not available: {e}")
