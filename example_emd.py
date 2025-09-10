#!/usr/bin/env python3
"""
Example script demonstrating the EMD 1D function in GOTO-SWAP.
"""

import numpy as np

def example_emd_usage():
    """Demonstrate EMD 1D usage with different backends."""
    print("GOTO-SWAP EMD 1D Example")
    print("=" * 40)
    
    # Create test data
    u = np.array([1.0, 2.0, 3.0, 4.0])  # Source positions (sorted)
    v = np.array([1.5, 2.5, 3.5, 4.5])  # Target positions (sorted)
    u_weights = np.array([0.2, 0.3, 0.3, 0.2])  # Source weights
    v_weights = np.array([0.1, 0.4, 0.4, 0.1])  # Target weights
    
    print(f"Source positions: {u}")
    print(f"Target positions: {v}")
    print(f"Source weights: {u_weights}")
    print(f"Target weights: {v_weights}")
    print()
    
    # Test JAX backend
    try:
        from goto_swap import emd_1d_sorted_differentiable
        from goto_swap.backends.jax import emd_1d_sorted_jax_differentiable
        import jax.numpy as jnp
        
        print("Testing JAX Backend:")
        u_jax = jnp.array(u)
        v_jax = jnp.array(v)
        u_weights_jax = jnp.array(u_weights)
        v_weights_jax = jnp.array(v_weights)
        
        # Test different metrics
        for metric in ['sqeuclidean', 'euclidean', 'cityblock']:
            loss = emd_1d_sorted_differentiable(
                u_jax, v_jax, u_weights_jax, v_weights_jax, 
                metric=metric, backend="jax"
            )
            print(f"  {metric}: {float(loss):.6f}")
        
        # Test JIT compilation
        print("\nTesting JIT compilation:")
        import jax
        jit_emd = jax.jit(emd_1d_sorted_differentiable, static_argnames=['metric', 'backend'])
        loss_jit = jit_emd(u_jax, v_jax, u_weights_jax, v_weights_jax, metric='sqeuclidean', backend="jax")
        print(f"  JIT compiled result: {float(loss_jit):.6f}")
        
        # Test gradients
        print("\nTesting gradients:")
        grad_fn = jax.grad(emd_1d_sorted_differentiable, argnums=2)  # w.r.t. u_weights
        gradients = grad_fn(u_jax, v_jax, u_weights_jax, v_weights_jax, metric='sqeuclidean', backend="jax")
        print(f"  Gradients w.r.t. u_weights: {gradients}")
        
        # Test direct JAX function
        print("\nTesting direct JAX function:")
        loss_direct = emd_1d_sorted_jax_differentiable(u_jax, v_jax, u_weights_jax, v_weights_jax, metric='sqeuclidean')
        print(f"  Direct JAX result: {float(loss_direct):.6f}")
        
    except ImportError as e:
        print(f"JAX backend not available: {e}")
    
    print()
    
    # Test PyTorch backend
    try:
        from goto_swap.backends.pytorch import emd_1d_sorted_pytorch_differentiable
        import torch
        
        print("Testing PyTorch Backend:")
        u_torch = torch.from_numpy(u)
        v_torch = torch.from_numpy(v)
        u_weights_torch = torch.from_numpy(u_weights)
        v_weights_torch = torch.from_numpy(v_weights)
        
        # Test different metrics
        for metric in ['sqeuclidean', 'euclidean', 'cityblock']:
            loss = emd_1d_sorted_differentiable(
                u_torch, v_torch, u_weights_torch, v_weights_torch, 
                metric=metric, backend="pytorch"
            )
            print(f"  {metric}: {float(loss):.6f}")
        
        # Test gradients
        print("\nTesting gradients:")
        u_weights_torch.requires_grad_(True)
        loss = emd_1d_sorted_differentiable(
            u_torch, v_torch, u_weights_torch, v_weights_torch, 
            metric='sqeuclidean', backend="pytorch"
        )
        loss.backward()
        print(f"  Gradients w.r.t. u_weights: {u_weights_torch.grad}")
        
        # Test direct PyTorch function
        print("\nTesting direct PyTorch function:")
        u_weights_torch_direct = torch.from_numpy(u_weights)
        loss_direct = emd_1d_sorted_pytorch_differentiable(u_torch, v_torch, u_weights_torch_direct, v_weights_torch, metric='sqeuclidean')
        print(f"  Direct PyTorch result: {float(loss_direct):.6f}")
        
    except ImportError as e:
        print(f"PyTorch backend not available: {e}")
    
    print()
    
    # Test backend comparison
    try:
        import jax.numpy as jnp
        import torch
        from goto_swap import emd_1d_sorted_differentiable
        
        print("Backend Comparison:")
        u_jax = jnp.array(u)
        v_jax = jnp.array(v)
        u_weights_jax = jnp.array(u_weights)
        v_weights_jax = jnp.array(v_weights)
        
        u_torch = torch.from_numpy(u)
        v_torch = torch.from_numpy(v)
        u_weights_torch = torch.from_numpy(u_weights)
        v_weights_torch = torch.from_numpy(v_weights)
        
        loss_jax = emd_1d_sorted_differentiable(
            u_jax, v_jax, u_weights_jax, v_weights_jax, 
            metric='sqeuclidean', backend="jax"
        )
        loss_pytorch = emd_1d_sorted_differentiable(
            u_torch, v_torch, u_weights_torch, v_weights_torch, 
            metric='sqeuclidean', backend="pytorch"
        )
        
        print(f"  JAX result: {float(loss_jax):.6f}")
        print(f"  PyTorch result: {float(loss_pytorch):.6f}")
        print(f"  Difference: {abs(float(loss_jax) - float(loss_pytorch)):.2e}")
        
    except ImportError as e:
        print(f"Backend comparison not available: {e}")


if __name__ == "__main__":
    example_emd_usage()
