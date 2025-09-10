"""
JAX implementation of Earth Mover's Distance (EMD) for GOTO-SWAP.
"""

import jax
import jax.numpy as jnp


def emd_1d_sorted_jax_differentiable(u, v, u_weights, v_weights, metric='sqeuclidean', tol=1e-12):
    """
    JAX implementation of differentiable 1D EMD cost with respect to u_weights (source).
    
    This is a JAX port of emd_1d_sorted_torch_differentiable, maintaining the same interface
    and behavior but using JAX operations for automatic differentiation and JIT compilation.

    Note that this implementation assumes sorted inputs (u, v).
    Ported to JAX from the PyTorch version.

    Args:
        u: JAX array of sorted source positions
        v: JAX array of sorted target positions  
        u_weights: JAX array of source weights
        v_weights: JAX array of target weights
        metric: Distance metric ('sqeuclidean', 'euclidean', or 'cityblock')
        tol: Tolerance for floating point comparisons
        
    Returns:
        JAX array containing the EMD cost
        
    Notes:
        This function is JAX-compilable and differentiable
    """
    n = u_weights.shape[0]
    m = v_weights.shape[0]
    
    def compute_metric(u_i, v_j, metric):
        if metric == 'sqeuclidean':
            return (u_i - v_j) ** 2
        elif metric in ('euclidean', 'cityblock'):
            return jnp.abs(u_i - v_j)
        else:
            raise ValueError("Unsupported metric")
    
    def emd_step(carry, _):
        i, j, w_i, w_j, cost = carry
        
        # Compute metric
        m_ij = compute_metric(u[i], v[j], metric)
        
        # Transport the smaller weight
        transported = jnp.minimum(w_i, w_j)
        
        # Accumulate cost
        cost = cost + m_ij * transported
        
        # Update weights
        w_i = w_i - transported
        w_j = w_j - transported
        
        # Advance indices
        i = jnp.where(w_i <= tol, i + 1, i)
        j = jnp.where(w_j <= tol, j + 1, j)
        
        # Update weights based on new indices
        w_i = jnp.where(w_i <= tol, 
                       jnp.where(i < n, u_weights[i], w_i), 
                       w_i)
        w_j = jnp.where(w_j <= tol,
                       jnp.where(j < m, v_weights[j], w_j),
                       w_j)
        
        return (i, j, w_i, w_j, cost), None
    
    # Initialize
    i = 0
    j = 0
    w_i = u_weights[0]
    w_j = v_weights[0]
    cost = jnp.array(0., dtype=u_weights.dtype)
    
    # Use scan to handle the while loop in a JAX-compatible way
    # We need to run for a maximum number of iterations to avoid infinite loops
    max_iterations = n + m  # Upper bound on total iterations needed
    
    carry_init = (i, j, w_i, w_j, cost)
    (final_i, final_j, final_w_i, final_w_j, final_cost), _ = jax.lax.scan(
        emd_step, carry_init, None, length=max_iterations
    )
    
    return final_cost
