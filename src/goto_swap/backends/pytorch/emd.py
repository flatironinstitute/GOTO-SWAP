"""
PyTorch implementation of Earth Mover's Distance (EMD) for GOTO-SWAP.
"""

import torch


def emd_1d_sorted_pytorch_differentiable(u, v, u_weights, v_weights, metric='sqeuclidean', tol=1e-12):
    """
    PyTorch implementation of differentiable 1D EMD cost with respect to u_weights (source).
    
    This is a PyTorch implementation equivalent to the JAX version, maintaining the same interface
    and behavior but using PyTorch operations for automatic differentiation.

    Note that this implementation assumes sorted inputs (u, v).
    Ported to PyTorch from the JAX version.

    Args:
        u: PyTorch tensor of sorted source positions
        v: PyTorch tensor of sorted target positions  
        u_weights: PyTorch tensor of source weights
        v_weights: PyTorch tensor of target weights
        metric: Distance metric ('sqeuclidean', 'euclidean', or 'cityblock')
        tol: Tolerance for floating point comparisons
        
    Returns:
        PyTorch tensor containing the EMD cost
        
    Notes:
        This function is differentiable with PyTorch autograd
    """
    n = u_weights.shape[0]
    m = v_weights.shape[0]
    
    def compute_metric(u_i, v_j, metric):
        if metric == 'sqeuclidean':
            return (u_i - v_j) ** 2
        elif metric in ('euclidean', 'cityblock'):
            return torch.abs(u_i - v_j)
        else:
            raise ValueError("Unsupported metric")
    
    i = 0
    j = 0
    w_i = u_weights[0]
    w_j = v_weights[0]
    cost = torch.tensor(0., dtype=u_weights.dtype, device=u_weights.device)
    
    while True:
        # Compute metric
        m_ij = compute_metric(u[i], v[j], metric)
        
        # Transport the smaller weight
        transported = torch.minimum(w_i, w_j)
        
        # Accumulate cost
        cost = cost + m_ij * transported
        
        # Update weights
        w_i = w_i - transported
        w_j = w_j - transported
        
        # Advance indices
        if w_i <= tol:
            i += 1
            if i == n:
                break
            w_i = u_weights[i]
        if w_j <= tol:
            j += 1
            if j == m:
                break
            w_j = v_weights[j]
    
    return cost
