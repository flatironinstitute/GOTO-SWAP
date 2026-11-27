"""
Legacy module for backward compatibility.

This module is deprecated. Please use the new backend system:
    from goto_swap import wasserstein_distance_loss_images, get_backend
    
    # Use default JAX backend
    loss = wasserstein_distance_loss_images(x, x_true)
    
    # Use specific backend
    loss = wasserstein_distance_loss_images(x, x_true, backend="pytorch")
    
    # Or get backend instance directly
    backend = get_backend("jax")
    loss = backend.wasserstein_distance_loss_images(x, x_true)
"""

import warnings
from .backends import wasserstein_distance_loss_images as _new_wasserstein_distance_loss_images

# Issue deprecation warning
warnings.warn(
    "goto_swap.base is deprecated. Use 'from goto_swap import wasserstein_distance_loss_images' instead. "
    "The new system supports multiple backends (JAX, PyTorch) with optional dependencies.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export the function for backward compatibility
def wasserstein_distance_loss_images(x, x_true, num_rotations=100):
    """
    Legacy function for backward compatibility.
    
    This function now uses the JAX backend by default.
    For PyTorch support, use the new backend system:
        from goto_swap import wasserstein_distance_loss_images
        loss = wasserstein_distance_loss_images(x, x_true, backend="pytorch")
    """
    return _new_wasserstein_distance_loss_images(x, x_true, method='dsort', projection_method='real_rotate', num_rotations=num_rotations, backend="jax").sum()