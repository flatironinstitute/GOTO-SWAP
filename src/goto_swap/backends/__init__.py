"""
Backend abstraction layer for GOTO-SWAP.

This module provides a unified interface for different computational backends
(JAX, PyTorch, etc.) while allowing backend-specific implementations.
"""

from abc import ABC, abstractmethod
from typing import Union, Optional
import numpy as np

# Type hints for different backends
ArrayLike = Union[np.ndarray, 'jax.numpy.ndarray', 'torch.Tensor']


class BackendInterface(ABC):
    """Abstract base class for backend implementations."""
    
    @abstractmethod
    def wasserstein_distance_loss_images(
        self, 
        x: ArrayLike, 
        x_true: ArrayLike, 
        num_rotations: int = 100
    ) -> ArrayLike:
        """
        Compute Wasserstein distance loss between images using sliced Wasserstein.
        
        Args:
            x: Predicted images (B, H, W) or (B, H, W, C)
            x_true: True images (B, H, W) or (B, H, W, C)
            num_rotations: Number of rotation angles to use
            
        Returns:
            Wasserstein distance loss
        """
        pass
    
    @abstractmethod
    def get_backend_name(self) -> str:
        """Return the name of the backend."""
        pass


def get_backend(backend: str = "jax") -> BackendInterface:
    """
    Factory function to get the appropriate backend implementation.
    
    Args:
        backend: Backend name ("jax" or "pytorch")
        
    Returns:
        BackendInterface instance
        
    Raises:
        ImportError: If the requested backend is not available
        ValueError: If the backend name is not supported
    """
    if backend.lower() == "jax":
        try:
            from .jax import JAXBackend
            return JAXBackend()
        except ImportError as e:
            raise ImportError(
                "JAX backend requested but JAX is not installed. "
                "Install with: pip install jax dm-pix flax"
            ) from e
    
    elif backend.lower() == "pytorch":
        try:
            from .pytorch import PyTorchBackend
            return PyTorchBackend()
        except ImportError as e:
            raise ImportError(
                "PyTorch backend requested but PyTorch is not installed. "
                "Install with: pip install torch torchvision"
            ) from e
    
    else:
        raise ValueError(f"Unsupported backend: {backend}. Supported backends: jax, pytorch")


# Convenience function for direct access
def wasserstein_distance_loss_images(
    x: ArrayLike, 
    x_true: ArrayLike, 
    num_rotations: int = 100,
    backend: str = "jax"
) -> ArrayLike:
    """
    Compute Wasserstein distance loss between images.
    
    Args:
        x: Predicted images
        x_true: True images  
        num_rotations: Number of rotation angles
        backend: Backend to use ("jax" or "pytorch")
        
    Returns:
        Wasserstein distance loss
    """
    backend_impl = get_backend(backend)
    return backend_impl.wasserstein_distance_loss_images(x, x_true, num_rotations)

