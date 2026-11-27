"""
JAX backend implementation for GOTO-SWAP.
"""

from .backend import JAXBackend
from .emd import emd_1d_sorted_jax_differentiable
from .fourier_slice import fourier_slice_2d_to_1d

__all__ = [
    "JAXBackend",
    "emd_1d_sorted_jax_differentiable",
    "fourier_slice_2d_to_1d",
]

