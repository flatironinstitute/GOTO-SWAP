"""
JAX backend implementation for GOTO-SWAP.
"""

from .backend import JAXBackend
from .emd import emd_1d_sorted_jax_differentiable

__all__ = ["JAXBackend", "emd_1d_sorted_jax_differentiable"]

