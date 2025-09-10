"""
PyTorch backend implementation for GOTO-SWAP.
"""

from .backend import PyTorchBackend
from .emd import emd_1d_sorted_pytorch_differentiable

__all__ = ["PyTorchBackend", "emd_1d_sorted_pytorch_differentiable"]
