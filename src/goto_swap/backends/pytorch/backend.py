"""
PyTorch backend implementation for GOTO-SWAP.
"""

import torch
import torch.nn.functional as F
import math

from .. import BackendInterface
from .emd import emd_1d_sorted_pytorch_differentiable


class PyTorchBackend(BackendInterface):
    """PyTorch implementation of the GOTO-SWAP backend."""
    
    def __init__(self):
        self.backend_name = "pytorch"
    
    def get_backend_name(self) -> str:
        return self.backend_name
    
    def wasserstein_distance_loss_images(
        self, 
        x: torch.Tensor, 
        x_true: torch.Tensor, 
        projection_method: str,
        method: str,
        num_rotations: int = 100
    ) -> torch.Tensor:
        """
        Compute Wasserstein distance loss between images using sliced Wasserstein.
        
        This is the PyTorch implementation with equivalent functionality to the JAX version.
        
        Args:
            x: Predicted images (B, H, W) or (B, H, W, C)
            x_true: True images (B, H, W) or (B, H, W, C)
            num_rotations: Number of rotation angles to use
            
        Returns:
            Wasserstein distance loss
        """
        device = x.device
        dtype = x.dtype
        
        # Soft circular mask
        Y, X = torch.meshgrid(
            torch.arange(x.shape[1], device=device, dtype=dtype),
            torch.arange(x.shape[2], device=device, dtype=dtype),
            indexing='ij'
        )
        dist_from_center = torch.sqrt((X - 0.5 * x.shape[0]) ** 2 + (Y - 0.5 * x.shape[1]) ** 2)
        mask = (dist_from_center <= 0.5 * x.shape[0]).float().unsqueeze(0).unsqueeze(-1)
        
        # Apply Gaussian blur to mask (PyTorch equivalent of dm_pix.gaussian_blur)
        mask = self._gaussian_blur(mask, sigma=1.0, kernel_size=5)[..., 0]

        # Apply soft mask to images
        x = x * mask
        x_true = x_true * mask

        # Generate rotation angles
        in_plane_rotations = torch.linspace(0, math.pi, num_rotations + 1, device=device, dtype=dtype)[:-1]  # Exclude the last angle to avoid duplication
        
        # Rotate images
        rotated_x = self._rotate_batch(x, in_plane_rotations)
        rotated_x_true = self._rotate_batch(x_true, in_plane_rotations)

        # Split in positive and negative
        rotated_x_pos = F.relu(rotated_x)
        rotated_x_neg = -F.relu(-rotated_x)
        rotated_x_true_pos = F.relu(rotated_x_true)
        rotated_x_true_neg = -F.relu(-rotated_x_true)

        # 1D strips
        strips_x_pos = rotated_x_pos.sum(dim=-1)  # (B, N, n)
        strips_x_neg = rotated_x_neg.sum(dim=-1)  # (B, N, n)
        strips_x_true_pos = rotated_x_true_pos.sum(dim=-1)  # (B, N, n)
        strips_x_true_neg = rotated_x_true_neg.sum(dim=-1)  # (B, N, n)

        # Wasserstein distance computation
        w_pos = self._wasserstein_over_strips(strips_x_pos, strips_x_true_pos).mean(dim=0)
        w_neg = self._wasserstein_over_strips(strips_x_neg, strips_x_true_neg).mean(dim=0)
        w = (w_pos + w_neg).mean()

        return w
    
    def _gaussian_blur(self, x: torch.Tensor, sigma: float, kernel_size: int) -> torch.Tensor:
        """Apply Gaussian blur to tensor (PyTorch equivalent of dm_pix.gaussian_blur)."""
        # Create Gaussian kernel
        kernel = self._create_gaussian_kernel(sigma, kernel_size, x.dtype, x.device)
        
        # Apply convolution for each channel
        channels = x.shape[-1]
        blurred = []
        for c in range(channels):
            channel_data = x[..., c].unsqueeze(1)  # Add channel dimension
            blurred_channel = F.conv2d(
                channel_data, 
                kernel.unsqueeze(0).unsqueeze(0), 
                padding=kernel_size//2
            )
            blurred.append(blurred_channel.squeeze(1))
        
        return torch.stack(blurred, dim=-1)
    
    def _create_gaussian_kernel(self, sigma: float, kernel_size: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        """Create a 2D Gaussian kernel."""
        kernel_1d = torch.arange(kernel_size, dtype=dtype, device=device) - kernel_size // 2
        kernel_1d = torch.exp(-(kernel_1d ** 2) / (2 * sigma ** 2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        
        kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
        return kernel_2d
    
    def _rotate_batch(self, x: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
        """Rotate batch of images by given angles."""
        batch_size = x.shape[0]
        num_angles = len(angles)
        height, width = x.shape[1], x.shape[2]
        
        # Expand x to include rotation dimension
        x_expanded = x.unsqueeze(1).expand(batch_size, num_angles, height, width)
        x_expanded = x_expanded.reshape(-1, 1, height, width)  # (B*N, 1, H, W)
        
        # Create rotation matrices
        cos_vals = torch.cos(angles)
        sin_vals = torch.sin(angles)
        
        # Create affine transformation matrices
        theta = torch.zeros(num_angles, 2, 3, dtype=x.dtype, device=x.device)
        theta[:, 0, 0] = cos_vals
        theta[:, 0, 1] = -sin_vals
        theta[:, 1, 0] = sin_vals
        theta[:, 1, 1] = cos_vals
        
        # Expand theta for batch
        theta_expanded = theta.unsqueeze(0).expand(batch_size, -1, -1, -1)
        theta_expanded = theta_expanded.reshape(-1, 2, 3)  # (B*N, 2, 3)
        
        # Apply rotation
        grid = F.affine_grid(theta_expanded, x_expanded.size(), align_corners=False)
        rotated = F.grid_sample(x_expanded, grid, mode='nearest', padding_mode='border', align_corners=False)
        
        # Reshape back to (B, N, H, W)
        rotated = rotated.reshape(batch_size, num_angles, height, width)
        
        return rotated
    
    def _wasserstein_over_strips(self, strips_x: torch.Tensor, strips_x_true: torch.Tensor) -> torch.Tensor:
        """Compute Wasserstein distance over strips."""
        # eps from floating point of strips_x
        eps = torch.finfo(strips_x.dtype).eps
        strips_x = strips_x / (strips_x.sum(dim=-1, keepdim=True) + eps)  # (B, N, n)
        strips_x_true = strips_x_true / (strips_x_true.sum(dim=-1, keepdim=True) + eps)  # (B, N, n)

        # Compute CDFs
        cdf_x = torch.cumsum(strips_x, dim=-1)  # (B, N, n)
        cdf_x_true = torch.cumsum(strips_x_true, dim=-1)  # (B, N, n)

        # Compute pairwise squared L2 distances between CDFs
        diff = cdf_x - cdf_x_true  # (B, N, n)
        return (diff ** 2).sum(dim=-1)  # (B, N)
    
    def emd_1d_sorted_differentiable(self, u, v, u_weights, v_weights, metric='sqeuclidean', tol=1e-12):
        """
        Compute 1D Earth Mover's Distance using PyTorch.
        
        Args:
            u: PyTorch tensor of sorted source positions
            v: PyTorch tensor of sorted target positions  
            u_weights: PyTorch tensor of source weights
            v_weights: PyTorch tensor of target weights
            metric: Distance metric ('sqeuclidean', 'euclidean', or 'cityblock')
            tol: Tolerance for floating point comparisons
            
        Returns:
            PyTorch tensor containing the EMD cost
        """
        return emd_1d_sorted_pytorch_differentiable(u, v, u_weights, v_weights, metric, tol)

