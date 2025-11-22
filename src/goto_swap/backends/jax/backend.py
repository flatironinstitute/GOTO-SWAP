"""
JAX backend implementation for GOTO-SWAP.
"""

import jax
import jax.numpy as jnp
import dm_pix
from flax import nnx

from .. import BackendInterface
from .emd import emd_1d_sorted_jax_differentiable


class JAXBackend(BackendInterface):
    """JAX implementation of the GOTO-SWAP backend."""
    
    def __init__(self):
        self.backend_name = "jax"
    
    def get_backend_name(self) -> str:
        return self.backend_name
    
    def wasserstein_distance_loss_images(
        self, 
        x: jnp.ndarray, 
        x_true: jnp.ndarray, 
        num_rotations: int = 100
    ) -> jnp.ndarray:
        """
        Compute Wasserstein distance loss between images using sliced Wasserstein.
        
        This is the JAX implementation. For JIT compilation, use the backend directly:
        jit_fn = jax.jit(backend.wasserstein_distance_loss_images, static_argnames=['num_rotations'])
        
        Args:
            x: Predicted images (B, H, W) or (B, H, W, C)
            x_true: True images (B, H, W) or (B, H, W, C)
            num_rotations: Number of rotation angles to use
            
        Returns:
            Wasserstein distance loss
        """
        # Check array lenghts
        if x.ndim < 3 or x_true.ndim < 3:
            raise ValueError("The input parameters x and x_true must be 3D (B, H, W) or 4D arrays (B, H, W, C).")
        elif x.ndim > 4 or x_true.ndim > 4:
            raise ValueError("The input parameters x and x_true must be 3D (B, H, W) or 4D arrays (B, H, W, C).")
        x = x if x.ndim == 3 else x[..., 0]
        x_true = x_true if x_true.ndim == 3 else x_true[..., 0]
        
        # Soft circular mask
        Y, X = jnp.ogrid[:x.shape[1], :x.shape[2]]
        dist_from_center = jnp.sqrt((X - 0.5 * x.shape[1]) ** 2 + (Y - 0.5 * x.shape[1]) ** 2)
        mask = (dist_from_center <= 0.5 * x.shape[1]).astype(jnp.float32)[None, ..., None]
        mask = dm_pix.gaussian_blur(mask, 1.0, kernel_size=5)[..., 0]

        # Apply soft mask to images
        x = x * mask
        x_true = x_true * mask

        # Rotate images
        rotate_batch_over_rotations = jax.vmap(
            lambda img, rotations: dm_pix.rotate(img, rotations, mode="wrap"),
            in_axes=(None, 0)
        )
        rotate = jax.vmap(
            lambda imgs, rotations: rotate_batch_over_rotations(imgs, rotations), 
            in_axes=(0, None)
        )
        in_plane_rotations = jnp.linspace(0, jnp.pi, num_rotations, endpoint=False)
        rotated_x = rotate(x[..., None], in_plane_rotations)[..., 0]
        rotated_x_true = rotate(x_true[..., None], in_plane_rotations)[..., 0]

        # Split in positive and negative
        rotated_x_pos, rotated_x_neg = nnx.relu(rotated_x), nnx.relu(-rotated_x)
        rotated_x_true_pos, rotated_x_true_neg = nnx.relu(rotated_x_true), nnx.relu(-rotated_x_true)

        # 1D strips
        strips_x_pos, strips_x_neg = rotated_x_pos.sum(axis=-1), rotated_x_neg.sum(axis=-1)  # (B, N, n)
        strips_x_true_pos, strips_x_true_neg = rotated_x_true_pos.sum(axis=-1), rotated_x_true_neg.sum(axis=-1)  # (B, N, n)

        # Wasserstein distance (based on code: https://github.com/flatironinstitute/Cryo-EM-Heterogeneity-Challenge-1/blob/sliced_w/src/cryo_challenge/map_to_map/sliced_wasserstein/pixel_strip.py#L49)
        def wasserstein_over_one_strip(a, b):
            eps = 1e-8
            a = a / (a.sum(axis=-1, keepdims=True) + eps)  # (B, n)
            b = b / (b.sum(axis=-1, keepdims=True) + eps)  # (B, n)

            # Compute CDFs
            cdf_a = jnp.cumsum(a, axis=-1)  # (B, n)
            cdf_b = jnp.cumsum(b, axis=-1)  # (B, n)

            # Compute pairwise squared L2 distances between CDFs
            diff = cdf_a - cdf_b  # (B, n)
            return (diff ** 2).sum(axis=-1)  # (B)
        
        wasserstein_over_strips = jax.vmap(wasserstein_over_one_strip, in_axes=(1, 1))  # (B, N)

        w_pos = wasserstein_over_strips(strips_x_pos, strips_x_true_pos).mean(axis=0)
        w_neg = wasserstein_over_strips(strips_x_neg, strips_x_true_neg).mean(axis=0)
        w = (w_pos + w_neg)#.mean()

        return w
    
    def emd_1d_sorted_differentiable(self, u, v, u_weights, v_weights, metric='sqeuclidean', tol=1e-12):
        """
        Compute 1D Earth Mover's Distance using JAX.
        
        Args:
            u: JAX array of sorted source positions
            v: JAX array of sorted target positions  
            u_weights: JAX array of source weights
            v_weights: JAX array of target weights
            metric: Distance metric ('sqeuclidean', 'euclidean', or 'cityblock')
            tol: Tolerance for floating point comparisons
            
        Returns:
            JAX array containing the EMD cost
        """
        return emd_1d_sorted_jax_differentiable(u, v, u_weights, v_weights, metric, tol)

