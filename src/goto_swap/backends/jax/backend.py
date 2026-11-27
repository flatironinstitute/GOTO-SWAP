"""
JAX backend implementation for GOTO-SWAP.
"""

from functools import partial
import jax
import jax.numpy as jnp
import dm_pix
from flax import nnx

from goto_swap.backends.jax.fourier_slice import fourier_slice_2d_to_1d, fourier_slice_2d_to_1d_precompute, fourier_slice_2d_to_1d_postcompute

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
        projection_method: str,
        method: str,
        num_rotations: int,
        n_interpolation_points: int = 1000,
    ) -> jnp.ndarray:
        """
        Compute Wasserstein distance loss between images using sliced Wasserstein.
        
        This is the JAX implementation. For JIT compilation, use the backend directly:
        jit_fn = jax.jit(backend.wasserstein_distance_loss_images, static_argnames=['num_rotations'])
        
        Args:
            x: Predicted images (B, H, W) 
            x_true: True images (B, H, W) 
            num_rotations: Number of rotation angles to use
            projection_method: Method for projecting 2D images to 1D (e.g., 'real' or 'fourier_slice')
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
        if projection_method == 'real_rotate':
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
            strips_x_pos, strips_x_neg = rotated_x_pos.sum(axis=-1), rotated_x_neg.sum(axis=-1)  # (B, n_proj, n)
            strips_x_true_pos, strips_x_true_neg = rotated_x_true_pos.sum(axis=-1), rotated_x_true_neg.sum(axis=-1)  # (B, n_proj, n)

        elif projection_method == 'fourier_slice':

            # Precompute once per image
            fourier_x = jax.vmap(
                lambda img: fourier_slice_2d_to_1d_precompute(
                    img, output_size=None
                )
            )(x)

            fourier_x_true = jax.vmap(
                lambda img: fourier_slice_2d_to_1d_precompute(
                    img, output_size=None
                )
            )(x_true)

            # Map over angles, reusing the precomputed Fourier transforms
            postcompute_batch = jax.vmap(
                lambda fourier, angle: fourier_slice_2d_to_1d_postcompute(
                    fourier, angle=angle, output_size=None
                ),
                in_axes=(None, 0),
            )

            # Map over images
            postcompute = jax.vmap(
                lambda fourier, angles: postcompute_batch(fourier, angles),
                in_axes=(0, None),
            )

            in_plane_rotations = jnp.linspace(
                0, jnp.pi, num_rotations, endpoint=False
            )

            sliced_x = postcompute(fourier_x, in_plane_rotations).real
            sliced_x_true = postcompute(fourier_x_true, in_plane_rotations).real

            # split in positive and negative
            strips_x_pos, strips_x_neg = nnx.relu(sliced_x), nnx.relu(-sliced_x)
            strips_x_true_pos, strips_x_true_neg = nnx.relu(sliced_x_true), nnx.relu(-sliced_x_true)
        else:
            raise ValueError(f"Unknown projection method '{projection_method}'. Supported methods are 'real_rotate', 'fourier_slice'.")



        
        if method == 'icdf':
            wasserstein_over_strips_single_image_pair = jax.vmap(partial(self.wasserstein_1d_inverse_cdf, x_a=jnp.arange(x.shape[-1]), n_interpolation_points=n_interpolation_points, p=2), in_axes=(0, 0))
            wasserstein_over_strips = jax.vmap(wasserstein_over_strips_single_image_pair, in_axes=(0, 0))  # (n_proj, B)
            axis=0

        elif method == 'cdf':
            # Wasserstein distance (based on code: https://github.com/flatironinstitute/Cryo-EM-Heterogeneity-Challenge-1/blob/sliced_w/src/cryo_challenge/map_to_map/sliced_wasserstein/pixel_strip.py#L49)
            wasserstein_over_strips = jax.vmap(self.cdf_over_one_strip, in_axes=(1, 1))  # (n_proj, B)
            axis=1

        elif method == 'dsort':
            space_1d_x = jnp.arange(x.shape[-1]).astype(x.dtype)
            space_1d_x_true = jnp.arange(x_true.shape[-1]).astype(x_true.dtype)
            def emd_1d_sorted_jax_differentiable_partial(u_weights, v_weights):
                eps = jnp.finfo(u_weights.dtype).eps
                u_weights = u_weights / (jnp.sum(u_weights) + eps)
                v_weights = v_weights / (jnp.sum(v_weights) + eps)
                return partial(self.emd_1d_sorted_differentiable)(
                    u=space_1d_x,
                    v=space_1d_x_true,
                    u_weights=u_weights, 
                    v_weights=v_weights,
                    metric='sqeuclidean', 
                    tol=1e-12)
            
            wasserstein_over_strips_single_image_pair = jax.vmap(emd_1d_sorted_jax_differentiable_partial, in_axes=(0,0))
            wasserstein_over_strips = jax.vmap(wasserstein_over_strips_single_image_pair, in_axes=(0, 0))
            axis=0
    
        else:
            raise ValueError(f"Unknown method '{method}'. Supported methods are 'cdf', 'icdf', 'dsort'.")

        # print('shapes', space_1d.shape, strips_x_pos.shape, strips_x_true_pos.shape)
        # print('dtypes', space_1d.dtype, strips_x_pos.dtype, strips_x_true_pos.dtype)
        w_pos = wasserstein_over_strips(strips_x_pos, strips_x_true_pos)
        w_neg = wasserstein_over_strips(strips_x_neg, strips_x_true_neg)
        w = (w_pos + w_neg).mean(axis)

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


    def cdf_over_one_strip(self, a, b):
        eps = jnp.finfo(a.dtype).eps
        a = a / (a.sum(axis=-1, keepdims=True) + eps)  # (B, n)
        b = b / (b.sum(axis=-1, keepdims=True) + eps)  # (B, n)

        # Compute CDFs
        cdf_a = jnp.cumsum(a, axis=-1)  # (B, n)
        cdf_b = jnp.cumsum(b, axis=-1)  # (B, n)

        # Compute pairwise squared L2 distances between CDFs
        diff = cdf_a - cdf_b  # (B, n)
        return (diff ** 2).sum(axis=-1)  # (B)


    def jax_inverse_cdf_from_pdf(self, x_a, a, p=None):
        """
        x_a: (n,)
        a:   (n,) PDF values over x_a
        p:   (m,) quantile levels in [0,1]
        """
        eps = jnp.finfo(a.dtype).eps
        a = a + eps
        # Default p
        if p is None:
            p = jnp.linspace(0.0, 1.0, 2000)

        dx = x_a[1] - x_a[0]

        # Compute CDF
        cdf = jnp.cumsum(a) * dx
        cdf = cdf / cdf[-1]
        # ensure last = 1
        cdf = cdf.at[-1].set(1.0)

        # Searchsorted (JAX has it)
        idx = jnp.searchsorted(cdf, p, side="left")

        # clamp between [1, len(x_a)-1]
        idx = jnp.clip(idx, 1, x_a.shape[0] - 1)

        # Values at bracketing indices
        cdf_lo = cdf[idx - 1]
        cdf_hi = cdf[idx]

        x_lo = x_a[idx - 1]
        x_hi = x_a[idx]

        # avoid divide-by-zero
        
        denom = jnp.maximum(cdf_hi - cdf_lo, eps)

        # Linear interpolation
        t = (p - cdf_lo) / denom
        Q = x_lo + t * (x_hi - x_lo)

        return Q


    def jax_trapz(self, y, x):
        """
        Simple trapezoidal rule ∫ y dx on a uniform or non-uniform grid.
        y: (..., n)
        x: (n,)
        Returns: (...,)
        """
        dx = x[1:] - x[:-1]      # (n-1,)
        avg = 0.5 * (y[..., 1:] + y[..., :-1])  # (..., n-1)
        return jnp.sum(avg * dx, axis=-1)


    def wasserstein_1d_inverse_cdf(self, a, b, x_a, n_interpolation_points, p: int):
        """
        Compute 1D Wasserstein via inverse CDF sampling.
        a, b: PDF arrays on common grid x_a
        p: 1 or 2  => W₁ or W₂
        """
        interpolation_points = jnp.linspace(0.0, 1.0, n_interpolation_points)

        Qf = self.jax_inverse_cdf_from_pdf(x_a, a, interpolation_points)
        Qg = self.jax_inverse_cdf_from_pdf(x_a, b, interpolation_points)
        diff = Qf - Qg

        if p == 1:
            return self.jax_trapz(jnp.abs(diff), interpolation_points)
        elif p == 2:
            return self.jax_trapz(diff**2, interpolation_points)
        else:
            raise ValueError("Only p=1 and p=2 are supported")