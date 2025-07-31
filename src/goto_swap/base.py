import jax
import jax.numpy as jnp
from functools import partial
import dm_pix
from flax import nnx

@partial(jax.jit, static_argnames=['num_rotations',])
def wasserstein_distance_loss_images(x: jnp.ndarray, x_true: jnp.ndarray, num_rotations: int = 100):
    # Soft circular mask
    Y, X = jnp.ogrid[:x.shape[1], :x.shape[2]]
    dist_from_center = jnp.sqrt((X - 0.5 * x.shape[0]) ** 2 + (Y - 0.5 * x.shape[1]) ** 2)
    mask = (dist_from_center <= 0.5 * x.shape[0]).astype(jnp.float32)[None, ..., None]
    mask = dm_pix.gaussian_blur(mask, 1.0, kernel_size=5)[..., 0]

    # Apply soft mask to images
    x = x * mask
    x_true = x_true * mask

    # Rotate images
    rotate_batch_over_rotations = jax.vmap(lambda img, rotations: dm_pix.rotate(img, rotations, mode="wrap"),
                                           in_axes=(None, 0))
    rotate = jax.vmap(lambda imgs, rotations: rotate_batch_over_rotations(imgs, rotations), in_axes=(0, None))
    in_plane_rotations = jnp.linspace(0, jnp.pi, num_rotations, endpoint=False)
    rotated_x = rotate(x[..., None], in_plane_rotations)[..., 0]
    rotated_x_true = rotate(x_true[..., None], in_plane_rotations)[..., 0]

    # Split in positive and negative
    rotated_x_pos, rotated_x_neg = nnx.relu(rotated_x), -nnx.relu(-rotated_x)
    rotated_x_true_pos, rotated_x_true_neg = nnx.relu(rotated_x_true), -nnx.relu(-rotated_x_true)

    # 1D strips
    strips_x_pos, strips_x_neg = rotated_x_pos.sum(axis=-1), rotated_x_neg.sum(axis=-1)  # (B, N, n)
    strips_x_true_pos, strips_x_true_neg = rotated_x_true_pos.sum(axis=-1), rotated_x_true_neg.sum(axis=-1)  # (B, N, n)

    # Wassestein distance (based on code: https://github.com/flatironinstitute/Cryo-EM-Heterogeneity-Challenge-1/blob/sliced_w/src/cryo_challenge/map_to_map/sliced_wasserstein/pixel_strip.py#L49)
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
    w = (w_pos + w_neg).mean()

    return w