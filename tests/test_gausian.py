import numpy as np
import pytest


def _gaussian_blob(grid, center, covariance):
    displacement = grid - np.asarray(center)
    inverse_covariance = np.diag(1.0 / np.asarray(covariance))
    exponent = np.einsum(
        "...i,ij,...j->...",
        displacement,
        inverse_covariance,
        displacement,
    )
    return np.exp(-0.5 * exponent)


@pytest.mark.jax
def test_jax_image_wasserstein_matches_pot_dense_grid():
    ot = pytest.importorskip("ot")
    jnp = pytest.importorskip("jax.numpy")
    try:
        from goto_swap import get_backend
        backend = get_backend("jax")
    except ImportError as error:
        pytest.skip(f"JAX backend dependencies are required: {error}")

    size = 32
    rows, columns = np.mgrid[:size, :size]
    grid = np.stack((rows, columns), axis=-1).astype(np.float64)
    tx_true = 14
    source = _gaussian_blob(grid, center=(tx_true, 16), covariance=(1, 2))

    delta = 5
    for tx in np.linspace(tx_true - delta, tx_true + delta, 10):
        target = _gaussian_blob(grid, center=(tx, 16), covariance=(3, 2))

        dim = 2

        source_weights = source.ravel()
        source_weights /= source_weights.sum()
        target_weights = target.ravel()
        target_weights /= target_weights.sum()
        points = grid.reshape(-1, 2)
        cost_matrix = ot.dist(points, points, metric="sqeuclidean")
        pot_loss = ot.emd2(source_weights, target_weights, cost_matrix)

        atol = rtol = 2e-2

        jax_loss = backend.wasserstein_distance_loss_images(
            jnp.asarray(source[None, ...]),
            jnp.asarray(target[None, ...]),
            projection_method="real_rotate",
            method="dsort",
            num_rotations=1,
        ).sum()
        np.testing.assert_allclose(float(jax_loss), pot_loss, rtol=rtol, atol=atol)

        jax_loss_sliced_real = backend.wasserstein_distance_loss_images(
            jnp.asarray(source[None, ...]),
            jnp.asarray(target[None, ...]),
            projection_method="real_rotate",
            method="dsort",
            num_rotations=100,
        ).mean() * dim

        np.testing.assert_allclose(float(jax_loss_sliced_real), pot_loss, rtol=rtol, atol=atol)

        # this fails
        # jax_loss_sliced_fourier = backend.wasserstein_distance_loss_images(
        #     jnp.asarray(source[None, ...]),
        #     jnp.asarray(target[None, ...]),
        #     projection_method="fourier_slice",
        #     method="dsort",
        #     num_rotations=100,
        # ).mean() * dim

        # np.testing.assert_allclose(float(jax_loss_sliced_fourier), pot_loss, rtol=rtol, atol=atol)

      