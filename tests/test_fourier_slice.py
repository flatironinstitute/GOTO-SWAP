import jax
import jax.numpy as jnp
import pytest
import matplotlib.pyplot as plt

from goto_swap import fourier_slice_2d_to_1d


@pytest.mark.jax
def test_fourier_slice_axis_projections_no_normalization():
    n_large = 128
    image = jnp.arange(n_large * n_large, dtype=jnp.float32).reshape(n_large, n_large)

    projection_x = fourier_slice_2d_to_1d(image, angle=0.0)
    projection_y = fourier_slice_2d_to_1d(image, angle=jnp.pi / 2.0)

    assert jnp.allclose(projection_x / projection_x.sum(), jnp.sum(image, axis=0) / jnp.sum(image), rtol=0.05, atol=1e-3)
    assert jnp.allclose(projection_y / projection_y.sum(), jnp.sum(image, axis=1) / jnp.sum(image), rtol=0.05, atol=1e-3)


@pytest.mark.jax
def test_fourier_slice_axis_projections():
    n_large = 128
    image = jnp.arange(n_large * n_large, dtype=jnp.float32).reshape(n_large, n_large)

    projection_x = fourier_slice_2d_to_1d(image, angle=0.0)
    projection_y = fourier_slice_2d_to_1d(image, angle=jnp.pi / 2.0)

    assert jnp.allclose(projection_x, jnp.sum(image, axis=0), rtol=0.05, atol=1e-3)
    assert jnp.allclose(projection_y , jnp.sum(image, axis=1), rtol=0.05, atol=1e-3)


@pytest.mark.jax
@pytest.mark.parametrize("angle", [0, jnp.pi / 4.0, jnp.pi / 3.0, jnp.pi, jnp.pi / 2.0])
def test_fourier_slice_gaussian_matches_analytical_projection(angle):
    size = 128
    coordinates = jnp.arange(size, dtype=jnp.float32) - size // 2
    x, y = jnp.meshgrid(coordinates, coordinates, indexing="xy")

    covariance_x = 30
    covariance_y = 10
    gaussian = jnp.exp(
        -0.5
        * (x**2 / covariance_x + y**2 / covariance_y)
    )
    fig, _ = plt.subplots()
    plt.imshow(gaussian, cmap="viridis")
    plt.colorbar()
    plt.savefig("fourier_slice_gaussian_input.png")
    plt.close(fig)

    projection = fourier_slice_2d_to_1d(gaussian, angle=angle)
    projected_coordinate = jnp.arange(size, dtype=jnp.float32) - size // 2
    projected_variance = (
        jnp.cos(angle) ** 2 * covariance_x
        + jnp.sin(angle) ** 2 * covariance_y
    )
    analytical = jnp.exp(
        -projected_coordinate**2 / (2.0 * projected_variance)
    )

    measured = jnp.real(projection)
    measured = jnp.abs(measured)
    measured = measured / jnp.sum(measured)
    analytical = analytical / jnp.sum(analytical)

    # fig, _ = plt.subplots()
    # plt.plot(measured, label="Measured")
    # plt.plot(analytical, label="Analytical")
    # plt.legend()
    # plt.title(f"Angle: {angle:.2f} rad")
    # plt.savefig(f"fourier_slice_gaussian_angle_{angle:.2f}.png")
    # plt.close(fig)

    assert jnp.allclose(measured, analytical, rtol=0.05, atol=1e-3)

def test_fourier_slice_gaussian_blobs_real_projection_0_90():
    size = 128
    coordinates = jnp.arange(size, dtype=jnp.float32) - size // 2
    x, y = jnp.meshgrid(coordinates, coordinates, indexing="xy")

    gaussian = 0
    for shift in [(-20, 0), (20, 0), (0, -20), (0, 20)]:
        x_shifted = x - shift[0]
        y_shifted = y - shift[1]

        covariance_x = 30
        covariance_y = 10
        gaussian += jnp.exp(
            -0.5
            * (x_shifted**2 / covariance_x + y_shifted**2 / covariance_y)
        )
        fig, _ = plt.subplots()
        plt.imshow(gaussian, cmap="viridis")
        plt.colorbar()
        plt.savefig(f"fourier_slice_gaussian_input_shift_{shift}.png")
        plt.close(fig)

    projection_0 = fourier_slice_2d_to_1d(gaussian, angle=0.0)
    projection_90 = fourier_slice_2d_to_1d(gaussian, angle=jnp.pi / 2.0)

    real_spaced_projection_0 = jnp.sum(gaussian, axis=0)
    real_spaced_projection_90 = jnp.sum(gaussian, axis=1)

    assert jnp.allclose(projection_0, real_spaced_projection_0, rtol=0.05, atol=1e-3)
    assert jnp.allclose(projection_90, real_spaced_projection_90, rtol=0.05, atol=1e-3)


@pytest.mark.jax
def test_fourier_slice_is_jittable_and_differentiable():
    image = jnp.ones((16, 16), dtype=jnp.float32)
    angle = jnp.asarray(jnp.pi / 5.0)

    jitted = jax.jit(fourier_slice_2d_to_1d)
    projected = jitted(image, angle)
    gradient = jax.grad(lambda value: jnp.sum(jitted(value, angle)))(image)

    assert projected.shape == (16,)
    assert gradient.shape == image.shape
