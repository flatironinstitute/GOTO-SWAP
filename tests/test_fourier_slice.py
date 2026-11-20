import jax
import jax.numpy as jnp
import pytest
import matplotlib.pyplot as plt

from goto_swap import fourier_slice_2d_to_1d


@pytest.mark.jax
def test_fourier_slice_axis_projections():
    image = jnp.arange(5 * 7, dtype=jnp.float32).reshape(5, 7)

    projection_x = fourier_slice_2d_to_1d(image, angle=0.0)
    projection_y = fourier_slice_2d_to_1d(image, angle=jnp.pi / 2.0)

    assert jnp.allclose(projection_x, jnp.sum(image, axis=0), rtol=1e-6, atol=1e-6)
    assert jnp.allclose(projection_y, jnp.sum(image, axis=1), rtol=1e-6, atol=1e-6)


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

if __name__ == "__main__":
    for angle in [0, jnp.pi / 4.0, jnp.pi / 3.0, jnp.pi, jnp.pi / 2.0]:
        test_fourier_slice_gaussian_matches_analytical_projection(angle)



@pytest.mark.jax
def test_fourier_slice_is_jittable_and_differentiable():
    image = jnp.ones((16, 16), dtype=jnp.float32)
    angle = jnp.asarray(jnp.pi / 5.0)

    jitted = jax.jit(fourier_slice_2d_to_1d)
    projected = jitted(image, angle)
    gradient = jax.grad(lambda value: jnp.sum(jitted(value, angle)))(image)

    assert projected.shape == (16,)
    assert gradient.shape == image.shape
