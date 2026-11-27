"""Fourier-slice projections between 2D images and 1D profiles."""

import jax
import jax.numpy as jnp

def _bilinear_sample(image: jnp.ndarray, y: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    """Sample a 2D array at floating-point row and column coordinates."""
    height, width = image.shape

    y0 = jnp.floor(y).astype(jnp.int32)
    x0 = jnp.floor(x).astype(jnp.int32)
    y1 = y0 + 1
    x1 = x0 + 1

    inside = (
    (y >= 0) & (y <= height - 1)
    & (x >= 0) & (x <= width - 1)
)
    safe_y0 = jnp.clip(y0, 0, height - 1)
    safe_y1 = jnp.clip(y1, 0, height - 1)
    safe_x0 = jnp.clip(x0, 0, width - 1)
    safe_x1 = jnp.clip(x1, 0, width - 1)

    wy = y - y0.astype(y.dtype)
    wx = x - x0.astype(x.dtype)

    value = (
        (1.0 - wy) * (1.0 - wx) * image[safe_y0, safe_x0]
        + (1.0 - wy) * wx * image[safe_y0, safe_x1]
        + wy * (1.0 - wx) * image[safe_y1, safe_x0]
        + wy * wx * image[safe_y1, safe_x1]
    )
    return jnp.where(inside, value, jnp.array(0.0, dtype=value.dtype))

def fourier_slice_2d_to_1d_precompute(
    image: jnp.ndarray,
    output_size: int | None = None,
) -> jnp.ndarray:
    """Project a 2D image to 1D using the Fourier slice theorem.

    ``angle`` is measured in radians counter-clockwise from the positive x axis.
    Angle 0 sums over rows (``image.sum(axis=0)``), while pi/2 sums over columns
    (``image.sum(axis=1)``). Arbitrary angles sample a line through the origin
    in the centered 2D Fourier transform and inverse-transform that line.

    Args:
        image: A 2D JAX array.
        angle: Projection angle in radians.
        output_size: Number of samples in the returned profile. Defaults to
            the image's larger spatial dimension.

    Returns:
        A real-valued 1D JAX array.
    """
    if image.ndim != 2:
        raise ValueError("image must be a 2D array")
    if image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError("image dimensions must be non-zero")

    if output_size is not None:
        if not isinstance(output_size, int) or output_size <= 0:
            raise ValueError("output_size must be a positive integer")


    # The Fourier transform is centered so the zero frequency is in the middle.
    image_unshifted = jnp.fft.ifftshift(image)
    fourier = jnp.fft.fftshift(jnp.fft.fft2(image_unshifted))

    return fourier

def fourier_slice_2d_to_1d_postcompute(
    fourier: jnp.ndarray,
    angle: float = 0.0,
    output_size: int | None = None,
) -> jnp.ndarray:
    """Project a 2D image to 1D using the Fourier slice theorem.

    ``angle`` is measured in radians counter-clockwise from the positive x axis.
    Angle 0 sums over rows (``image.sum(axis=0)``), while pi/2 sums over columns
    (``image.sum(axis=1)``). Arbitrary angles sample a line through the origin
    in the centered 2D Fourier transform and inverse-transform that line.

    Args:
        image: A 2D JAX array.
        angle: Projection angle in radians.
        output_size: Number of samples in the returned profile. Defaults to
            the image's larger spatial dimension.

    Returns:
        A real-valued 1D JAX array.
    """
    if fourier.ndim != 2:
        raise ValueError("fourier must be a 2D array")
    if fourier.shape[0] == 0 or fourier.shape[1] == 0:
        raise ValueError("fourier dimensions must be non-zero")

    if output_size is not None:
        if not isinstance(output_size, int) or output_size <= 0:
            raise ValueError("output_size must be a positive integer")

    height, width = fourier.shape
    n = output_size if output_size is not None else max(fourier.shape)
    frequency_offset = jnp.arange(n, dtype=jnp.float32) - n // 2


    fx = jnp.cos(angle) * frequency_offset / n
    fy = jnp.sin(angle) * frequency_offset / n

    x_coordinate = fx * width + width // 2
    y_coordinate = fy * height + height // 2

    sliced = _bilinear_sample(
        fourier,
        y_coordinate,
        x_coordinate,
    )
    profile = jnp.fft.fftshift(jnp.fft.ifft(jnp.fft.ifftshift(sliced)))

    return profile


def fourier_slice_2d_to_1d(
    image: jnp.ndarray,
    angle: float = 0.0,
    output_size: int | None = None,
) -> jnp.ndarray:
    """Project a 2D image to 1D using the Fourier slice theorem.

    ``angle`` is measured in radians counter-clockwise from the positive x axis.
    Angle 0 sums over rows (``image.sum(axis=0)``), while pi/2 sums over columns
    (``image.sum(axis=1)``). Arbitrary angles sample a line through the origin
    in the centered 2D Fourier transform and inverse-transform that line.

    Args:
        image: A 2D JAX array.
        angle: Projection angle in radians.
        output_size: Number of samples in the returned profile. Defaults to
            the image's larger spatial dimension.

    Returns:
        A real-valued 1D JAX array.
    """
    angle_array = jnp.asarray(angle, dtype=jnp.float32)
    is_static = not isinstance(angle_array, jax.core.Tracer)

    periodic_angle = jnp.mod(angle_array, jnp.pi)
    near_zero = jnp.isclose(periodic_angle, 0.0, rtol=0.0, atol=1e-6)
    near_90 = jnp.isclose(periodic_angle, jnp.pi / 2.0, rtol=0.0, atol=1e-6)

    if is_static:
        if near_zero:
            return jnp.sum(image, axis=0)
        if near_90:
            return jnp.sum(image, axis=1)

    fourier = fourier_slice_2d_to_1d_precompute(image, output_size=output_size)
    profile = fourier_slice_2d_to_1d_postcompute(fourier, angle=angle, output_size=output_size)
    return jnp.real(profile).astype(image.real.dtype)


# def fourier_slice_2d_to_1d(
#     image: jnp.ndarray,
#     angle: float = 0.0,
#     output_size: int | None = None,
# ) -> jnp.ndarray:
#     """Project a 2D image to 1D using the Fourier slice theorem.

#     ``angle`` is measured in radians counter-clockwise from the positive x axis.
#     Angle 0 sums over rows (``image.sum(axis=0)``), while pi/2 sums over columns
#     (``image.sum(axis=1)``). Arbitrary angles sample a line through the origin
#     in the centered 2D Fourier transform and inverse-transform that line.

#     Args:
#         image: A 2D JAX array.
#         angle: Projection angle in radians.
#         output_size: Number of samples in the returned profile. Defaults to
#             the image's larger spatial dimension.

#     Returns:
#         A real-valued 1D JAX array.
#     """
#     if image.ndim != 2:
#         raise ValueError("image must be a 2D array")
#     if image.shape[0] == 0 or image.shape[1] == 0:
#         raise ValueError("image dimensions must be non-zero")

#     if output_size is not None:
#         if not isinstance(output_size, int) or output_size <= 0:
#             raise ValueError("output_size must be a positive integer")

#     angle_array = jnp.asarray(angle, dtype=jnp.float32)
#     is_static = not isinstance(angle_array, jax.core.Tracer)

#     periodic_angle = jnp.mod(angle_array, jnp.pi)
#     near_zero = jnp.isclose(periodic_angle, 0.0, rtol=0.0, atol=1e-6)
#     near_90 = jnp.isclose(periodic_angle, jnp.pi / 2.0, rtol=0.0, atol=1e-6)

#     if is_static:
#         if near_zero:
#             return jnp.sum(image, axis=0)
#         if near_90:
#             return jnp.sum(image, axis=1)

#     height, width = image.shape
#     n = output_size if output_size is not None else max(image.shape)
#     frequency_offset = jnp.arange(n, dtype=jnp.float32) - n // 2

#     # The Fourier transform is centered so the zero frequency is in the middle.
#     image_unshifted = jnp.fft.ifftshift(image)
#     fourier = jnp.fft.fftshift(jnp.fft.fft2(image_unshifted))

#     fx = jnp.cos(angle) * frequency_offset / n
#     fy = jnp.sin(angle) * frequency_offset / n

#     x_coordinate = fx * width + width // 2
#     y_coordinate = fy * height + height // 2

#     sliced = _bilinear_sample(
#         fourier,
#         y_coordinate,
#         x_coordinate,
#     )
#     profile = jnp.fft.fftshift(jnp.fft.ifft(jnp.fft.ifftshift(sliced)))
#     print(jnp.linalg.norm(profile.real), jnp.linalg.norm(profile.imag), 'real and imag norms')

#     return jnp.real(profile).astype(image.real.dtype)