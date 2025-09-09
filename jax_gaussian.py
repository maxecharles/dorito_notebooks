from jax._src.typing import ArrayLike, Array
from jax import numpy as jnp
import jax
import functools
from collections.abc import Callable, Sequence
from scipy.ndimage import _ni_support


def _gaussian(x, sigma):
    return jnp.exp(-0.5 / sigma**2 * x**2) / jnp.sqrt(2 * jnp.pi * sigma**2)


def _grad_order(func, order):
    """Compute higher order grads recursively"""
    if order == 0:
        return func

    return jax.grad(_grad_order(func, order - 1))


def _gaussian_kernel1d(sigma, order, radius):
    """Computes a 1-D Gaussian convolution kernel"""
    if order < 0:
        raise ValueError(f"Order must be non-negative, got {order}")

    x = jnp.arange(-radius, radius + 1, dtype=jnp.float32)
    func = _grad_order(functools.partial(_gaussian, sigma=sigma), order)
    kernel = jax.vmap(func)(x)

    if order == 0:
        return kernel / jnp.sum(kernel)

    return kernel


def gaussian_filter1d(
    input: ArrayLike,
    sigma: float,
    axis: int = -1,
    order: int = 0,
    mode: str = "reflect",
    cval: float = 0.0,
    truncate: float = 4.0,
    *,
    radius: int | None = None,
    method: str = "auto",
):
    """Compute a 1D Gaussian filter on the input array along the specified axis.
    Args:
        input: N-dimensional input array to filter.
        sigma: The standard deviation of the Gaussian filter.
        axis: The axis along which to apply the filter.
        order: The order of the Gaussian filter.
        mode: The mode to use for padding the input array. See :func:`jax.numpy.pad` for more details.
        cval: The value to use for padding the input array.
        truncate: The number of standard deviations to include in the filter.
        radius: The radius of the filter. Overrides `truncate` if provided.
        method: The method to use for the convolution.
    Returns:
        The filtered array.
    Examples:
        >>> from jax import numpy as jnp
        >>> import jax
        >>> input = jnp.arange(12.0).reshape(3, 4)
        >>> input
        Array([[ 0.,  1.,  2.,  3.],
               [ 4.,  5.,  6.,  7.],
               [ 8.,  9., 10., 11.]], dtype=float32)
        >>> jax.scipy.ndimage.gaussian_filter1d(input, sigma=1.0, axis=0, order=0)
       Array([[2.8350844, 3.8350847, 4.8350844, 5.8350844],
              [4.0000005, 5.       , 6.       , 7.0000005],
              [5.1649156, 6.1649156, 7.164916 , 8.164916 ]], dtype=float32)
    """
    if radius is None:
        radius = int(truncate * sigma + 0.5)

    if radius < 0:
        raise ValueError(f"Radius must be non-negative, got {radius}")

    if sigma <= 0:
        raise ValueError(f"Sigma must be positive, got {sigma}")

    pad_width = [(0, 0)] * input.ndim
    pad_width[axis] = (int(radius), int(radius))

    pad_kwargs = {"mode": mode}

    if mode == "constant":
        # jnp.pad errors if constant_values is provided and mode is not 'constant'
        pad_kwargs["constant_values"] = cval

    input_pad = jnp.pad(input, pad_width=pad_width, **pad_kwargs)

    kernel = _gaussian_kernel1d(sigma, order=order, radius=radius)

    axes = list(range(input.ndim))
    axes.pop(axis)
    kernel = jnp.expand_dims(kernel, axes)

    # boundary handling is done by jnp.pad, so we use the fixed valid mode
    return jax.scipy.signal.convolve(input_pad, kernel, mode="valid", method=method)


def gaussian_filter(
    input: ArrayLike,
    sigma: float | Sequence[float],
    order: int | Sequence[int] = 0,
    mode: str = "reflect",
    cval: float | Sequence[float] = 0.0,
    truncate: float | Sequence[float] = 4.0,
    *,
    radius: None | Sequence[int] = None,
    axes: Sequence[int] = None,
    method="auto",
):
    """Gaussian filter for N-dimensional input

    Args:
       input: N-dimensional input array to filter.
       sigma: The standard deviation of the Gaussian filter.
       order: The order of the Gaussian filter.
       mode: The mode to use for padding the input array. See :func:`jax.numpy.pad` for more details.
       cval: The value to use for padding the input array.
       truncate: The number of standard deviations to include in the filter.
       radius: The radius of the filter. Overrides `truncate` if provided.
       method: The method to use for the convolution.
    """
    axes = _ni_support._check_axes(axes, input.ndim)
    num_axes = len(axes)
    orders = _ni_support._normalize_sequence(order, num_axes)
    sigmas = _ni_support._normalize_sequence(sigma, num_axes)
    modes = _ni_support._normalize_sequence(mode, num_axes)
    radii = _ni_support._normalize_sequence(radius, num_axes)

    # the loop goes over the input axes, so it is always low-dimensional and
    # keeping a Python loop is ok
    for idx in range(input.ndim):
        input = gaussian_filter1d(
            input,
            sigmas[idx],
            axis=idx,
            order=orders[idx],
            mode=modes[idx],
            cval=cval,
            truncate=truncate,
            radius=radii[idx],
            method=method,
        )

    return input
