"""PyTorch implementation of Savitzky-Golay window.
Inspired by SciPy's savgol_filter.
from scipy.signal._savitzky_golay import savgol_coeffs, savgol_filter
"""

from typing import Literal

import torch
import torch.nn.functional as F

__all__ = ["savgol_coeffs", "savgol"]


def _float_factorial(n: int) -> float:
    """Compute factorial as float."""
    val = 1.0
    for i in range(2, int(n) + 1):
        val *= float(i)
    return val


def savgol_coeffs(
    window_length: int,
    polyorder: int,
    deriv: int = 0,
    delta: float = 1.0,
    pos: int | None = None,
    use: Literal["conv", "dot"] = "conv",
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    r"""Compute the coefficients for a 1-D Savitzky-Golay FIR filter.

    Args:
        window_length (int): Length of the filter window (must be a positive odd integer).
        polyorder (int): Order of the polynomial used to fit the samples.
        deriv (int, optional): Order of the derivative to compute. Default: 0
        delta (float, optional): Sample spacing. Default: 1.0
        pos (int, optional): Position in the window to return coefficients for. Default: center
        use (str, optional): Either 'conv' or 'dot'. Default: 'conv'
        device (torch.device, optional): Device for the returned tensor.

    Returns:
        Tensor: The filter coefficients.
    """
    if polyorder >= window_length:
        raise ValueError("polyorder must be less than window_length.")

    halflen, rem = divmod(window_length, 2)

    if pos is None:
        if rem == 0:
            pos = halflen - 0.5
        else:
            pos = halflen

    if not (0 <= pos < window_length):
        raise ValueError("pos must be nonnegative and less than window_length.")

    if use not in ["conv", "dot"]:
        raise ValueError("`use` must be 'conv' or 'dot'")

    if deriv > polyorder:
        return torch.zeros(window_length, device=device)

    # Form the design matrix A
    x = torch.arange(-pos, window_length - pos, dtype=torch.float64, device=device)

    if use == "conv":
        # Reverse so that result can be used in a convolution
        x = x.flip(0)

    order = torch.arange(polyorder + 1, dtype=torch.float64, device=device).reshape(
        -1, 1
    )
    A = x**order

    # y determines which order derivative is returned
    y = torch.zeros(polyorder + 1, dtype=torch.float64, device=device)
    # The coefficient assigned to y[deriv] scales the result
    y[deriv] = _float_factorial(deriv) / (delta**deriv)

    # Find the least-squares solution of A*c = y
    coeffs, _, _, _ = torch.linalg.lstsq(A, y, rcond=None)

    return coeffs.to(torch.float32)


def _polyder(p: torch.Tensor, m: int) -> torch.Tensor:
    """Differentiate polynomials represented with coefficients."""
    if m == 0:
        return p

    n = len(p)
    if n <= m:
        return torch.zeros_like(p[:1, ...])

    dp = p[:-m].clone()
    for k in range(m):
        rng = torch.arange(n - k - 1, m - k - 1, -1, dtype=dp.dtype, device=dp.device)
        dp *= rng.reshape((n - m,) + (1,) * (p.ndim - 1))
    return dp


def _polyfit(x: torch.Tensor, y: torch.Tensor, deg: int) -> torch.Tensor:
    """Polynomial fit using least squares."""
    # Build Vandermonde matrix (highest power first)
    vander = torch.stack([x ** (deg - i) for i in range(deg + 1)], dim=0)
    # Solve least squares
    coeffs, _, _, _ = torch.linalg.lstsq(vander.T, y, rcond=None)
    return coeffs


def _axis_slice(
    x: torch.Tensor, start: int, stop: int, axis: int
) -> torch.Tensor:
    """Get a slice along a specific axis."""
    indices = torch.arange(start, stop, device=x.device)
    return torch.index_select(x, axis, indices)


def _fit_edge(
    x: torch.Tensor,
    window_start: int,
    window_stop: int,
    interp_start: int,
    interp_stop: int,
    axis: int,
    polyorder: int,
    deriv: int,
    delta: float,
    y: torch.Tensor,
) -> None:
    """Fit polynomial to edge and interpolate in-place."""
    # Get the edge into a (window_length, -1) array
    x_edge = _axis_slice(x, window_start, window_stop, axis)
    if axis == 0 or axis == -x.ndim:
        xx_edge = x_edge
    else:
        xx_edge = x_edge.swapaxes(axis, 0)
    xx_edge = xx_edge.reshape(xx_edge.shape[0], -1)

    # Fit the edges
    poly_coeffs = _polyfit(
        torch.arange(0, window_stop - window_start, dtype=x.dtype, device=x.device),
        xx_edge,
        polyorder,
    )

    if deriv > 0:
        poly_coeffs = _polyder(poly_coeffs, deriv)


def _fit_edges_polyfit(
    x: torch.Tensor,
    window_length: int,
    polyorder: int,
    deriv: int,
    delta: float,
    axis: int,
    y: torch.Tensor,
) -> None:
    """Fit edges using polynomial interpolation."""
    halflen = window_length // 2
    _fit_edge(x, 0, window_length, 0, halflen, axis, polyorder, deriv, delta, y)
    n = x.shape[axis]
    _fit_edge(
        x, n - window_length, n, n - halflen, n, axis, polyorder, deriv, delta, y
    )


def savgol(
    x: torch.Tensor,
    window_length: int,
    polyorder: int,
    deriv: int = 0,
    delta: float = 1.0,
    axis: int = -1,
    mode: Literal["mirror", "constant", "nearest", "interp", "reflect", "replicate"] = "interp",
    cval: float = 0.0,
) -> torch.Tensor:
    r"""Computes a window with Savitzky-Golay filter.

    Args:
        x (Tensor): Input tensor.
        window_length (int): Length of the filter window (must be a positive odd integer).
        polyorder (int): Order of the polynomial used to fit the samples.
        deriv (int, optional): Order of the derivative to compute. Default: 0
        delta (float, optional): Sample spacing. Default: 1.0
        axis (int, optional): Axis along which to apply the filter. Default: -1
        mode (str, optional): Padding mode. Default: 'interp'
        cval (float, optional): Value to fill for 'constant' mode. Default: 0.0

    Returns:
        Tensor: The filtered tensor.
    """
    if mode not in ["mirror", "constant", "nearest", "interp", "reflect", "replicate"]:
        raise ValueError(
            "mode must be 'mirror', 'constant', 'nearest', 'interp', 'reflect', or 'replicate'."
        )

    x = x.clone()
    if x.dtype not in [torch.float32, torch.float64]:
        x = x.to(torch.float32)

    coeffs = savgol_coeffs(
        window_length, polyorder, deriv=deriv, delta=delta, device=x.device
    )

    if mode == "interp":
        if window_length > x.shape[axis]:
            raise ValueError(
                "If mode is 'interp', window_length must be less "
                "than or equal to the size of x."
            )

        # Convolve with coefficients
        # Move axis to last dimension for conv1d
        x_moved = x.movedim(axis, -1)
        original_shape = x_moved.shape

        # Reshape for conv1d: (batch, channels, length)
        x_reshaped = x_moved.reshape(-1, 1, x_moved.shape[-1])
        coeffs_reshaped = coeffs.reshape(1, 1, -1)

        # Apply convolution
        y_reshaped = F.conv1d(x_reshaped, coeffs_reshaped, padding=window_length // 2)

        y = y_reshaped.reshape(original_shape)
        y = y.movedim(-1, axis)

        # Fit edges with polynomial
        _fit_edges_polyfit(x, window_length, polyorder, deriv, delta, axis, y)

    else:
        # Map mode names
        mode_map = {"mirror": "reflect", "nearest": "replicate", "constant": "constant"}
        pad_mode = mode_map.get(mode, mode)

        x_moved = x.movedim(axis, -1)
        original_shape = x_moved.shape
        x_reshaped = x_moved.reshape(-1, 1, x_moved.shape[-1])

        # Pad manually
        pad_size = window_length // 2
        if pad_mode == "constant":
            x_padded = F.pad(x_reshaped, (pad_size, pad_size), mode=pad_mode, value=cval)
        else:
            x_padded = F.pad(x_reshaped, (pad_size, pad_size), mode=pad_mode)

        coeffs_reshaped = coeffs.reshape(1, 1, -1)
        y_reshaped = F.conv1d(x_padded, coeffs_reshaped)

        y = y_reshaped.reshape(original_shape)
        y = y.movedim(-1, axis)

    return y
