import torch

import numpy as np

from torch import Tensor
from torch.types import _dtype, _device, _layout
from torch.fft import fft

__all__ = ['cosine_window']


def _window_function_checks(function_name: str, window_length: int, dtype: _dtype, layout: _layout, device: _device):
    def is_floating_type(t: _dtype) -> bool:
        return t == torch.float32 or t == torch.bfloat16 or t == torch.float64 or t == torch.float16

    def is_complex_type(t: _dtype) -> bool:
        return t == torch.complex64 or t == torch.complex128 or t == torch.complex32

    if window_length < 0:
        raise RuntimeError(f'{function_name} requires non-negative window_length, got window_length: {window_length}')
    if layout is torch.sparse:
        raise RuntimeError(f'{function_name} is not implemented for sparse types, got layout: {layout}')
    if not is_floating_type(dtype) and not is_complex_type(dtype):
        raise RuntimeError(f'{function_name} expects floating point dtypes, got: {dtype}')


def chebyshev_window(window_length: int,
                     attenuation: float,
                     periodic: bool = True,
                     dtype: _dtype = None,
                     layout: _layout = torch.strided,
                     device: _device = None) -> Tensor:
    _window_function_checks('chebyshev_window', window_length, dtype, layout, device)

    if window_length == 0:
        return torch.empty((0,), dtype=dtype, layout=layout, device=device)

    if window_length == 1:
        return torch.ones((1,), dtype=dtype, layout=layout, device=device)

    if not periodic:
        window_length += 1

    k = torch.arange(window_length, dtype=dtype, layout=layout, device=device)

    order = window_length - 1
    beta = np.cosh(1.0 / order * np.arccosh(np.power(10, attenuation / 20.0)))

    x = beta * torch.cos(torch.pi * k / window_length)
    window = torch.special.chebyshev_polynomial_t(x, order) / np.power(10, attenuation / 20.0)

    if window_length % 2 != 0:
        window = torch.real(fft(window))
        n = (window_length + 1) // 2
        window = torch.concat((torch.flip(window[1:n], (0,)), window[:n]))
    else:
        window = window * torch.exp(1.j * torch.pi / window_length * torch.arange(window_length))
        window = torch.real(fft(window))
        n = window_length // 2 + 1
        window = torch.concat((torch.flip(window[1:n], (0,)), window[1:n]))

    window /= torch.max(window)

    return window if periodic else window[:window_length - 1]


def exponential_window(window_length: int,
                       periodic: bool = True,
                       center: float = None,
                       tau: float = 1.0,
                       dtype: _dtype = None,
                       layout: _layout = torch.strided,
                       device: _device = None) -> Tensor:
    """r
    Computes a window with a simple cosine waveform.

    Args:
        window_length:

    Keyword args:
        {dtype}
        {device}

    """
    _window_function_checks('exponential_window', window_length, dtype, layout, device)

    if window_length == 0:
        return torch.empty((0,), dtype=dtype, layout=layout, device=device)

    if window_length == 1:
        return torch.ones((1,), dtype=dtype, layout=layout, device=device)

    if not periodic:
        window_length += 1

    if periodic and center is not None:
        raise ValueError('Center must be \'None\' for periodic equal True')

    if center is None:
        center = (window_length - 1) / 2

    k = torch.arange(window_length, dtype=dtype, layout=layout, device=device)
    window = torch.exp(-torch.abs(k - center) / tau)

    return window if periodic else window[:window_length - 1]


def cosine_window(window_length: int,
                  periodic: bool = True,
                  dtype: _dtype = None,
                  layout: _layout = torch.strided,
                  device: _device = None) -> Tensor:
    """r
    Computes a window with a simple cosine waveform.

    The cosine window is also known as the sine window due to the following
    equality:

    .. math::
        w(n) = \cos{(\frac{\pi n}{M}) - \frac{\pi}{2})} = \sin{(\frac{\pi n}{M})}

    Where `M


    Args:
        window_length: the length of the output window. In other words, the number of points of the cosine window.
        periodic: If `True`, returns a periodic window suitable for use in spectral analysis. If `False`,
        returns a symmetric window suitable for use in filter design.

    Keyword args:
        {dtype}
        layout (:class:`torch.layout`, optional): the desired layout of returned window tensor. Only
              `torch.strided` (dense layout) is supported.
        {device}
        {requires_grad}
    """
    _window_function_checks('cosine_window', window_length, dtype, layout, device)

    if window_length == 0:
        return torch.empty((0,), dtype=dtype, layout=layout, device=device)

    if window_length == 1:
        return torch.ones((1,), dtype=dtype, layout=layout, device=device)

    if not periodic:
        window_length += 1

    # k = torch.arange(window_length, dtype=dtype, layout=layout, device=device)
    k = np.arange(0, window_length)
    window = np.sin(np.pi / window_length * (k + .5))
    window = torch.from_numpy(window)

    return window if periodic else window[:window_length - 1]
