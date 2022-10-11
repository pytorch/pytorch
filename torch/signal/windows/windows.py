import torch

import numpy as np

from torch import Tensor
from torch.types import _dtype, _device, _layout
from torch.fft import fft

__all__ = [
    'cosine_window',
    'exponential_window',
    'gaussian_window',
]


def _window_function_checks(function_name: str, window_length: int, dtype: _dtype, layout: _layout):
    def is_floating_type(t: _dtype) -> bool:
        return t == torch.float32 or t == torch.bfloat16 or t == torch.float64 or t == torch.float16

    def is_complex_type(t: _dtype) -> bool:
        return t == torch.complex64 or t == torch.complex128 or t == torch.complex32

    if window_length < 0:
        raise RuntimeError(f'{function_name} requires non-negative window_length, got window_length={window_length}')
    if layout is torch.sparse_coo:
        raise RuntimeError(f'{function_name} is not implemented for sparse types, got: {layout}')
    if not is_floating_type(dtype) and not is_complex_type(dtype):
        raise RuntimeError(f'{function_name} expects floating point dtypes, got: {dtype}')


def exponential_window(window_length: int,
                       periodic: bool = True,
                       center: float = None,
                       tau: float = 1.0,
                       dtype: _dtype = None,
                       layout: _layout = torch.strided,
                       device: _device = None,
                       requires_grad: bool = False) -> Tensor:
    """r
    Computes a window with an exponential form. The window
    is also known as Poisson window.

    The exponential window is defined as follows:

    .. math::
        w(n) = \exp{-\frac{|n - center|}{\tau}}

    Args:
        window_length: the length of the output window. In other words, the number of points of the cosine window.
        periodic: If `True`, returns a periodic window suitable for use in spectral analysis. If `False`,
        returns a symmetric window suitable for use in filter design.
        center: this value defines where the center of the window will be located. In other words, at which
        sample the peak of the window can be found.
        tau: the decay value. For `center = 0`, it's suggested to use `tau = -(M - 1) / ln(x)`, if `` is
        the fraction of the window remaining at the end.

    Keyword args:
        {dtype}
        layout (:class:`torch.layout`, optional): the desired layout of returned window tensor. Only
              `torch.strided` (dense layout) is supported.
        {device}
        {requires_grad}
    """
    if dtype is None:
        dtype = torch.get_default_dtype()

    _window_function_checks('exponential_window', window_length, dtype, layout, device)

    if window_length == 0:
        return torch.empty((0,), dtype=dtype, layout=layout, device=device, requires_grad=requires_grad)

    if window_length == 1:
        return torch.ones((1,), dtype=dtype, layout=layout, device=device, requires_grad=requires_grad)

    if periodic:
        window_length += 1

    if periodic and center is not None:
        raise ValueError('Center must be \'None\' for periodic equal True')

    if center is None:
        center = (window_length - 1) / 2

    k = torch.arange(window_length, dtype=dtype, layout=layout, device=device, requires_grad=requires_grad)
    window = torch.exp(-torch.abs(k - center) / tau)

    return window[:-1] if periodic else window


def cosine_window(window_length: int,
                  periodic: bool = True,
                  dtype: _dtype = None,
                  layout: _layout = torch.strided,
                  device: _device = None,
                  requires_grad: bool = False) -> Tensor:
    """r
    Computes a window with a simple cosine waveform.

    The cosine window is also known as the sine window due to the following
    equality:

    .. math::
        w(n) = \cos{(\frac{\pi n}{M}) - \frac{\pi}{2})} = \sin{(\frac{\pi n}{M})}

    Where `M` is the window length.


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
    if dtype is None:
        dtype = torch.get_default_dtype()

    _window_function_checks('cosine_window', window_length, dtype, layout, device)

    if window_length == 0:
        return torch.empty((0,), dtype=dtype, layout=layout, device=device, requires_grad=requires_grad)

    if window_length == 1:
        return torch.ones((1,), dtype=dtype, layout=layout, device=device, requires_grad=requires_grad)

    if periodic:
        window_length += 1

    k = torch.arange(window_length, dtype=dtype, layout=layout, device=device, requires_grad=requires_grad)
    window = torch.sin(torch.pi / window_length * (k + .5))
    return window[:-1] if periodic else window


def gaussian_window(window_length: int,
                    periodic: bool = True,
                    std: float = 0.5,
                    dtype: _dtype = None,
                    layout: _layout = torch.strided,
                    device: _device = None,
                    requires_grad: bool = False) -> Tensor:
    """r
    Computes a window with a gaussian waveform.

    The gaussian window is defined as follows:

    .. math::
        w(n) = \exp{-\frac{1}{2}\frac{n}{\sigma}^2}

    Args:
        window_length: the length of the output window. In other words, the number of points of the cosine window.
        periodic: If `True`, returns a periodic window suitable for use in spectral analysis. If `False`,
        returns a symmetric window suitable for use in filter design.
        std: the standard deviation of the gaussian. It controls how narrow or wide the window is.


    Keyword args:
        {dtype}
        layout (:class:`torch.layout`, optional): the desired layout of returned window tensor. Only
              `torch.strided` (dense layout) is supported.
        {device}
        {requires_grad}
    """
    if dtype is None:
        dtype = torch.get_default_dtype()

    _window_function_checks('cosine_window', window_length, dtype, layout, device)

    if window_length == 0:
        return torch.empty((0,), dtype=dtype, layout=layout, device=device, requires_grad=requires_grad)

    if window_length == 1:
        return torch.ones((1,), dtype=dtype, layout=layout, device=device, requires_grad=requires_grad)

    if periodic:
        window_length += 1

    k = torch.arange(window_length, dtype=dtype, layout=layout, device=device, requires_grad=requires_grad)
    k = k - (window_length - 1.0) / 2.0
    sig2 = 2 * std * std
    window = torch.exp(-k ** 2 / sig2)
    return window[:-1] if periodic else window
