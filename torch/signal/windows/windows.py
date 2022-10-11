import torch
import numpy as np

from torch import Tensor
from torch._prims_common import is_float_dtype, is_complex_dtype
from torch._torch_docs import factory_common_args
from torch.types import _dtype, _device, _layout

__all__ = [
    'cosine',
    'exponential',
    'gaussian',
]


def _add_docstr(*args):
    r"""Adds docstrings to a given decorated function.

    Specially useful when then docstrings needs string interpolation, e.g., with
    str.format().
    REMARK: Do not use this function if the docstring doesn't need string
    interpolation, just write a conventional docstring.

    Args:
        args (str):
    """
    def decorator(o):
        o.__doc__ = ""
        for arg in args:
            o.__doc__ += arg
        return o
    return decorator


def _window_function_checks(function_name: str, window_length: int, dtype: _dtype, layout: _layout) -> None:
    r"""Performs common checks for all the defined windows.
     This function should be called before computing any window

     Args:
         function_name (str): name of the window function.
         window_length (int): length of the window.
         dtype (:class:`torch.dtype`): the desired data type of the window tensor.
         layout (:class:`torch.layout`): the desired layout of the window tensor.
     """
    if window_length < 0:
        raise ValueError(f'{function_name} requires non-negative window_length, got window_length={window_length}')
    if layout is not torch.strided:
        raise ValueError(f'{function_name} is implemented for strided tensors only, got: {layout}')
    if not is_float_dtype(dtype):
        raise ValueError(f'{function_name} expects floating point dtypes, got: {dtype}')


@_add_docstr(
    r"""
Computes a window with an exponential waveform.
Also known as Poisson window.

The exponential window is defined as follows:

.. math::
    w(n) = \exp{\left(-\frac{|n - center|}{\tau}\right)}
    """,
    r"""

Args:
    M (int): the length of the output window.
        In other words, the number of points of the exponential window.
    center (float, optional): where the center of the window will be located.
        In other words, at which sample the peak of the window can be found.
        Default: `window_length / 2` if `periodic` is `True` (default), else `(window_length - 1) / 2`.
    tau (float, optional): the decay value.
        For `center = 0`, it's suggested to use :math:`\tau = -\frac{(M - 1)}{\ln(x)}`,
        if `x` is the fraction of the window remaining at the end. Default: 1.0.
    sym (bool, optional): If `False`, returns a periodic window suitable for use in spectral analysis.
        If `True`, returns a symmetric window suitable for use in filter design. Default: `True`.
    """ +
    r"""

.. note::
    The window is normalized to 1 (maximum value is 1), however, the 1 doesn't appear if `M` is even
    and `sym` is `True`.

Keyword args:
    {dtype}
    {layout}
    {device}
    {requires_grad}

Examples:
    >>> # Generate an exponential window without keyword args.
    >>> torch.signal.windows.exponential(10)
    tensor([0.0067, 0.0183, 0.0498, 0.1353, 0.3679, 1.0000, 0.3679, 0.1353, 0.0498,
    0.0183])

    >>> # Generate a symmetric exponential window and decay factor equal to .5
    >>> torch.signal.windows.exponential(10,periodic=False,tau=.5)
    tensor([1.2341e-04, 9.1188e-04, 6.7379e-03, 4.9787e-02, 3.6788e-01, 3.6788e-01,
    4.9787e-02, 6.7379e-03, 9.1188e-04, 1.2341e-04])
    """.format(
        **factory_common_args
    ),
)
def exponential(M: int, center: float = None, tau: float = 1.0, sym: bool = True, *, dtype: _dtype = None,
                layout: _layout = torch.strided, device: _device = None, requires_grad: bool = False) -> Tensor:
    if dtype is None:
        dtype = torch.get_default_dtype()

    _window_function_checks('exponential', M, dtype, layout)

    if M == 0:
        return torch.empty((0,), dtype=dtype, layout=layout, device=device, requires_grad=requires_grad)

    if M == 1:
        return torch.ones((1,), dtype=dtype, layout=layout, device=device, requires_grad=requires_grad)

    if tau <= 0:
        raise ValueError(f'Tau must be positive, got: {tau} instead.')

    if not sym and center is not None:
        raise ValueError('Center must be \'None\' for non-symmetric windows')

    if center is None:
        center = -(M if not sym else M - 1) / 2.0

    constant = 1 / tau

    """
    Note that non-integer step is subject to floating point rounding errors when comparing against end;
    thus, to avoid inconsistency, we added an epsilon equal to `step / 2` to `end`.
    """
    k = torch.arange(start=center * constant,
                     end=(center + (M - 1)) * constant + constant / 2,
                     step=constant,
                     dtype=dtype,
                     layout=layout,
                     device=device,
                     requires_grad=requires_grad)

    return torch.exp(-torch.abs(k))


@_add_docstr(
    r"""
Computes a window with a simple cosine waveform.
Also known as the sine window.

The cosine window is defined as follows:

.. math::
    w(n) = \cos{\left(\frac{\pi n}{M} - \frac{\pi}{2}\right)} = \sin{\left(\frac{\pi n}{M}\right)}

Where `M` is the length of the window.
    """,
    r"""

Args:
    M (int): the length of the output window.
        In other words, the number of points of the cosine window.
    sym (bool, optional): If `False`, returns a periodic window suitable for use in spectral analysis.
        If `True`, returns a symmetric window suitable for use in filter design. Default: `True`.

.. note::
    The window is normalized to 1 (maximum value is 1), however, the 1 doesn't appear if `M` is even
    and `sym` is `True`.

Keyword args:
    {dtype}
    {layout}
    {device}
    {requires_grad}

Examples:
    >>> # Generate a cosine window without keyword args.
    >>> torch.signal.windows.cosine(10)
    tensor([0.1423, 0.4154, 0.6549, 0.8413, 0.9595, 1.0000, 0.9595, 0.8413, 0.6549,
    0.4154])

    >>> # Generate a symmetric cosine window.
    >>> torch.signal.windows.cosine(10,periodic=False)
    tensor([0.1564, 0.4540, 0.7071, 0.8910, 0.9877, 0.9877, 0.8910, 0.7071, 0.4540,
    0.1564])
""".format(
        **factory_common_args
    ),
)
def cosine(M: int,
           sym: bool = True,
           *,
           dtype: _dtype = None,
           layout: _layout = torch.strided,
           device: _device = None,
           requires_grad: bool = False) -> Tensor:
    if dtype is None:
        dtype = torch.get_default_dtype()

    _window_function_checks('cosine', M, dtype, layout)

    if M == 0:
        return torch.empty((0,), dtype=dtype, layout=layout, device=device, requires_grad=requires_grad)

    if M == 1:
        return torch.ones((1,), dtype=dtype, layout=layout, device=device, requires_grad=requires_grad)

    start = 0.5
    constant = torch.pi / (M + 1 if not sym else M)

    """
    Note that non-integer step is subject to floating point rounding errors when comparing against end;
    thus, to avoid inconsistency, we added an epsilon equal to `step / 2` to `end`.
    """
    k = torch.arange(start=start * constant,
                     end=(start + (M - 1)) * constant + constant / 2,
                     step=constant,
                     dtype=dtype,
                     layout=layout,
                     device=device,
                     requires_grad=requires_grad)

    return torch.sin(k)


@_add_docstr(
    r"""
Computes a window with a gaussian waveform.

The gaussian window is defined as follows:

.. math::
    w(n) = \exp{\left(-\left(\frac{n}{2\sigma}\right)^2\right)}
    """,
    r"""

Args:
    M (int): the length of the output window.
        In other words, the number of points of the cosine window.
    std (float): the standard deviation of the gaussian. It controls how narrow or wide the window is.
    sym (bool, optional): If `False`, returns a periodic window suitable for use in spectral analysis.
        If `True`, returns a symmetric window suitable for use in filter design. Default: `True`

.. note::
    The window is normalized to 1 (maximum value is 1), however, the 1 doesn't appear if `M` is even
    and `sym` is `True`.

Keyword args:
    {dtype}
    {layout}
    {device}
    {requires_grad}

Examples:
    >>> # Generate a gaussian window without keyword args.
    >>> torch.signal.windows.gaussian(10)
    tensor([1.9287e-22, 1.2664e-14, 1.5230e-08, 3.3546e-04, 1.3534e-01, 1.0000e+00,
    1.3534e-01, 3.3546e-04, 1.5230e-08, 1.2664e-14])

    >>> # Generate a symmetric gaussian window and standard deviation equal to 0.9.
    >>> torch.signal.windows.gaussian(10,periodic=False,std=0.9)
    tensor([3.7267e-06, 5.1998e-04, 2.1110e-02, 2.4935e-01, 8.5700e-01, 8.5700e-01,
    2.4935e-01, 2.1110e-02, 5.1998e-04, 3.7267e-06])
""".format(
        **factory_common_args
    ),
)
def gaussian(M: int,
             std: float,
             sym: bool = True,
             *,
             dtype: _dtype = None,
             layout: _layout = torch.strided,
             device: _device = None,
             requires_grad: bool = False) -> Tensor:
    if dtype is None:
        dtype = torch.get_default_dtype()

    _window_function_checks('gaussian', M, dtype, layout)

    if M == 0:
        return torch.empty((0,), dtype=dtype, layout=layout, device=device, requires_grad=requires_grad)

    if M == 1:
        return torch.ones((1,), dtype=dtype, layout=layout, device=device, requires_grad=requires_grad)

    if std <= 0:
        raise ValueError(f'Standard deviation must be positive, got: {std} instead.')

    start = -(M if not sym else M - 1) / 2.0

    constant = 1 / (std * np.sqrt(2))

    """
    Note that non-integer step is subject to floating point rounding errors when comparing against end;
    thus, to avoid inconsistency, we added an epsilon equal to `step / 2` to `end`.
    """
    k = torch.arange(start=start * constant,
                     end=(start + (M - 1)) * constant + constant / 2,
                     step=constant,
                     dtype=dtype,
                     layout=layout,
                     device=device,
                     requires_grad=requires_grad)

    return torch.exp(-k ** 2)
