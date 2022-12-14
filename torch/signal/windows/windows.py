from typing import Optional

import torch
from math import sqrt

from torch import Tensor
from torch._prims_common import is_float_dtype
from torch._torch_docs import factory_common_args, parse_kwargs, merge_dicts

__all__ = [
    'cosine',
    'exponential',
    'gaussian',
    'kaiser',
]

window_common_args = merge_dicts(
    parse_kwargs(
        """
    M (int): the length of the window.
        In other words, the number of points of the returned window.
    sym (bool, optional): If `False`, returns a periodic window suitable for use in spectral analysis.
        If `True`, returns a symmetric window suitable for use in filter design. Default: `True`.
"""
    ),
    factory_common_args,
    {"normalization": "The window is normalized to 1 (maximum value is 1). However, the 1 doesn't appear if "
                      ":attr:`M` is even and :attr:`sym` is `True`."}
)


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
        o.__doc__ = "".join(args)
        return o

    return decorator


def _window_function_checks(function_name: str, M: int, dtype: torch.dtype, layout: torch.layout) -> None:
    r"""Performs common checks for all the defined windows.
     This function should be called before computing any window.

     Args:
         function_name (str): name of the window function.
         M (int): length of the window.
         dtype (:class:`torch.dtype`): the desired data type of returned tensor.
         layout (:class:`torch.layout`): the desired layout of returned tensor.
     """
    if M < 0:
        raise ValueError(f'{function_name} requires non-negative window length, got M={M}')
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
    w(n) = \exp{\left(-\frac{|n - c|}{\tau}\right)}

where `c` is the center of the window.
    """,
    r"""

{normalization}

Args:
    {M}

Keyword args:
    center (float, optional): where the center of the window will be located.
        Default: `M / 2` if `sym` is `False`, else `(M - 1) / 2`.
    tau (float, optional): the decay value.
        Tau is generally associated with a percentage, that means, that the value should
        vary within the interval (0, 100]. If tau is 100, it is considered the uniform window.
        Default: 1.0.
    {sym}
    {dtype}
    {layout}
    {device}
    {requires_grad}

Examples::

    >>> # Generate a symmetric exponential window of size 10 and with a decay value of 1.0.
    >>> # The center will be at (M - 1) / 2, where M is 10.
    >>> torch.signal.windows.exponential(10)
    tensor([0.0111, 0.0302, 0.0821, 0.2231, 0.6065, 0.6065, 0.2231, 0.0821, 0.0302, 0.0111])

    >>> # Generate a periodic exponential window and decay factor equal to .5
    >>> torch.signal.windows.exponential(10,sym=False,tau=.5)
    tensor([4.5400e-05, 3.3546e-04, 2.4788e-03, 1.8316e-02, 1.3534e-01, 1.0000e+00, 1.3534e-01, 1.8316e-02, 2.4788e-03, 3.3546e-04])
    """.format(
        **window_common_args
    ),
)
def exponential(
        M: int,
        *,
        center: Optional[float] = None,
        tau: float = 1.0,
        sym: bool = True,
        dtype: Optional[torch.dtype] = None,
        layout: torch.layout = torch.strided,
        device: Optional[torch.device] = None,
        requires_grad: bool = False
) -> Tensor:
    if dtype is None:
        dtype = torch.get_default_dtype()

    _window_function_checks('exponential', M, dtype, layout)

    if M == 0:
        return torch.empty((0,), dtype=dtype, layout=layout, device=device, requires_grad=requires_grad)

    if tau <= 0:
        raise ValueError(f'Tau must be positive, got: {tau} instead.')

    if sym and center is not None:
        raise ValueError('Center must be None for symmetric windows')

    if center is None:
        center = (M if not sym and M > 1 else M - 1) / 2.0

    constant = 1 / tau

    """
    Note that non-integer step is subject to floating point rounding errors when comparing against end;
    thus, to avoid inconsistency, we added an epsilon equal to `step / 2` to `end`.
    """
    k = torch.linspace(start=-center * constant,
                       end=(-center + (M - 1)) * constant,
                       steps=M,
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
    """,
    r"""

{normalization}

Args:
    {M}

Keyword args:
    {sym}
    {dtype}
    {layout}
    {device}
    {requires_grad}

Examples::

    >>> # Generate a symmetric cosine window.
    >>> torch.signal.windows.cosine(10)
    tensor([0.1564, 0.4540, 0.7071, 0.8910, 0.9877, 0.9877, 0.8910, 0.7071, 0.4540, 0.1564])

    >>> # Generate a periodic cosine window.
    >>> torch.signal.windows.cosine(10,sym=False)
    tensor([0.1423, 0.4154, 0.6549, 0.8413, 0.9595, 1.0000, 0.9595, 0.8413, 0.6549, 0.4154])
""".format(
        **window_common_args,
    ),
)
def cosine(
        M: int,
        *,
        sym: bool = True,
        dtype: Optional[torch.dtype] = None,
        layout: torch.layout = torch.strided,
        device: Optional[torch.device] = None,
        requires_grad: bool = False
) -> Tensor:
    if dtype is None:
        dtype = torch.get_default_dtype()

    _window_function_checks('cosine', M, dtype, layout)

    if M == 0:
        return torch.empty((0,), dtype=dtype, layout=layout, device=device, requires_grad=requires_grad)

    start = 0.5
    constant = torch.pi / (M + 1 if not sym and M > 1 else M)

    k = torch.linspace(start=start * constant,
                       end=(start + (M - 1)) * constant,
                       steps=M,
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

{normalization}

Args:
    {M}

Keyword args:
    std (float, optional): the standard deviation of the gaussian. It controls how narrow or wide the window is.
        Default: 1.0.
    {sym}
    {dtype}
    {layout}
    {device}
    {requires_grad}

Examples::

    >>> # Generate a symmetric gaussian window with a standard deviation of 1.0.
    >>> torch.signal.windows.gaussian(10)
    tensor([4.0065e-05, 2.1875e-03, 4.3937e-02, 3.2465e-01, 8.8250e-01, 8.8250e-01, 3.2465e-01, 4.3937e-02, 2.1875e-03, 4.0065e-05])

    >>> # Generate a periodic gaussian window and standard deviation equal to 0.9.
    >>> torch.signal.windows.gaussian(10,sym=False,std=0.9)
    tensor([1.9858e-07, 5.1365e-05, 3.8659e-03, 8.4658e-02, 5.3941e-01, 1.0000e+00, 5.3941e-01, 8.4658e-02, 3.8659e-03, 5.1365e-05])
""".format(
        **window_common_args,
    ),
)
def gaussian(
        M: int,
        *,
        std: float = 1.0,
        sym: bool = True,
        dtype: Optional[torch.dtype] = None,
        layout: torch.layout = torch.strided,
        device: Optional[torch.device] = None,
        requires_grad: bool = False
) -> Tensor:
    if dtype is None:
        dtype = torch.get_default_dtype()

    _window_function_checks('gaussian', M, dtype, layout)

    if M == 0:
        return torch.empty((0,), dtype=dtype, layout=layout, device=device, requires_grad=requires_grad)

    if std <= 0:
        raise ValueError(f'Standard deviation must be positive, got: {std} instead.')

    start = -(M if not sym and M > 1 else M - 1) / 2.0

    constant = 1 / (std * sqrt(2))

    k = torch.linspace(start=start * constant,
                       end=(start + (M - 1)) * constant,
                       steps=M,
                       dtype=dtype,
                       layout=layout,
                       device=device,
                       requires_grad=requires_grad)

    return torch.exp(-k ** 2)


@_add_docstr(
    r"""
Computes the Kaiser window.

The Kaiser window is defined as follows:

.. math::
    out_i = I_0 \left( \beta \sqrt{1 - \left( {\frac{i - N/2}{N/2}} \right) ^2 } \right) / I_0( \beta )

where ``I_0`` is the zeroth order modified Bessel function of the first kind (see :func:`torch.special.i0`), and
``N = M - 1 if sym else M``.

``M`` is the window length.

    """,
    r"""

{normalization}

Args:
    {M}

Keyword args:
    beta (float, optional): shape parameter for the window. Must be non-negative. Default: 12.0
    {sym}
    {dtype}
    {layout}
    {device}
    {requires_grad}

Examples::

    >>> # Generate a symmetric gaussian window with a standard deviation of 1.0.
    >>> torch.signal.windows.kaiser(5)
    tensor([4.0065e-05, 2.1875e-03, 4.3937e-02, 3.2465e-01, 8.8250e-01, 8.8250e-01, 3.2465e-01, 4.3937e-02, 2.1875e-03, 4.0065e-05])

    >>> # Generate a periodic gaussian window and standard deviation equal to 0.9.
    >>> torch.signal.windows.kaiser(5,sym=False,std=0.9)
    tensor([1.9858e-07, 5.1365e-05, 3.8659e-03, 8.4658e-02, 5.3941e-01, 1.0000e+00, 5.3941e-01, 8.4658e-02, 3.8659e-03, 5.1365e-05])
""".format(
        **window_common_args,
    ),
)
def kaiser(
        M: int,
        *,
        beta: float = 12.0,
        sym: bool = True,
        dtype: Optional[torch.dtype] = None,
        layout: torch.layout = torch.strided,
        device: Optional[torch.device] = None,
        requires_grad: bool = False
) -> Tensor:
    if dtype is None:
        dtype = torch.get_default_dtype()

    _window_function_checks('kaiser', M, dtype, layout)

    if beta < 0:
        raise ValueError(f'beta must be non-negative, got: {beta} instead.')

    if M == 0:
        return torch.empty((0,), dtype=dtype, layout=layout, device=device, requires_grad=requires_grad)

    if M == 1:
        return torch.ones((1,), dtype=dtype, layout=layout, device=device, requires_grad=requires_grad)

    start = -beta
    constant = 2.0 * beta / (M if not sym else M - 1)

    k = torch.linspace(start=start,
                       end=start + (M - 1) * constant,
                       steps=M,
                       dtype=dtype,
                       layout=layout,
                       device=device,
                       requires_grad=requires_grad)

    return torch.i0(torch.sqrt(beta * beta - torch.pow(k, 2))) / torch.i0(torch.tensor(beta))
