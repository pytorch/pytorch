# mypy: allow-untyped-defs
from typing import Optional, Iterable, TypeVar, Callable

import torch
from math import sqrt

from torch import Tensor
from torch._torch_docs import factory_common_args, parse_kwargs, merge_dicts

__all__ = [
    'bartlett',
    'blackman',
    'cosine',
    'exponential',
    'gaussian',
    'general_cosine',
    'general_hamming',
    'hamming',
    'hann',
    'kaiser',
    'nuttall',
]

_T = TypeVar("_T")

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
    {
        "normalization": "The window is normalized to 1 (maximum value is 1). However, the 1 doesn't appear if "
                         ":attr:`M` is even and :attr:`sym` is `True`.",
    }
)


def _add_docstr(*args: str) -> Callable[[_T], _T]:
    r"""Adds docstrings to a given decorated function.

    Specially useful when then docstrings needs string interpolation, e.g., with
    str.format().
    REMARK: Do not use this function if the docstring doesn't need string
    interpolation, just write a conventional docstring.

    Args:
        args (str):
    """

    def decorator(o: _T) -> _T:
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
    if dtype not in [torch.float32, torch.float64]:
        raise ValueError(f'{function_name} expects float32 or float64 dtypes, got: {dtype}')


@_add_docstr(
    r"""
Computes a window with an exponential waveform.
Also known as Poisson window.

The exponential window is defined as follows:

.. math::
    w_n = \exp{\left(-\frac{|n - c|}{\tau}\right)}

where `c` is the ``center`` of the window.
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

    >>> # Generates a symmetric exponential window of size 10 and with a decay value of 1.0.
    >>> # The center will be at (M - 1) / 2, where M is 10.
    >>> torch.signal.windows.exponential(10)
    tensor([0.0111, 0.0302, 0.0821, 0.2231, 0.6065, 0.6065, 0.2231, 0.0821, 0.0302, 0.0111])

    >>> # Generates a periodic exponential window and decay factor equal to .5
    >>> torch.signal.windows.exponential(10, sym=False,tau=.5)
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

    if tau <= 0:
        raise ValueError(f'Tau must be positive, got: {tau} instead.')

    if sym and center is not None:
        raise ValueError('Center must be None for symmetric windows')

    if M == 0:
        return torch.empty((0,), dtype=dtype, layout=layout, device=device, requires_grad=requires_grad)

    if center is None:
        center = (M if not sym and M > 1 else M - 1) / 2.0

    constant = 1 / tau

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
Computes a window with a simple cosine waveform, following the same implementation as SciPy.
This window is also known as the sine window.

The cosine window is defined as follows:

.. math::
    w_n = \sin\left(\frac{\pi (n + 0.5)}{M}\right)

This formula differs from the typical cosine window formula by incorporating a 0.5 term in the numerator,
which shifts the sample positions. This adjustment results in a window that starts and ends with non-zero values.

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

    >>> # Generates a symmetric cosine window.
    >>> torch.signal.windows.cosine(10)
    tensor([0.1564, 0.4540, 0.7071, 0.8910, 0.9877, 0.9877, 0.8910, 0.7071, 0.4540, 0.1564])

    >>> # Generates a periodic cosine window.
    >>> torch.signal.windows.cosine(10, sym=False)
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
    w_n = \exp{\left(-\left(\frac{n}{2\sigma}\right)^2\right)}
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

    >>> # Generates a symmetric gaussian window with a standard deviation of 1.0.
    >>> torch.signal.windows.gaussian(10)
    tensor([4.0065e-05, 2.1875e-03, 4.3937e-02, 3.2465e-01, 8.8250e-01, 8.8250e-01, 3.2465e-01, 4.3937e-02, 2.1875e-03, 4.0065e-05])

    >>> # Generates a periodic gaussian window and standard deviation equal to 0.9.
    >>> torch.signal.windows.gaussian(10, sym=False,std=0.9)
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

    if std <= 0:
        raise ValueError(f'Standard deviation must be positive, got: {std} instead.')

    if M == 0:
        return torch.empty((0,), dtype=dtype, layout=layout, device=device, requires_grad=requires_grad)

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
    w_n = I_0 \left( \beta \sqrt{1 - \left( {\frac{n - N/2}{N/2}} \right) ^2 } \right) / I_0( \beta )

where ``I_0`` is the zeroth order modified Bessel function of the first kind (see :func:`torch.special.i0`), and
``N = M - 1 if sym else M``.
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

    >>> # Generates a symmetric gaussian window with a standard deviation of 1.0.
    >>> torch.signal.windows.kaiser(5)
    tensor([4.0065e-05, 2.1875e-03, 4.3937e-02, 3.2465e-01, 8.8250e-01, 8.8250e-01, 3.2465e-01, 4.3937e-02, 2.1875e-03, 4.0065e-05])
    >>> # Generates a periodic gaussian window and standard deviation equal to 0.9.
    >>> torch.signal.windows.kaiser(5, sym=False,std=0.9)
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

    # Avoid NaNs by casting `beta` to the appropriate dtype.
    beta = torch.tensor(beta, dtype=dtype, device=device)

    start = -beta
    constant = 2.0 * beta / (M if not sym else M - 1)
    end = torch.minimum(beta, start + (M - 1) * constant)

    k = torch.linspace(start=start,
                       end=end,
                       steps=M,
                       dtype=dtype,
                       layout=layout,
                       device=device,
                       requires_grad=requires_grad)

    return torch.i0(torch.sqrt(beta * beta - torch.pow(k, 2))) / torch.i0(beta)


@_add_docstr(
    r"""
Computes the Hamming window.

The Hamming window is defined as follows:

.. math::
    w_n = \alpha - \beta\ \cos \left( \frac{2 \pi n}{M - 1} \right)
    """,
    r"""

{normalization}

Arguments:
    {M}

Keyword args:
    {sym}
    alpha (float, optional): The coefficient :math:`\alpha` in the equation above.
    beta (float, optional): The coefficient :math:`\beta` in the equation above.
    {dtype}
    {layout}
    {device}
    {requires_grad}

Examples::

    >>> # Generates a symmetric Hamming window.
    >>> torch.signal.windows.hamming(10)
    tensor([0.0800, 0.1876, 0.4601, 0.7700, 0.9723, 0.9723, 0.7700, 0.4601, 0.1876, 0.0800])

    >>> # Generates a periodic Hamming window.
    >>> torch.signal.windows.hamming(10, sym=False)
    tensor([0.0800, 0.1679, 0.3979, 0.6821, 0.9121, 1.0000, 0.9121, 0.6821, 0.3979, 0.1679])
""".format(
        **window_common_args
    ),
)
def hamming(M: int,
            *,
            sym: bool = True,
            dtype: Optional[torch.dtype] = None,
            layout: torch.layout = torch.strided,
            device: Optional[torch.device] = None,
            requires_grad: bool = False) -> Tensor:
    return general_hamming(M, sym=sym, dtype=dtype, layout=layout, device=device, requires_grad=requires_grad)


@_add_docstr(
    r"""
Computes the Hann window.

The Hann window is defined as follows:

.. math::
    w_n = \frac{1}{2}\ \left[1 - \cos \left( \frac{2 \pi n}{M - 1} \right)\right] =
    \sin^2 \left( \frac{\pi n}{M - 1} \right)
    """,
    r"""

{normalization}

Arguments:
    {M}

Keyword args:
    {sym}
    {dtype}
    {layout}
    {device}
    {requires_grad}

Examples::

    >>> # Generates a symmetric Hann window.
    >>> torch.signal.windows.hann(10)
    tensor([0.0000, 0.1170, 0.4132, 0.7500, 0.9698, 0.9698, 0.7500, 0.4132, 0.1170, 0.0000])

    >>> # Generates a periodic Hann window.
    >>> torch.signal.windows.hann(10, sym=False)
    tensor([0.0000, 0.0955, 0.3455, 0.6545, 0.9045, 1.0000, 0.9045, 0.6545, 0.3455, 0.0955])
""".format(
        **window_common_args
    ),
)
def hann(M: int,
         *,
         sym: bool = True,
         dtype: Optional[torch.dtype] = None,
         layout: torch.layout = torch.strided,
         device: Optional[torch.device] = None,
         requires_grad: bool = False) -> Tensor:
    return general_hamming(M,
                           alpha=0.5,
                           sym=sym,
                           dtype=dtype,
                           layout=layout,
                           device=device,
                           requires_grad=requires_grad)


@_add_docstr(
    r"""
Computes the Blackman window.

The Blackman window is defined as follows:

.. math::
    w_n = 0.42 - 0.5 \cos \left( \frac{2 \pi n}{M - 1} \right) + 0.08 \cos \left( \frac{4 \pi n}{M - 1} \right)
    """,
    r"""

{normalization}

Arguments:
    {M}

Keyword args:
    {sym}
    {dtype}
    {layout}
    {device}
    {requires_grad}

Examples::

    >>> # Generates a symmetric Blackman window.
    >>> torch.signal.windows.blackman(5)
    tensor([-1.4901e-08,  3.4000e-01,  1.0000e+00,  3.4000e-01, -1.4901e-08])

    >>> # Generates a periodic Blackman window.
    >>> torch.signal.windows.blackman(5, sym=False)
    tensor([-1.4901e-08,  2.0077e-01,  8.4923e-01,  8.4923e-01,  2.0077e-01])
""".format(
        **window_common_args
    ),
)
def blackman(M: int,
             *,
             sym: bool = True,
             dtype: Optional[torch.dtype] = None,
             layout: torch.layout = torch.strided,
             device: Optional[torch.device] = None,
             requires_grad: bool = False) -> Tensor:
    if dtype is None:
        dtype = torch.get_default_dtype()

    _window_function_checks('blackman', M, dtype, layout)

    return general_cosine(M, a=[0.42, 0.5, 0.08], sym=sym, dtype=dtype, layout=layout, device=device,
                          requires_grad=requires_grad)


@_add_docstr(
    r"""
Computes the Bartlett window.

The Bartlett window is defined as follows:

.. math::
    w_n = 1 - \left| \frac{2n}{M - 1} - 1 \right| = \begin{cases}
        \frac{2n}{M - 1} & \text{if } 0 \leq n \leq \frac{M - 1}{2} \\
        2 - \frac{2n}{M - 1} & \text{if } \frac{M - 1}{2} < n < M \\ \end{cases}
    """,
    r"""

{normalization}

Arguments:
    {M}

Keyword args:
    {sym}
    {dtype}
    {layout}
    {device}
    {requires_grad}

Examples::

    >>> # Generates a symmetric Bartlett window.
    >>> torch.signal.windows.bartlett(10)
    tensor([0.0000, 0.2222, 0.4444, 0.6667, 0.8889, 0.8889, 0.6667, 0.4444, 0.2222, 0.0000])

    >>> # Generates a periodic Bartlett window.
    >>> torch.signal.windows.bartlett(10, sym=False)
    tensor([0.0000, 0.2000, 0.4000, 0.6000, 0.8000, 1.0000, 0.8000, 0.6000, 0.4000, 0.2000])
""".format(
        **window_common_args
    ),
)
def bartlett(M: int,
             *,
             sym: bool = True,
             dtype: Optional[torch.dtype] = None,
             layout: torch.layout = torch.strided,
             device: Optional[torch.device] = None,
             requires_grad: bool = False) -> Tensor:
    if dtype is None:
        dtype = torch.get_default_dtype()

    _window_function_checks('bartlett', M, dtype, layout)

    if M == 0:
        return torch.empty((0,), dtype=dtype, layout=layout, device=device, requires_grad=requires_grad)

    if M == 1:
        return torch.ones((1,), dtype=dtype, layout=layout, device=device, requires_grad=requires_grad)

    start = -1
    constant = 2 / (M if not sym else M - 1)

    k = torch.linspace(start=start,
                       end=start + (M - 1) * constant,
                       steps=M,
                       dtype=dtype,
                       layout=layout,
                       device=device,
                       requires_grad=requires_grad)

    return 1 - torch.abs(k)


@_add_docstr(
    r"""
Computes the general cosine window.

The general cosine window is defined as follows:

.. math::
    w_n = \sum^{M-1}_{i=0} (-1)^i a_i \cos{ \left( \frac{2 \pi i n}{M - 1}\right)}
    """,
    r"""

{normalization}

Arguments:
    {M}

Keyword args:
    a (Iterable): the coefficients associated to each of the cosine functions.
    {sym}
    {dtype}
    {layout}
    {device}
    {requires_grad}

Examples::

    >>> # Generates a symmetric general cosine window with 3 coefficients.
    >>> torch.signal.windows.general_cosine(10, a=[0.46, 0.23, 0.31], sym=True)
    tensor([0.5400, 0.3376, 0.1288, 0.4200, 0.9136, 0.9136, 0.4200, 0.1288, 0.3376, 0.5400])

    >>> # Generates a periodic general cosine window wit 2 coefficients.
    >>> torch.signal.windows.general_cosine(10, a=[0.5, 1 - 0.5], sym=False)
    tensor([0.0000, 0.0955, 0.3455, 0.6545, 0.9045, 1.0000, 0.9045, 0.6545, 0.3455, 0.0955])
""".format(
        **window_common_args
    ),
)
def general_cosine(M, *,
                   a: Iterable,
                   sym: bool = True,
                   dtype: Optional[torch.dtype] = None,
                   layout: torch.layout = torch.strided,
                   device: Optional[torch.device] = None,
                   requires_grad: bool = False) -> Tensor:
    if dtype is None:
        dtype = torch.get_default_dtype()

    _window_function_checks('general_cosine', M, dtype, layout)

    if M == 0:
        return torch.empty((0,), dtype=dtype, layout=layout, device=device, requires_grad=requires_grad)

    if M == 1:
        return torch.ones((1,), dtype=dtype, layout=layout, device=device, requires_grad=requires_grad)

    if not isinstance(a, Iterable):
        raise TypeError("Coefficients must be a list/tuple")

    if not a:
        raise ValueError("Coefficients cannot be empty")

    constant = 2 * torch.pi / (M if not sym else M - 1)

    k = torch.linspace(start=0,
                       end=(M - 1) * constant,
                       steps=M,
                       dtype=dtype,
                       layout=layout,
                       device=device,
                       requires_grad=requires_grad)

    a_i = torch.tensor([(-1) ** i * w for i, w in enumerate(a)], device=device, dtype=dtype, requires_grad=requires_grad)
    i = torch.arange(a_i.shape[0], dtype=a_i.dtype, device=a_i.device, requires_grad=a_i.requires_grad)
    return (a_i.unsqueeze(-1) * torch.cos(i.unsqueeze(-1) * k)).sum(0)


@_add_docstr(
    r"""
Computes the general Hamming window.

The general Hamming window is defined as follows:

.. math::
    w_n = \alpha - (1 - \alpha) \cos{ \left( \frac{2 \pi n}{M-1} \right)}
    """,
    r"""

{normalization}

Arguments:
    {M}

Keyword args:
    alpha (float, optional): the window coefficient. Default: 0.54.
    {sym}
    {dtype}
    {layout}
    {device}
    {requires_grad}

Examples::

    >>> # Generates a symmetric Hamming window with the general Hamming window.
    >>> torch.signal.windows.general_hamming(10, sym=True)
    tensor([0.0800, 0.1876, 0.4601, 0.7700, 0.9723, 0.9723, 0.7700, 0.4601, 0.1876, 0.0800])

    >>> # Generates a periodic Hann window with the general Hamming window.
    >>> torch.signal.windows.general_hamming(10, alpha=0.5, sym=False)
    tensor([0.0000, 0.0955, 0.3455, 0.6545, 0.9045, 1.0000, 0.9045, 0.6545, 0.3455, 0.0955])
""".format(
        **window_common_args
    ),
)
def general_hamming(M,
                    *,
                    alpha: float = 0.54,
                    sym: bool = True,
                    dtype: Optional[torch.dtype] = None,
                    layout: torch.layout = torch.strided,
                    device: Optional[torch.device] = None,
                    requires_grad: bool = False) -> Tensor:
    return general_cosine(M,
                          a=[alpha, 1. - alpha],
                          sym=sym,
                          dtype=dtype,
                          layout=layout,
                          device=device,
                          requires_grad=requires_grad)


@_add_docstr(
    r"""
Computes the minimum 4-term Blackman-Harris window according to Nuttall.

.. math::
    w_n = 1 - 0.36358 \cos{(z_n)} + 0.48917 \cos{(2z_n)} - 0.13659 \cos{(3z_n)} + 0.01064 \cos{(4z_n)}

where ``z_n = 2 \u03c0 n/ M``.
    """,
    """

{normalization}

Arguments:
    {M}

Keyword args:
    {sym}
    {dtype}
    {layout}
    {device}
    {requires_grad}

References::

    - A. Nuttall, "Some windows with very good sidelobe behavior,"
      IEEE Transactions on Acoustics, Speech, and Signal Processing, vol. 29, no. 1, pp. 84-91,
      Feb 1981. https://doi.org/10.1109/TASSP.1981.1163506

    - Heinzel G. et al., "Spectrum and spectral density estimation by the Discrete Fourier transform (DFT),
      including a comprehensive list of window functions and some new flat-top windows",
      February 15, 2002 https://holometer.fnal.gov/GH_FFT.pdf

Examples::

    >>> # Generates a symmetric Nutall window.
    >>> torch.signal.windows.general_hamming(5, sym=True)
    tensor([3.6280e-04, 2.2698e-01, 1.0000e+00, 2.2698e-01, 3.6280e-04])

    >>> # Generates a periodic Nuttall window.
    >>> torch.signal.windows.general_hamming(5, sym=False)
    tensor([3.6280e-04, 1.1052e-01, 7.9826e-01, 7.9826e-01, 1.1052e-01])
""".format(
        **window_common_args
    ),
)
def nuttall(
        M: int,
        *,
        sym: bool = True,
        dtype: Optional[torch.dtype] = None,
        layout: torch.layout = torch.strided,
        device: Optional[torch.device] = None,
        requires_grad: bool = False
) -> Tensor:
    return general_cosine(M,
                          a=[0.3635819, 0.4891775, 0.1365995, 0.0106411],
                          sym=sym,
                          dtype=dtype,
                          layout=layout,
                          device=device,
                          requires_grad=requires_grad)
