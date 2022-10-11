import torch

from torch import Tensor
from torch.types import _dtype, _device, _layout

__all__ = [
    'cosine',
    'exponential',
    'gaussian',
]


def _window_function_checks(function_name: str, window_length: int, dtype: _dtype, layout: _layout) -> None:
    r"""Performs common checks for all the defined windows.
     This function should be called before computing any window

     Args:
         function_name (str): name of the window function.
         window_length (int): length of the window.
         dtype (:class:`torch.dtype`): the desired data type of the window tensor.
         layout (:class:`torch.layout`): the desired layout of the window tensor.
     """

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


def exponential(window_length: int,
                periodic: bool = True,
                center: float = None,
                tau: float = 1.0,
                dtype: _dtype = None,
                layout: _layout = torch.strided,
                device: _device = None,
                requires_grad: bool = False) -> Tensor:
    r"""Computes a window with an exponential form.
    Also known as Poisson window.

    The exponential window is defined as follows:

    .. math::
        w(n) = \exp{\left(-\frac{|n - center|}{\tau}\right)}

    Args:
        window_length (int): the length of the output window. In other words, the number of points of the cosine window.
        periodic (bool, optional): If `True`, returns a periodic window suitable for use in spectral analysis.
            If `False`, returns a symmetric window suitable for use in filter design. Default: `True`.
        center (float, optional): his value defines where the center of the window will be located.
            In other words, at which sample the peak of the window can be found.
            Default: `window_length / 2` if `periodic` is `True` (default), else `(window_length - 1) / 2`.
        tau (float, optional): the decay value. For `center = 0`, it's suggested to use
            :math:`\tau = -\frac{(M - 1)}{\ln(x)}`, if `x` is the fraction of the window remaining at the end.
            Default: 1.0.

    Keyword args:
        dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
            Default: if ``None``, uses a global default (see :func:`torch.set_default_tensor_type`).
        layout (:class:`torch.layout`, optional): the desired layout of returned window tensor.
            Only `torch.strided` (dense layout) is supported.
        device (:class:`torch.device`, optional): the desired device of returned tensor.
            Default: if ``None``, uses the current device for the default tensor type
            (see :func:`torch.set_default_tensor_type`).
            :attr:`device` will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.
        requires_grad (bool, optional): If autograd should record operations on the returned tensor. Default: ``False``.

    Examples:
        >>> # Generate an exponential window without keyword args.
        >>> torch.signal.windows.exponential(10)
        tensor([0.0067, 0.0183, 0.0498, 0.1353, 0.3679, 1.0000, 0.3679, 0.1353, 0.0498,
        0.0183])

        >>> # Generate a symmetric exponential window and decay factor equal to .5
        >>> torch.signal.windows.exponential(10,periodic=False,tau=.5)
        tensor([1.2341e-04, 9.1188e-04, 6.7379e-03, 4.9787e-02, 3.6788e-01, 3.6788e-01,
        4.9787e-02, 6.7379e-03, 9.1188e-04, 1.2341e-04])
    """
    if dtype is None:
        dtype = torch.get_default_dtype()

    _window_function_checks('exponential_window', window_length, dtype, layout)

    if window_length == 0:
        return torch.empty((0,), dtype=dtype, layout=layout, device=device, requires_grad=requires_grad)

    if window_length == 1:
        return torch.ones((1,), dtype=dtype, layout=layout, device=device, requires_grad=requires_grad)

    if periodic and center is not None:
        raise ValueError('Center must be \'None\' for periodic equal True')

    if center is None:
        center = -(window_length if periodic else window_length - 1) / 2.0

    k = torch.arange(start=center,
                     end=center + window_length,
                     dtype=dtype, layout=layout,
                     device=device,
                     requires_grad=requires_grad)

    return torch.exp(-torch.abs(k) / tau)


def cosine(window_length: int,
           periodic: bool = True,
           dtype: _dtype = None,
           layout: _layout = torch.strided,
           device: _device = None,
           requires_grad: bool = False) -> Tensor:
    r"""Computes a window with a simple cosine waveform.
    Also known as the sine window.

    The cosine window is defined as follows:

    .. math::
        w(n) = \cos{\left(\frac{\pi n}{M} - \frac{\pi}{2}\right)} = \sin{\left(\frac{\pi n}{M}\right)}

    Where `M` is the window length.

    Args:
        window_length (int): the length of the output window.
            In other words, the number of points of the cosine window.
        periodic (bool, optional): If `True`, returns a periodic window suitable for use in spectral analysis.
            If `False`, returns a symmetric window suitable for use in filter design. Default: `True`.

    Keyword args:
        dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
            Default: if ``None``, uses a global default (see :func:`torch.set_default_tensor_type`).
        layout (:class:`torch.layout`, optional): the desired layout of returned window tensor.
            Only `torch.strided` (dense layout) is supported.
        device (:class:`torch.device`, optional): the desired device of returned tensor.
            Default: if ``None``, uses the current device for the default tensor type
            (see :func:`torch.set_default_tensor_type`).
            :attr:`device` will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.
        requires_grad (bool, optional): If autograd should record operations on the returned tensor. Default: ``False``.

    Examples:
        >>> # Generate a cosine window without keyword args.
        >>> torch.signal.windows.cosine(10)
        tensor([0.1423, 0.4154, 0.6549, 0.8413, 0.9595, 1.0000, 0.9595, 0.8413, 0.6549,
        0.4154])

        >>> # Generate a symmetric cosine window.
        >>> torch.signal.windows.cosine(10,periodic=False)
        tensor([0.1564, 0.4540, 0.7071, 0.8910, 0.9877, 0.9877, 0.8910, 0.7071, 0.4540,
        0.1564])
    """
    if dtype is None:
        dtype = torch.get_default_dtype()

    _window_function_checks('cosine_window', window_length, dtype, layout)

    if window_length == 0:
        return torch.empty((0,), dtype=dtype, layout=layout, device=device, requires_grad=requires_grad)

    if window_length == 1:
        return torch.ones((1,), dtype=dtype, layout=layout, device=device, requires_grad=requires_grad)

    if periodic:
        window_length += 1

    k = torch.arange(window_length, dtype=dtype, layout=layout, device=device, requires_grad=requires_grad)
    window = torch.sin(torch.pi / window_length * (k + .5))

    return window[:-1] if periodic else window


def gaussian(window_length: int,
             periodic: bool = True,
             std: float = 0.5,
             dtype: _dtype = None,
             layout: _layout = torch.strided,
             device: _device = None,
             requires_grad: bool = False) -> Tensor:
    r"""Computes a window with a gaussian waveform.

    The gaussian window is defined as follows:

    .. math::
        w(n) = \exp{\left(-\left(\frac{n}{2\sigma}\right)^2\right)}

    Args:
        window_length (int): the length of the output window.
            In other words, the number of points of the cosine window.
        periodic (bool, optional): If `True`, returns a periodic window suitable for use in spectral analysis.
            If `False`, returns a symmetric window suitable for use in filter design. Default: `True`
        std (float, optional): the standard deviation of the gaussian. It controls how narrow or wide the window is.
            Default: 0.5.

    Keyword args:
        dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
            Default: if ``None``, uses a global default (see :func:`torch.set_default_tensor_type`).
        layout (:class:`torch.layout`, optional): the desired layout of returned window tensor.
            Only `torch.strided` (dense layout) is supported.
        device (:class:`torch.device`, optional): the desired device of returned tensor.
            Default: if ``None``, uses the current device for the default tensor type
            (see :func:`torch.set_default_tensor_type`).
            :attr:`device` will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.
        requires_grad (bool, optional): If autograd should record operations on the returned tensor. Default: ``False``.

    Examples:
        >>> # Generate a gaussian window without keyword args.
        >>> torch.signal.windows.gaussian(10)
        tensor([1.9287e-22, 1.2664e-14, 1.5230e-08, 3.3546e-04, 1.3534e-01, 1.0000e+00,
        1.3534e-01, 3.3546e-04, 1.5230e-08, 1.2664e-14])

        >>> # Generate a symmetric gaussian window and standard deviation equal to 0.9.
        >>> torch.signal.windows.gaussian(10,periodic=False,std=0.9)
        tensor([3.7267e-06, 5.1998e-04, 2.1110e-02, 2.4935e-01, 8.5700e-01, 8.5700e-01,
        2.4935e-01, 2.1110e-02, 5.1998e-04, 3.7267e-06])
    """

    if dtype is None:
        dtype = torch.get_default_dtype()

    _window_function_checks('cosine_window', window_length, dtype, layout)

    if window_length == 0:
        return torch.empty((0,), dtype=dtype, layout=layout, device=device, requires_grad=requires_grad)

    if window_length == 1:
        return torch.ones((1,), dtype=dtype, layout=layout, device=device, requires_grad=requires_grad)

    start = -(window_length if periodic else window_length - 1) / 2.0

    k = torch.arange(start,
                     start + window_length,
                     dtype=dtype,
                     layout=layout,
                     device=device,
                     requires_grad=requires_grad)

    return torch.exp(-(k / std) ** 2 / 2)
