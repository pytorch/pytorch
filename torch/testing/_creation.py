"""
This module contains tensor creation utilities.
"""

import torch
from typing import Optional, List, Tuple, Union, cast
import math
import collections.abc

# Used by make_tensor for generating complex tensor.
complex_to_corresponding_float_type_map = {torch.complex32: torch.float16,
                                           torch.complex64: torch.float32,
                                           torch.complex128: torch.float64}
float_to_corresponding_complex_type_map = {v: k for k, v in complex_to_corresponding_float_type_map.items()}

def make_tensor(
    *shape: Union[int, torch.Size, List[int], Tuple[int, ...]],
    dtype: torch.dtype,
    device: Union[str, torch.device],
    low: Optional[float] = None,
    high: Optional[float] = None,
    requires_grad: bool = False,
    noncontiguous: bool = False,
    exclude_zero: bool = False
) -> torch.Tensor:
    r"""Creates a tensor with the given :attr:`shape`, :attr:`device`, and :attr:`dtype`, and filled with
    values uniformly drawn from ``[low, high)``.

    If :attr:`low` or :attr:`high` are specified and are outside the range of the :attr:`dtype`'s representable
    finite values then they are clamped to the lowest or highest representable finite value, respectively.
    If ``None``, then the following table describes the default values for :attr:`low` and :attr:`high`,
    which depend on :attr:`dtype`.

    +---------------------------+------------+----------+
    | ``dtype``                 | ``low``    | ``high`` |
    +===========================+============+==========+
    | boolean type              | ``0``      | ``2``    |
    +---------------------------+------------+----------+
    | unsigned integral type    | ``0``      | ``10``   |
    +---------------------------+------------+----------+
    | signed integral types     | ``-9``     | ``10``   |
    +---------------------------+------------+----------+
    | floating types            | ``-9``     | ``9``    |
    +---------------------------+------------+----------+
    | complex types             | ``-9``     | ``9``    |
    +---------------------------+------------+----------+

    Args:
        shape (Tuple[int, ...]): Single integer or a sequence of integers defining the shape of the output tensor.
        dtype (:class:`torch.dtype`): The data type of the returned tensor.
        device (Union[str, torch.device]): The device of the returned tensor.
        low (Optional[Number]): Sets the lower limit (inclusive) of the given range. If a number is provided it is
            clamped to the least representable finite value of the given dtype. When ``None`` (default),
            this value is determined based on the :attr:`dtype` (see the table above). Default: ``None``.
        high (Optional[Number]): Sets the upper limit (exclusive) of the given range. If a number is provided it is
            clamped to the greatest representable finite value of the given dtype. When ``None`` (default) this value
            is determined based on the :attr:`dtype` (see the table above). Default: ``None``.
        requires_grad (Optional[bool]): If autograd should record operations on the returned tensor. Default: ``False``.
        noncontiguous (Optional[bool]): If `True`, the returned tensor will be noncontiguous. This argument is
            ignored if the constructed tensor has fewer than two elements.
        exclude_zero (Optional[bool]): If ``True`` then zeros are replaced with the dtype's small positive value
            depending on the :attr:`dtype`. For bool and integer types zero is replaced with one. For floating
            point types it is replaced with the dtype's smallest positive normal number (the "tiny" value of the
            :attr:`dtype`'s :func:`~torch.finfo` object), and for complex types it is replaced with a complex number
            whose real and imaginary parts are both the smallest positive normal number representable by the complex
            type. Default ``False``.

    Raises:
        ValueError: if ``requires_grad=True`` is passed for integral `dtype`
        ValueError: If ``low > high``.
        ValueError: If either :attr:`low` or :attr:`high` is ``nan``.
        TypeError: If :attr:`dtype` isn't supported by this function.

    Examples:
        >>> from torch.testing import make_tensor
        >>> # Creates a float tensor with values in [-1, 1)
        >>> make_tensor((3,), device='cpu', dtype=torch.float32, low=-1, high=1)
        >>> # xdoctest: +SKIP
        tensor([ 0.1205, 0.2282, -0.6380])
        >>> # Creates a bool tensor on CUDA
        >>> make_tensor((2, 2), device='cuda', dtype=torch.bool)
        tensor([[False, False],
                [False, True]], device='cuda:0')
    """
    def _modify_low_high(low, high, lowest, highest, default_low, default_high, dtype):
        """
        Modifies (and raises ValueError when appropriate) low and high values given by the user (input_low, input_high) if required.
        """
        def clamp(a, l, h):
            return min(max(a, l), h)

        low = low if low is not None else default_low
        high = high if high is not None else default_high

        # Checks for error cases
        if low != low or high != high:
            raise ValueError("make_tensor: one of low or high was NaN!")
        if low > high:
            raise ValueError("make_tensor: low must be weakly less than high!")

        low = clamp(low, lowest, highest)
        high = clamp(high, lowest, highest)

        if dtype in [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]:
            return math.floor(low), math.ceil(high)

        return low, high

    if len(shape) == 1 and isinstance(shape[0], collections.abc.Sequence):
        shape = shape[0]  # type: ignore[assignment]
    shape = cast(Tuple[int, ...], tuple(shape))

    _integral_types = [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]
    _floating_types = [torch.float16, torch.bfloat16, torch.float32, torch.float64]
    _complex_types = [torch.complex32, torch.complex64, torch.complex128]
    if requires_grad and dtype not in _floating_types and dtype not in _complex_types:
        raise ValueError("make_tensor: requires_grad must be False for integral dtype")

    if dtype is torch.bool:
        result = torch.randint(0, 2, shape, device=device, dtype=dtype)  # type: ignore[call-overload]
    elif dtype is torch.uint8:
        ranges = (torch.iinfo(dtype).min, torch.iinfo(dtype).max)
        low, high = cast(Tuple[int, int], _modify_low_high(low, high, ranges[0], ranges[1], 0, 10, dtype))
        result = torch.randint(low, high, shape, device=device, dtype=dtype)   # type: ignore[call-overload]
    elif dtype in _integral_types:
        ranges = (torch.iinfo(dtype).min, torch.iinfo(dtype).max)
        low, high = _modify_low_high(low, high, ranges[0], ranges[1], -9, 10, dtype)
        result = torch.randint(low, high, shape, device=device, dtype=dtype)  # type: ignore[call-overload]
    elif dtype in _floating_types:
        ranges_floats = (torch.finfo(dtype).min, torch.finfo(dtype).max)
        low, high = _modify_low_high(low, high, ranges_floats[0], ranges_floats[1], -9, 9, dtype)
        rand_val = torch.rand(shape, device=device, dtype=dtype)
        result = high * rand_val + low * (1 - rand_val)
    elif dtype in _complex_types:
        float_dtype = complex_to_corresponding_float_type_map[dtype]
        ranges_floats = (torch.finfo(float_dtype).min, torch.finfo(float_dtype).max)
        low, high = _modify_low_high(low, high, ranges_floats[0], ranges_floats[1], -9, 9, dtype)
        real_rand_val = torch.rand(shape, device=device, dtype=float_dtype)
        imag_rand_val = torch.rand(shape, device=device, dtype=float_dtype)
        real = high * real_rand_val + low * (1 - real_rand_val)
        imag = high * imag_rand_val + low * (1 - imag_rand_val)
        result = torch.complex(real, imag)
    else:
        raise TypeError(f"The requested dtype '{dtype}' is not supported by torch.testing.make_tensor()."
                        " To request support, file an issue at: https://github.com/pytorch/pytorch/issues")

    if noncontiguous and result.numel() > 1:
        result = torch.repeat_interleave(result, 2, dim=-1)
        result = result[..., ::2]

    if exclude_zero:
        if dtype in _integral_types or dtype is torch.bool:
            replace_with = torch.tensor(1, device=device, dtype=dtype)
        elif dtype in _floating_types:
            replace_with = torch.tensor(torch.finfo(dtype).tiny, device=device, dtype=dtype)
        else:  # dtype in _complex_types:
            float_dtype = complex_to_corresponding_float_type_map[dtype]
            float_eps = torch.tensor(torch.finfo(float_dtype).tiny, device=device, dtype=float_dtype)
            replace_with = torch.complex(float_eps, float_eps)
        result[result == 0] = replace_with

    if dtype in _floating_types + _complex_types:
        result.requires_grad = requires_grad

    return result
