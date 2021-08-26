"""
This module contains tensor creation utilities.
"""

import torch
from typing import Optional, Tuple, Union
import math

__all__ = [
    "make_tensor",
]

def make_tensor(
    shape: Tuple[int, ...],
    device: Union[str, torch.device],
    dtype: torch.dtype,
    low: Optional[float] = None,
    high: Optional[float] = None,
    requires_grad: bool = False,
    noncontiguous: bool = False,
    exclude_zero: bool = False
) -> torch.Tensor:
    r"""Creates a random tensor with the given :attr:`shape`, :attr:`device`, and :attr:`dtype`.

    If :attr:`low` or :attr:`high` are outside the range of the datatype's representable finite values
    then they are clamped to the lowest or highest representable finite value, respectively. A random
    tensor is then created with values within ``[low, high)`` range. If not passed, following are the
    default values for :attr:`low` and :attr:`high` depending on the :attr:`dtype`.

    +---------------------------+------------+----------+
    | ``dtype``                 | ``low``    | ``high`` |
    +===========================+============+==========+
    | boolean type              | ``0``      | ``2``    |
    +---------------------------+------------+----------+
    | unsigned integral type    | ``0``      | ``10``   |
    +---------------------------+------------+----------+
    | integral types            | ``-9``     | ``10``   |
    +---------------------------+------------+----------+
    | floating types            | ``-9``     | ``9``    |
    +---------------------------+------------+----------+
    | complex types             | ``-9``     | ``9``    |
    +---------------------------+------------+----------+

    If :attr:`low` and :attr:`high` are passed, they are considered only if they are within the
    limit of the :attr:`dtype`. Following are a few conditions that are taken care of:

    - If :attr:`low` and/or :attr:`high` are specified and within dtype limits: the values are taken as they were.
    - If :attr:`low` and/or :attr:`high` are specified but exceed the limits: :attr:`dtype` limits are considered instead.
    - If :attr:`low` is ``-inf`` and/or :attr:`high` is ``inf``: :attr:`dtype` limits are considered instead.

    If a boolean type is requested for the output tensor (through :attr:`dtype`), :attr:`low` and :attr:`high` are
    always set to ``0`` and ``2`` respectively.

    If :attr:`noncontiguous` is ``True``, a non-contiguous tensor with the given :attr:`shape` will be returned unless
    the :attr:`shape` specifies a tensor with a 1 or 0 elements in which case the non-contiguous parameter is ignored
    because it is not possible to create a non-contiguous Tensor with a single element.

    Args:
        shape (Tuple[int, ...]): A sequence of integers defining the shape of the output tensor.
        device (Union[str, torch.device]): The desired device of the returned tensor.
        dtype (torch.dtype): The desired data type of the returned tensor.
        low (Optional[float]): Sets the lower range (inclusive), considered only if they are within the limit of
            :attr:`dtype` passed. Default: see the table above for default values.
        high (Optional[float]): Sets the upper range (exclusive) as specified above for :attr:`low`.
        requires_grad (Optional[bool]): If autograd should record operations on the returned tensor. Default: ``False``.
        noncontiguous (Optional[bool]): If the returned tensor should be made noncontiguous. Default: ``False``.
        exclude_zero (Optional[bool]): If zeros (if any) should be excluded from the returned tensor. Each value matching
            zero (if any) is replaced with a ``tiny`` (smallest positive representable number) value if floating type,
            [``tiny + tiny.j``] if complex type and ``1`` if integral/boolean type. Default: ``False``.

    Raises:
        ValueError: if :attr:`low` is either ``inf`` or ``nan`` and/or :attr:`high` is either ``-inf`` or ``nan``.
        TypeError: if the given :attr:`dtype` isn't supported by this function
            (see the table for default values of :attr:`low` above for the data types supported).

    Examples:
        >>> from torch.testing import make_tensor
        >>> # Creates a float tensor with values in [0, 1)
        >>> make_tensor((3,), device='cpu', dtype=torch.float32, low=0, high=1)
        >>> tensor([0.7682, 0.4189, 0.2718])
        >>> # Creates a bool tensor on CUDA
        >>> make_tensor((2, 2), device='cuda', dtype=torch.bool)
        >>> tensor([[False, False],
                    [False, True]], device='cuda:0')

        >>> # Passing low > high, will raise ValueError
        >>> make_tensor((2,), device='cpu', dtype=torch.float32, low=9, high=8)
        ValueError: make_tensor: low must be weakly less than high!
        >>> # Passing low or high as float('nan') will also raise a ValueError
        >>> make_tensor((2,), device='cpu', dtype=torch.float32, low=9, high=float('nan'))
        ValueError: make_tensor: one of low or high was NaN!
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

    _integral_types = [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]
    _floating_types = [torch.float16, torch.bfloat16, torch.float32, torch.float64]
    _complex_types = [torch.cfloat, torch.cdouble]

    if dtype is torch.bool:
        result = torch.randint(0, 2, shape, device=device, dtype=dtype)
    elif dtype is torch.uint8:
        ranges = (torch.iinfo(dtype).min, torch.iinfo(dtype).max)
        low, high = _modify_low_high(low, high, ranges[0], ranges[1], 0, 9, dtype)
        result = torch.randint(low, high, shape, device=device, dtype=dtype)
    elif dtype in _integral_types:
        ranges = (torch.iinfo(dtype).min, torch.iinfo(dtype).max)
        low, high = _modify_low_high(low, high, ranges[0], ranges[1], -9, 9, dtype)
        result = torch.randint(low, high, shape, device=device, dtype=dtype)
    elif dtype in _floating_types:
        ranges_floats = (torch.finfo(dtype).min, torch.finfo(dtype).max)
        low, high = _modify_low_high(low, high, ranges_floats[0], ranges_floats[1], -9, 9, dtype)
        rand_val = torch.rand(shape, device=device, dtype=dtype)
        result = high * rand_val + low * (1 - rand_val)
    elif dtype in _complex_types:
        float_dtype = torch.float if dtype is torch.cfloat else torch.double
        ranges_floats = (torch.finfo(float_dtype).min, torch.finfo(float_dtype).max)
        low, high = _modify_low_high(low, high, ranges_floats[0], ranges_floats[1], -9, 9, dtype)
        real_rand_val = torch.rand(shape, device=device, dtype=float_dtype)
        imag_rand_val = torch.rand(shape, device=device, dtype=float_dtype)
        real = high * real_rand_val + low * (1 - real_rand_val)
        imag = high * imag_rand_val + low * (1 - imag_rand_val)
        result = torch.complex(real, imag)
    else:
        raise TypeError(f"The requested dtype {dtype} is not supported by torch.testing.make_tensor()."
                        " To request support, file an issue at: https://github.com/pytorch/pytorch/issues")

    if noncontiguous and result.numel() > 1:
        result = torch.repeat_interleave(result, 2, dim=-1)
        result = result[..., ::2]

    if exclude_zero:
        if dtype in _integral_types or dtype is torch.bool:
            replace_with = torch.tensor(1, device=device, dtype=dtype)
        elif dtype in _floating_types:
            replace_with = torch.tensor(torch.finfo(dtype).tiny, device=device, dtype=dtype)
        elif dtype in _complex_types:
            float_dtype = torch.float if dtype is torch.cfloat else torch.double
            float_eps = torch.tensor(torch.finfo(float_dtype).tiny, device=device, dtype=float_dtype)
            replace_with = torch.complex(float_eps, float_eps)
        else:
            raise TypeError(f"The requested dtype {dtype} is not supported by torch.testing.make_tensor()."
                            " To request support, file an issue at: https://github.com/pytorch/pytorch/issues")
        result[result == 0] = replace_with

    if dtype in _floating_types + _complex_types:
        result.requires_grad = requires_grad

    return result
