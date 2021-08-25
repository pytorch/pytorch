"""
This package contains tensor creation utilities.
"""

import torch
from typing import Optional, Tuple, Union
import math

__all__ = [
    "make_tensor"
]

def make_tensor(
    size: Tuple[int, ...],
    *,
    device: Union[str, torch.device] = "cpu",
    dtype: torch.dtype = torch.float32,
    low: Optional[float] = None,
    high: Optional[float] = None,
    requires_grad: bool = False,
    noncontiguous: bool = False,
    exclude_zero: bool = False
) -> torch.Tensor:
    r"""Creates a random tensor with the given :attr:`size`.

    The function comes with other optional arguments to allow random tensor generation for the user's needs.

    By default, :attr:`device` is ``cpu`` and :attr:`dtype` is ``torch.float32``, but user can pass these arguments
    as required.

    If :attr:`low` and :attr:`high` are not passed, following default values are considered depending on the given
    :attr:`dtype`:

    - boolean type: `low` = 0, `high` = 2.
    - uint8 type: `low` = 0, `high` = 9.
    - floating and integral types: `low` = -9 and `high` = 9.
    - complex types, for each real and imaginary part: `low` = -9, `high` = 9.

    If :attr:`low` and :attr:`high` are passed, they are considered only if they are within the
    limit of the :attr:`dtype`. Following are a few conditions that are taken care of:

    - If :attr:`low` and/or :attr:`high` are specified and within dtype limits: the values are taken as they were.
    - If :attr:`low` and/or :attr:`high` are specified but exceed the limits: :attr:`dtype` limits ar considered instead.
    - If :attr:`low` is ``-inf`` and/or :attr:`high` is ``inf``: :attr:`dtype` limits are considered instead
    - If :attr:`low` is ``inf`` or ``nan`` and/or :attr:`high` is ``-inf`` or nan: a `ValueError` is raised, since these are invalid values for the range of output tensor.

    If :attr:`noncontiguous` is ``True``, a non-contiguous tensor with the given size will be returned unless the
    size specifies a tensor with a 1 or 0 elements in which case the non-contiguous parameter is ignored because
    it is not possible to create a non-contiguous Tensor with a single element.

    If :attr:`exclude_zero` is ``True`` (default is ``False``), all the values matching to zero in
    the created tensor are replaced with a ``tiny`` (smallest positive representable number) value if floating type,
    [``tiny`` + ``tiny``.j] if complex type and ``1`` if integer/boolean type.

    Examples:
        >>> import torch
        >>> from torch.testing import make_tensor
        >>> # Create a sample integral type (int64) tensor, default range = [-9, 9)
        >>> make_tensor((3,), device='cuda', dtype=torch.int64)
        tensor([-8, -7, -2], device='cuda:0')
        >>> # Create a sample float tensor (double), with range as [0, 1)
        >>> make_tensor((3,), device='cpu', dtype=torch.float64, low=0, high=1)
        tensor([0.2468, 0.9723, 0.6779], dtype=torch.float64)
        >>> # Passing low as -inf and high as inf (ranges will be clamped to dtype limits)
        >>> make_tensor((2,), device='cpu', dtype=torch.float16, low=float('-inf'), high=float('inf'))
        tensor([-38496., -53792.], dtype=torch.float16)

        >>> # Passing low > high, will lead to ValueError
        >>> make_tensor((2,), device='cpu', dtype=torch.float32, low=9, high=8)
        ValueError: make_tensor: low must be weakly less than high!
        >>> # Passing low or high as float('nan') will also lead to a ValueError
        >>> make_tensor((2,), device='cpu', dtype=torch.float32, low=9, high=float('nan'))
        ValueError: make_tensor: one of low or high was NaN!

        >>> # Create a non-contiguous tensor
        >>> make_tensor((2, 2), device='cpu', dtype=torch.float64, noncontiguous=True).is_contiguous()
        False
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
        result = torch.randint(0, 2, size, device=device, dtype=dtype)
    elif dtype is torch.uint8:
        ranges = (torch.iinfo(dtype).min, torch.iinfo(dtype).max)
        low, high = _modify_low_high(low, high, ranges[0], ranges[1], 0, 9, dtype)
        result = torch.randint(low, high, size, device=device, dtype=dtype)
    elif dtype in _integral_types:
        ranges = (torch.iinfo(dtype).min, torch.iinfo(dtype).max)
        low, high = _modify_low_high(low, high, ranges[0], ranges[1], -9, 9, dtype)
        result = torch.randint(low, high, size, device=device, dtype=dtype)
    elif dtype in _floating_types:
        ranges_floats = (torch.finfo(dtype).min, torch.finfo(dtype).max)
        low, high = _modify_low_high(low, high, ranges_floats[0], ranges_floats[1], -9, 9, dtype)
        rand_val = torch.rand(size, device=device, dtype=dtype)
        result = high * rand_val + low * (1 - rand_val)
    else:
        assert dtype in _complex_types
        float_dtype = torch.float if dtype is torch.cfloat else torch.double
        ranges_floats = (torch.finfo(float_dtype).min, torch.finfo(float_dtype).max)
        low, high = _modify_low_high(low, high, ranges_floats[0], ranges_floats[1], -9, 9, dtype)
        real_rand_val = torch.rand(size, device=device, dtype=float_dtype)
        imag_rand_val = torch.rand(size, device=device, dtype=float_dtype)
        real = high * real_rand_val + low * (1 - real_rand_val)
        imag = high * imag_rand_val + low * (1 - imag_rand_val)
        result = torch.complex(real, imag)

    if noncontiguous and result.numel() > 1:
        result = torch.repeat_interleave(result, 2, dim=-1)
        result = result[..., ::2]

    if exclude_zero:
        if dtype in _integral_types or dtype is torch.bool:
            replace_with = torch.tensor(1, device=device, dtype=dtype)
        elif dtype in _floating_types:
            replace_with = torch.tensor(torch.finfo(dtype).tiny, device=device, dtype=dtype)
        else:
            assert dtype in _complex_types
            float_dtype = torch.float if dtype is torch.cfloat else torch.double
            float_eps = torch.tensor(torch.finfo(float_dtype).tiny, device=device, dtype=float_dtype)
            replace_with = torch.complex(float_eps, float_eps)
        result[result == 0] = replace_with

    if dtype in _floating_types or\
       dtype in _complex_types:
        result.requires_grad = requires_grad

    return result
