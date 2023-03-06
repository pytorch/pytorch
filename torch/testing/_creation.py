"""
This module contains tensor creation utilities.
"""

import collections.abc
import math
from typing import cast, List, Optional, Tuple, Union

import torch

# Used by make_tensor for generating complex tensor.
complex_to_corresponding_float_type_map = {
    torch.complex32: torch.float16,
    torch.complex64: torch.float32,
    torch.complex128: torch.float64,
}


def _uniform_random_(t: torch.Tensor, low: float, high: float) -> torch.Tensor:
    # uniform_ requires to-from <= std::numeric_limits<scalar_t>::max()
    # Work around this by scaling the range before and after the PRNG
    if high - low >= torch.finfo(t.dtype).max:
        return t.uniform_(low / 2, high / 2).mul_(2)
    else:
        return t.uniform_(low, high)


def make_tensor(
    *shape: Union[int, torch.Size, List[int], Tuple[int, ...]],
    dtype: torch.dtype,
    device: Union[str, torch.device],
    low: Optional[float] = None,
    high: Optional[float] = None,
    requires_grad: bool = False,
    noncontiguous: bool = False,
    exclude_zero: bool = False,
    memory_format: Optional[torch.memory_format] = None,
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
            ignored if the constructed tensor has fewer than two elements. Mutually exclusive with ``memory_format``.
        exclude_zero (Optional[bool]): If ``True`` then zeros are replaced with the dtype's small positive value
            depending on the :attr:`dtype`. For bool and integer types zero is replaced with one. For floating
            point types it is replaced with the dtype's smallest positive normal number (the "tiny" value of the
            :attr:`dtype`'s :func:`~torch.finfo` object), and for complex types it is replaced with a complex number
            whose real and imaginary parts are both the smallest positive normal number representable by the complex
            type. Default ``False``.
        memory_format (Optional[torch.memory_format]): The memory format of the returned tensor. Mutually exclusive
            with ``noncontiguous``.

    Raises:
        ValueError: If ``requires_grad=True`` is passed for integral `dtype`
        ValueError: If ``low >= high``.
        ValueError: If either :attr:`low` or :attr:`high` is ``nan``.
        ValueError: If both :attr:`noncontiguous` and :attr:`memory_format` are passed.
        TypeError: If :attr:`dtype` isn't supported by this function.

    Examples:
        >>> # xdoctest: +SKIP
        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CUDA)
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

    def modify_low_high(
        low: Optional[float],
        high: Optional[float],
        *,
        lowest: float,
        highest: float,
        default_low: float,
        default_high: float,
    ) -> Tuple[float, float]:
        """
        Modifies (and raises ValueError when appropriate) low and high values given by the user (input_low, input_high)
        if required.
        """

        def clamp(a: float, l: float, h: float) -> float:
            return min(max(a, l), h)

        low = low if low is not None else default_low
        high = high if high is not None else default_high

        # Checks for error cases
        if low != low or high != high:
            raise ValueError(
                f"`low` and `high` cannot be NaN, but got {low=} and {high=}"
            )
        elif low > high:
            raise ValueError(
                f"`low` must be weakly less than `high`, but got {low} >= {high}"
            )

        low = clamp(low, lowest, highest)
        high = clamp(high, lowest, highest)

        if dtype in [
            torch.bool,
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        ]:
            # 1. `low` is ceiled to avoid creating values smaller than `low` and thus outside the specified interval
            # 2. Following the same reasoning as for 1., `high` should be floored. However, the higher bound of
            #    `torch.randint` is exclusive and thus ceiling here as well is fine.
            return math.ceil(low), math.ceil(high)

        return low, high

    if len(shape) == 1 and isinstance(shape[0], collections.abc.Sequence):
        shape = shape[0]  # type: ignore[assignment]
    shape = cast(Tuple[int, ...], tuple(shape))

    if noncontiguous and memory_format is not None:
        raise ValueError(
            f"The parameters `noncontiguous` and `memory_format` are mutually exclusive, "
            f"but got {noncontiguous=} and {memory_format=}"
        )

    _integral_types = [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]
    _floating_types = [torch.float16, torch.bfloat16, torch.float32, torch.float64]
    _complex_types = [torch.complex32, torch.complex64, torch.complex128]
    if requires_grad and dtype not in _floating_types and dtype not in _complex_types:
        raise ValueError(
            f"`requires_grad=True` is not supported for integral dtypes, but got {dtype=}"
        )

    if dtype is torch.bool:
        low, high = cast(
            Tuple[int, int],
            modify_low_high(
                low, high, lowest=0, highest=2, default_low=0, default_high=2
            ),
        )
        result = torch.randint(low, high, shape, device=device, dtype=dtype)
    elif dtype in _integral_types:
        low, high = cast(
            Tuple[int, int],
            modify_low_high(
                low,
                high,
                lowest=torch.iinfo(dtype).min,
                highest=torch.iinfo(dtype).max,
                # This is incorrect for `torch.uint8`, but since we clamp to `lowest`, i.e. 0 for `torch.uint8`,
                # _after_ we use the default value, we don't need to special case it here
                default_low=-9,
                default_high=10,
            ),
        )
        result = torch.randint(low, high, shape, device=device, dtype=dtype)
    elif dtype in _floating_types:
        low, high = modify_low_high(
            low,
            high,
            lowest=torch.finfo(dtype).min,
            highest=torch.finfo(dtype).max,
            default_low=-9,
            default_high=9,
        )
        result = torch.empty(shape, device=device, dtype=dtype)
        _uniform_random_(result, low, high)
    elif dtype in _complex_types:
        float_dtype = complex_to_corresponding_float_type_map[dtype]
        low, high = modify_low_high(
            low,
            high,
            lowest=torch.finfo(float_dtype).min,
            highest=torch.finfo(float_dtype).max,
            default_low=-9,
            default_high=9,
        )
        result = torch.empty(shape, device=device, dtype=dtype)
        _uniform_random_(torch.view_as_real(result), low, high)
    else:
        raise TypeError(
            f"The requested dtype '{dtype}' is not supported by torch.testing.make_tensor()."
            " To request support, file an issue at: https://github.com/pytorch/pytorch/issues"
        )

    if noncontiguous and result.numel() > 1:
        result = torch.repeat_interleave(result, 2, dim=-1)
        result = result[..., ::2]
    elif memory_format is not None:
        result = result.clone(memory_format=memory_format)

    if exclude_zero:
        if dtype in _integral_types or dtype is torch.bool:
            replace_with = torch.tensor(1, device=device, dtype=dtype)
        elif dtype in _floating_types:
            replace_with = torch.tensor(
                torch.finfo(dtype).tiny, device=device, dtype=dtype
            )
        else:  # dtype in _complex_types:
            float_dtype = complex_to_corresponding_float_type_map[dtype]
            float_eps = torch.tensor(
                torch.finfo(float_dtype).tiny, device=device, dtype=float_dtype
            )
            replace_with = torch.complex(float_eps, float_eps)
        result[result == 0] = replace_with

    if dtype in _floating_types + _complex_types:
        result.requires_grad = requires_grad

    return result
