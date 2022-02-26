"""This module exists since the `torch.testing` exposed a lot of stuff that shouldn't have been public. Although this
was never documented anywhere, some other internal FB projects as well as downstream OSS projects might use this. Thus,
we don't internalize without warning, but still go through a deprecation cycle.
"""

import functools
import random
import warnings
from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch

from . import _legacy


__all__ = [
    "rand",
    "randn",
    "assert_allclose",
    "get_all_device_types",
    "make_non_contiguous",
]


def warn_deprecated(instructions: Union[str, Callable[[str, Tuple[Any, ...], Dict[str, Any], Any], str]]) -> Callable:
    def outer_wrapper(fn: Callable) -> Callable:
        name = fn.__name__
        head = f"torch.testing.{name}() is deprecated since 1.12 and will be removed in 1.14. "

        @functools.wraps(fn)
        def inner_wrapper(*args: Any, **kwargs: Any) -> Any:
            return_value = fn(*args, **kwargs)
            tail = instructions(name, args, kwargs, return_value) if callable(instructions) else instructions
            msg = (head + tail).strip()
            warnings.warn(msg, FutureWarning)
            return return_value

        return inner_wrapper

    return outer_wrapper


rand = warn_deprecated("Use torch.rand() instead.")(torch.rand)
randn = warn_deprecated("Use torch.randn() instead.")(torch.randn)


_DTYPE_PRECISIONS = {
    torch.float16: (1e-3, 1e-3),
    torch.float32: (1e-4, 1e-5),
    torch.float64: (1e-5, 1e-8),
}


def _get_default_rtol_and_atol(actual: torch.Tensor, expected: torch.Tensor) -> Tuple[float, float]:
    actual_rtol, actual_atol = _DTYPE_PRECISIONS.get(actual.dtype, (0.0, 0.0))
    expected_rtol, expected_atol = _DTYPE_PRECISIONS.get(expected.dtype, (0.0, 0.0))
    return max(actual_rtol, expected_rtol), max(actual_atol, expected_atol)


@warn_deprecated(
    "Use torch.testing.assert_close() instead. "
    "For detailed upgrade instructions see https://github.com/pytorch/pytorch/issues/61844."
)
def assert_allclose(
    actual: Any,
    expected: Any,
    rtol: Optional[float] = None,
    atol: Optional[float] = None,
    equal_nan: bool = True,
    msg: str = "",
) -> None:
    if not isinstance(actual, torch.Tensor):
        actual = torch.tensor(actual)
    if not isinstance(expected, torch.Tensor):
        expected = torch.tensor(expected, dtype=actual.dtype)

    if rtol is None and atol is None:
        rtol, atol = _get_default_rtol_and_atol(actual, expected)

    torch.testing.assert_close(
        actual,
        expected,
        rtol=rtol,
        atol=atol,
        equal_nan=equal_nan,
        check_device=True,
        check_dtype=False,
        check_stride=False,
        msg=msg or None,
    )


getter_instructions = (
    lambda name, args, kwargs, return_value: f"This call can be replaced with {return_value}."  # noqa: E731
)

# Deprecate and expose all dtype getters
for name in _legacy.__all_dtype_getters__:
    fn = getattr(_legacy, name)
    globals()[name] = warn_deprecated(getter_instructions)(fn)
    __all__.append(name)

get_all_device_types = warn_deprecated(getter_instructions)(_legacy.get_all_device_types)


@warn_deprecated(
    "Depending on the use case there a different replacement options:\n\n"
    "- If you are using `make_non_contiguous` in combination with a creation function to create a noncontiguous tensor "
    "with random values, use `torch.testing.make_tensor(..., noncontiguous=True)` instead.\n"
    "- If you are using `make_non_contiguous` with a specific tensor, you can replace this call with "
    "`torch.repeat_interleave(input, 2, dim=-1)[..., ::2]`.\n"
    "- If you are using `make_non_contiguous` in the PyTorch test suite, use "
    "`torch.testing._internal.common_utils.noncontiguous_like` instead."
)
def make_non_contiguous(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.numel() <= 1:  # can't make non-contiguous
        return tensor.clone()
    osize = list(tensor.size())

    # randomly inflate a few dimensions in osize
    for _ in range(2):
        dim = random.randint(0, len(osize) - 1)
        add = random.randint(4, 15)
        osize[dim] = osize[dim] + add

    # narrow doesn't make a non-contiguous tensor if we only narrow the 0-th dimension,
    # (which will always happen with a 1-dimensional tensor), so let's make a new
    # right-most dimension and cut it off

    input = tensor.new(torch.Size(osize + [random.randint(2, 3)]))
    input = input.select(len(input.size()) - 1, random.randint(0, 1))
    # now extract the input of correct size from 'input'
    for i in range(len(osize)):
        if input.size(i) != tensor.size(i):
            bounds = random.randint(1, input.size(i) - tensor.size(i))
            input = input.narrow(i, bounds, tensor.size(i))

    input.copy_(tensor)

    # Use .data here to hide the view relation between input and other temporary Tensors
    return input.data
