"""This module exists since the `torch.testing` exposed a lot of stuff that shouldn't have been public. Although this
was never documented anywhere, some other internal FB projects as well as downstream OSS projects might use this. Thus,
we don't internalize without warning, but still go through a deprecation cycle.
"""

import functools
import warnings
from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch


__all__ = ["assert_allclose"]


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
