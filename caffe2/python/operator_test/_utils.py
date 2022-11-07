"""
This file only exists since `torch.testing.assert_allclose` was removed, but used extensively throughout the tests in
this package. The replacement `torch.testing.assert_close` doesn't support one feature that is needed here: comparison
between numpy arrays and torch tensors. See https://github.com/pytorch/pytorch/issues/61844 for the reasoning why this
was removed.
"""

import torch
from typing import Tuple, Any, Optional

_DTYPE_PRECISIONS = {
    torch.float16: (1e-3, 1e-3),
    torch.float32: (1e-4, 1e-5),
    torch.float64: (1e-5, 1e-8),
}


def _get_default_rtol_and_atol(actual: torch.Tensor, expected: torch.Tensor) -> Tuple[float, float]:
    actual_rtol, actual_atol = _DTYPE_PRECISIONS.get(actual.dtype, (0.0, 0.0))
    expected_rtol, expected_atol = _DTYPE_PRECISIONS.get(expected.dtype, (0.0, 0.0))
    return max(actual_rtol, expected_rtol), max(actual_atol, expected_atol)


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