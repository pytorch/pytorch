import functools
import inspect
import warnings
from typing import Callable, Optional, Tuple

import torch
from torch.testing._internal.common_utils import get_comparison_dtype as _get_comparison_dtype, TestCase as _TestCase

__all__ = ["assert_tensors_equal", "assert_tensors_allclose"]

try:
    import pytest

    UsageError = pytest.UsageError
except ImportError:

    class UsageError(Exception):
        pass


class UsageWarning(Warning):
    pass


def _get_default_rtol_and_atol(a: torch.Tensor, b: torch.Tensor) -> Tuple[float, float]:
    return _TestCase()._getDefaultRtolAndAtol(a.dtype, b.dtype)


def _hide_internal_traceback(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except AssertionError:
            for frame_info in inspect.trace():
                frame_info.frame.f_locals.setdefault("__tracebackhide__", True)
            raise

    return wrapper


# These attributes will be checked by default in every call to assert_tensors_(equal|allclose) and
# assert_(equal|allclose) if the inputs are tensors.
DEFAULT_CHECKED_ATTRIBUTES = (
    "shape",
    "dtype",
)


def _compare_tensors_meta(a: torch.Tensor, b: torch.Tensor, **attributes: bool) -> None:
    if not (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)):
        raise AssertionError(f"Both inputs have to be tensors, but got {type(a)} and {type(b)} instead.")

    for attr in DEFAULT_CHECKED_ATTRIBUTES:
        attributes.setdefault(attr, True)

    for attr, check in attributes.items():
        if not check:
            continue

        try:
            val_a = getattr(a, attr)
            val_b = getattr(b, attr)
            if val_a != val_b:
                raise AssertionError(f"The values for attribute '{attr}' do not match: {val_a} != {val_b}.")
        except AttributeError as error:
            raise UsageError(f"At least one of the inputs does not have an attribute '{attr}'.") from error


def _compare_tensors(
    a: torch.Tensor,
    b: torch.Tensor,
    asserter: Callable[[torch.Tensor, torch.Tensor], None],
    **strict_attributes: bool,
) -> None:
    if any(tensor.dtype in (torch.complex32, torch.complex64, torch.complex128) for tensor in (a, b)):
        raise UsageError("Comparison for complex tensors is not supported yet.")
    if any(tensor.is_quantized for tensor in (a, b)):
        raise UsageError("Comparison for quantized tensors is not supported yet.")
    if any(tensor.is_sparse for tensor in (a, b)):
        raise UsageError("Comparison for sparse tensors is not supported yet.")

    _compare_tensors_meta(a, b, **strict_attributes)

    if a.device != b.device:
        a = a.cpu()
        b = b.cpu()

    if a.dtype != b.dtype:
        dtype = _get_comparison_dtype(a, b)
        a = a.to(dtype)
        b = b.to(dtype)

    asserter(a, b)


def _tensors_equal_asserter(a: torch.Tensor, b: torch.Tensor):
    if not torch.all(torch.eq(a, b)):
        raise AssertionError("ADDME")


@_hide_internal_traceback
def assert_tensors_equal(
    a: torch.Tensor,
    b: torch.Tensor,
    /,
    *,
    warn_floating_point: bool = True,
    **strict_attributes: bool,
) -> None:
    """Assert that two tensors are equal.

    Before the values are compared, the ``shape`` and the ``dtype`` of the tensors are checked for equality. You can
    change this behavior with the ``strict_attributes`` parameter.

    Depending on the strictness some additional steps may be performed before the value comparison:

    - If the tensors don't live in the same device memory they are moved to CPU memory.
    - If the tensors don't have the same ``dtype``, they are converted to the one determined by
        :func:`torch.promote_types`.

    Args:
        a (torch.Tensor): First tensor.
        b (torch.Tensor): Second tensor.
        warn_floating_point (bool): Emit a warning if at least one of the inputs is a floating point tensor.
        **strict_attributes (bool): Set strictness for attribute equality checking. If omitted, this defaults to
            ``shape=True, dtype=True``.

    Raises:
        AssertionError: If the specified attributes do not match.
        AssertionError: If the tensors are not equal.
    """
    is_floating_point = a.dtype.is_floating_point or b.dtype.is_floating_point
    if is_floating_point and warn_floating_point:
        warnings.warn(
            "Due to the limitations of floating point arithmetic, comparing floating-point tensors for equality is not "
            "recommended. Use the respective 'allclose' variant instead.",
            UsageWarning,
        )

    _compare_tensors(a, b, _tensors_equal_asserter, **strict_attributes)


def _tensors_allclose_asserter(a: torch.Tensor, b: torch.Tensor, rtol: Optional[float], atol: Optional[float]) -> None:
    default_rtol, default_atol = _get_default_rtol_and_atol(a, b)
    rtol = rtol if rtol is not None else default_rtol
    atol = atol if atol is not None else default_atol

    if not torch.allclose(a, b, rtol=rtol, atol=atol):
        raise AssertionError("ADDME")


@_hide_internal_traceback
def assert_tensors_allclose(
    a: torch.Tensor,
    b: torch.Tensor,
    /,
    *,
    rtol: Optional[float] = None,
    atol: Optional[float] = None,
    **strict_attributes: bool,
) -> None:
    """Assert that two tensors are equal up to a desired precision.

    Before the values are compared, the ``shape`` and the ``dtype`` of the tensors are checked for equality. You can
    change this behavior with the ``strict_attributes`` parameter.

    Depending on the strictness some additional steps may be performed before the value comparison:

    - If the tensors don't live in the same device memory they are moved to CPU memory.
    - If the tensors don't have the same ``dtype``, they are converted to the one determined by
        :func:`torch.promote_types`.

    Args:
        a (torch.Tensor): First tensor.
        b (torch.Tensor): Second tensor.
        rtol (Optional[float]): Relative tolerance. If omitted, a default value is determined by the ``dtype`` of the
            tensors.
        atol (Optional[float]): Absolute tolerance. If omitted, a default value is determined by the ``dtype`` of the
            tensors.
        **strict_attributes (bool): Set strictness for attribute equality checking. If omitted, this defaults to
            ``shape=True, dtype=True``.

    Raises:
        AssertionError: If the specified attributes do not match.
        AssertionError: If the tensors are not equal up to the desired precision.
    """
    _compare_tensors(a, b, functools.partial(_tensors_allclose_asserter, rtol=rtol, atol=atol), **strict_attributes)
