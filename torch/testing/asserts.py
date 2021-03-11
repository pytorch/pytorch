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


def _get_default_rtol_and_atol(tensor1: torch.Tensor, tensor2: torch.Tensor) -> Tuple[float, float]:
    return _TestCase()._getDefaultRtolAndAtol(tensor1.dtype, tensor2.dtype)


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


def _compare_tensors_meta(tensor1: torch.Tensor, tensor2: torch.Tensor, **attributes: bool) -> None:
    if not (isinstance(tensor1, torch.Tensor) and isinstance(tensor2, torch.Tensor)):
        raise AssertionError(f"Both inputs have to be tensors, but got {type(tensor1)} and {type(tensor2)} instead.")

    for attr in DEFAULT_CHECKED_ATTRIBUTES:
        attributes.setdefault(attr, True)

    for attr, check in attributes.items():
        if not check:
            continue

        try:
            val1 = getattr(tensor1, attr)
            val2 = getattr(tensor2, attr)
            if val1 != val2:
                raise AssertionError(f"The values for attribute '{attr}' do not match: {val1} != {val2}.")
        except AttributeError as error:
            raise UsageError(f"At least one of the inputs does not have an attribute '{attr}'.") from error


def _compare_tensors(
    tensor1: torch.Tensor,
    tensor2: torch.Tensor,
    asserter: Callable[[torch.Tensor, torch.Tensor], None],
    **strict_attributes: bool,
) -> None:
    if any(tensor.dtype in (torch.complex32, torch.complex64, torch.complex128) for tensor in (tensor1, tensor2)):
        raise UsageError("Comparison for complex tensors is not supported yet.")
    if any(tensor.is_quantized for tensor in (tensor1, tensor2)):
        raise UsageError("Comparison for quantized tensors is not supported yet.")
    if any(tensor.is_sparse for tensor in (tensor1, tensor2)):
        raise UsageError("Comparison for sparse tensors is not supported yet.")

    _compare_tensors_meta(tensor1, tensor2, **strict_attributes)

    if tensor1.device != tensor2.device:
        tensor1 = tensor1.cpu()
        tensor2 = tensor2.cpu()

    if tensor1.dtype != tensor2.dtype:
        dtype = _get_comparison_dtype(tensor1, tensor2)
        tensor1 = tensor1.to(dtype)
        tensor2 = tensor2.to(dtype)

    asserter(tensor1, tensor2)


def _tensors_equal_asserter(tensor1: torch.Tensor, tensor2: torch.Tensor):
    if not torch.all(torch.eq(tensor1, tensor2)):
        raise AssertionError("ADDME")


@_hide_internal_traceback
def assert_tensors_equal(
    tensor1: torch.Tensor, tensor2: torch.Tensor, *, warn_floating_point: bool = True, **strict_attributes: bool
) -> None:
    """Assert that two tensors are equal.

    Before the values are compared, the ``shape`` and the ``dtype`` of the tensors are checked for equality. You can
    change this behavior with the ``strict_attributes`` parameter.

    Depending on the strictness some additional steps may be performed before the value comparison:

    - If the tensors don't live in the same device memory they are moved to CPU memory.
    - If the tensors don't have the same ``dtype``, they are converted to the one determined by
        :func:`torch.promote_types`.

    Args:
        tensor1 (torch.Tensor): First tensor.
        tensor2 (torch.Tensor): Second tensor.
        warn_floating_point (bool): Emit a warning if at least one of the inputs is a floating point tensor.
        **strict_attributes (bool): Set strictness for attribute equality checking. If omitted, this defaults to
            ``shape=True, dtype=True``.

    Raises:
        AssertionError: If the specified attributes do not match.
        AssertionError: If the tensors are not equal.
    """
    is_floating_point = tensor1.dtype.is_floating_point or tensor2.dtype.is_floating_point
    if is_floating_point and warn_floating_point:
        warnings.warn(
            "Due to the limitations of floating point arithmetic, comparing floating-point tensors for equality is not "
            "recommended. Use the respective 'allclose' variant instead.",
            UsageWarning,
        )

    _compare_tensors(tensor1, tensor2, _tensors_equal_asserter, **strict_attributes)


def _tensors_allclose_asserter(
    tensor1: torch.Tensor, tensor2: torch.Tensor, rtol: Optional[float], atol: Optional[float]
) -> None:
    default_rtol, default_atol = _get_default_rtol_and_atol(tensor1, tensor2)
    rtol = rtol if rtol is not None else default_rtol
    atol = atol if atol is not None else default_atol

    if not torch.allclose(tensor1, tensor2, rtol=rtol, atol=atol):
        raise AssertionError("ADDME")


@_hide_internal_traceback
def assert_tensors_allclose(
    tensor1: torch.Tensor,
    tensor2: torch.Tensor,
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
        tensor1 (torch.Tensor): First tensor.
        tensor2 (torch.Tensor): Second tensor.
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
    _compare_tensors(
        tensor1, tensor2, functools.partial(_tensors_allclose_asserter, rtol=rtol, atol=atol), **strict_attributes
    )
