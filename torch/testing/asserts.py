import sys
from collections import namedtuple
from typing import Any, Optional, Tuple

import torch
from torch.testing import _unravel_index
from torch.testing._internal.common_utils import get_comparison_dtype as _get_comparison_dtype, TestCase as _TestCase

__all__ = ["assert_tensors_equal", "assert_tensors_allclose"]


# The UsageError should be raised in case the test function is not used correctly. With this the user is able to
# differentiate between a test failure (there is a bug in the tested code) and a test error (there is a bug in the
# test). If pytest is the test runner, we use the built-in UsageError instead our custom one.
try:
    # The module 'pytest' will be imported if the 'pytest' runner is used. This will only give false-positives in case
    # a previously imported module already directly or indirectly imported 'pytest', but the test is run by another
    # runner such as 'unittest'.
    pytest = sys.modules["pytest"]
    UsageError = pytest.UsageError  # type: ignore[attr-defined]
except KeyError:

    class UsageError(Exception):  # type: ignore[no-redef]
        pass


def _get_default_rtol_and_atol(a: torch.Tensor, b: torch.Tensor) -> Tuple[float, float]:
    dtype = a.dtype if a.dtype == b.dtype else _get_comparison_dtype(a, b)
    return _TestCase.dtype_precisions.get(dtype, (0.0, 0.0))


def _assert_are_tensors(a: Any, b: Any) -> None:
    """Asserts that both inputs are tensors.

    Args:
        a (Any): First input.
        b (Any): Second input.

    Raises:
        AssertionError: If :attr:`a` or :attr:`b` is not a :class:`~torch.Tensor`.
    """
    # Hide this function from the pytest traceback
    __tracebackhide__ = True

    if not (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)):
        raise AssertionError(f"Both inputs have to be tensors, but got {type(a)} and {type(b)} instead.")


def _check_supported(a: torch.Tensor, b: torch.Tensor) -> None:
    """Checks if the tensors are supported by the current infrastructure.

    All checks are temporary and will be relaxed in the future.

    Args:
        a (torch.Tensor): First tensor.
        b (torch.Tensor): Second tensor.

    Raises:
        UsageError: If :attr:`a` or :attr:`b` is complex, quantized, or sparse.
    """
    # Hide this function from the pytest traceback
    __tracebackhide__ = True

    if any(t.dtype in (torch.complex32, torch.complex64, torch.complex128) for t in (a, b)):
        raise UsageError("Comparison for complex tensors is not supported yet.")
    if any(t.is_quantized for t in (a, b)):
        raise UsageError("Comparison for quantized tensors is not supported yet.")
    if any(t.is_sparse for t in (a, b)):
        raise UsageError("Comparison for sparse tensors is not supported yet.")


def _assert_attributes_equal(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    check_device: bool = True,
    check_dtype: bool = True,
    check_stride: bool = True,
) -> None:
    """Asserts that attributes of two tensors match.

    Always checks the :attr:`~torch.Tensor.shape`. Other checks are optional and can be disabled.

    Args:
        a (torch.Tensor): First tensor.
        b (torch.Tensor): Second tensor.
        check_device (bool): If ``True`` (default), asserts that both :attr:`a` and :attr:`b` live in the same
            :attr:`~torch.Tensor.device` memory.
        check_dtype (bool): If ``True`` (default), asserts that both :attr:`a` and :attr:`b` have the same
            :attr:`~torch.Tensor.dtype`.
        check_stride (bool): If ``True`` (default), asserts that both :attr:`a` and :attr:`b` have the same
            :meth:`~torch.Tensor.stride`.

    Raises:
        AssertionError: If :attr:`a` and :attr:`b` do not have the same :attr:`~torch.Tensor.shape`.
        AssertionError: If :attr:`check_device`, but :attr:`a` and :attr:`b` do not live in the same
            :attr:`~torch.Tensor.device` memory.
        AssertionError: If :attr:`check_dtype`, but :attr:`a` and :attr:`b` do not have the same
            :attr:`~torch.Tensor.dtype`.
        AssertionError: If :attr:`check_stride`, but :attr:`a` and :attr:`b` do not have the same
            :meth:`~torch.Tensor.stride`.
    """
    # Hide this function from the pytest traceback
    __tracebackhide__ = True

    msg_fmtstr = "The values for attribute '{}' do not match: {} != {}."

    if a.shape != b.shape:
        raise AssertionError(msg_fmtstr.format("shape", a.shape, b.shape))

    if check_device and a.device != b.device:
        raise AssertionError(msg_fmtstr.format("device", a.device, b.device))

    if check_dtype and a.dtype != b.dtype:
        raise AssertionError(msg_fmtstr.format("dtype", a.dtype, b.dtype))

    if check_stride and a.stride() != b.stride():
        raise AssertionError(msg_fmtstr.format("stride()", a.stride(), b.stride()))


def _equalize_attributes(a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Equalizes some attributes of two tensors for value comparison.

    If :attr:`a` and :attr:`b`
    - do not live in the same memory :attr:`~torch.Tensor.device`, they are moved CPU memory, and
    - do not have the same :attr:`~torch.Tensor.dtype`, they are copied to the :class:`~torch.dtype` returned by
        :func:`torch.promote_types`.

    Args:
        a (torch.Tensor): First tensor.
        b (torch.Tensor): Second tensor.

    Returns:
        Tuple(torch.Tensor, torch.Tensor): Equalized tensors.
    """
    if a.device != b.device:
        a = a.cpu()
        b = b.cpu()

    if a.dtype != b.dtype:
        dtype = _get_comparison_dtype(a, b)
        a = a.to(dtype)
        b = b.to(dtype)

    return a, b


_Trace = namedtuple("Trace", ("total", "abs", "rel", "idx", "diff", "a", "b"))


def _trace_mismatches(a: torch.Tensor, b: torch.Tensor, mismatches: torch.Tensor) -> _Trace:
    """Traces mismatches and returns the found information.

    The returned named tuple has the following fields:
    - total (int): Total number of values in :attr:`a` and :attr:`b`.
    - abs (int): Absolute number of mismatches.
    - rel (float): Relative number of mismatches.
    - idx (Union[int, Tuple[int, ...]]): Index of greatest absolute difference.
    - diff (Union[int, float]): Greatest absolute difference.
    - a (Union[int, float]): Value of :attr:`a` at the greatest absolute difference.
    - b (Union[int, float]): Value of :attr:`a` at the greatest absolute difference.

    For ``diff``, ``a``, and ``b`` the returned type depends on the :attr:`~torch.Tensor.dtype` of :attr:`a` and
    :attr:`b`.

    Args:
        a (torch.Tensor): First tensor.
        b (torch.Tensor): Second tensor.
        mismatches (torch.Tensor): Boolean mask of the same shape as :attr:`a` and :attr:`b` that indicates the
            location of mismatches.
    """
    total = mismatches.numel()
    abs = torch.sum(mismatches).item()
    rel = abs / total

    dtype = torch.float64 if a.dtype.is_floating_point else torch.int64
    a_flat = a.flatten().to(dtype)
    b_flat = b.flatten().to(dtype)

    abs_diff_flat = torch.abs(a_flat - b_flat)
    idx_flat = torch.argmax(abs_diff_flat)

    return _Trace(
        total=total,
        abs=abs,
        rel=rel,
        idx=_unravel_index(idx_flat, a.shape),
        diff=abs_diff_flat[idx_flat].item(),
        a=a_flat[idx_flat].item(),
        b=b_flat[idx_flat].item(),
    )


def _assert_values_equal(a: torch.Tensor, b: torch.Tensor):
    """Asserts that the values of two tensors are bitwise equal.

    Args:
        a (torch.Tensor): First tensor.
        b (torch.Tensor): Second tensor.

    Raises:
         AssertionError: If the values of :attr:`a` and :attr:`b` are not bitwise equal.
    """
    # Hide this function from the pytest traceback
    __tracebackhide__ = True

    mismatches = torch.ne(a, b)
    if not torch.any(mismatches):
        return

    trace = _trace_mismatches(a, b, mismatches)
    msg = (
        f"Found {trace.abs} different element(s) out of {trace.total} ({trace.rel:.1%}). "
        f"The greatest difference of {trace.diff} ({trace.a} vs. {trace.b}) occurred at index {trace.idx}"
    )
    raise AssertionError(msg)


def _assert_values_allclose(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    rtol,
    atol,
) -> None:
    """Asserts that the values of two tensors are close up to a desired tolerance.

    Args:
        a (torch.Tensor): First tensor.
        b (torch.Tensor): Second tensor.
        rtol (float): Relative tolerance.
        atol (float): Absolute tolerance.

    Raises:
         AssertionError: If the values of :attr:`a` and :attr:`b` are close up to a desired tolerance.
    """
    # Hide this function from the pytest traceback
    __tracebackhide__ = True

    mismatches = ~torch.isclose(a, b, rtol=rtol, atol=atol)
    if not torch.any(mismatches):
        return

    trace = _trace_mismatches(a, b, mismatches)
    msg = (
        f"With rtol={rtol} and atol={atol}, "
        f"found {trace.abs} different element(s) out of {trace.total} ({trace.rel:.1%}). "
        f"The greatest difference of {trace.diff} ({trace.a} vs. {trace.b}) occurred at index {trace.idx}"
    )
    raise AssertionError(msg)


def assert_tensors_equal(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    check_device: bool = True,
    check_dtype: bool = True,
    check_stride: bool = True,
) -> None:
    """Asserts that the values of two tensors are bitwise equal.

    Optionally, checks that some attributes of both tensors are equal.

    Args:
        a (torch.Tensor): First tensor.
        b (torch.Tensor): Second tensor.
        check_device (bool): If ``True`` (default), asserts that both :attr:`a` and :attr:`b` live in the same
            :attr:`~torch.Tensor.device` memory. If this check is disabled **and** :attr:`a` and :attr:`b` do not live
            in the same memory :attr:`~torch.Tensor.device`, they are moved CPU memory before their values are
            compared.
        check_dtype (bool): If ``True`` (default), asserts that both :attr:`a` and :attr:`b` have the same
            :attr:`~torch.Tensor.dtype`. If this check is disabled **and** :attr:`a` and :attr:`b` do not have the same
            :attr:`~torch.Tensor.dtype`, they are copied to the :class:`~torch.dtype` returned by
            :func:`torch.promote_types` before their values are compared.
        check_stride (bool): If ``True`` (default), asserts that both :attr:`a` and :attr:`b` have the same stride.

    Raises:
        UsageError: If :attr:`a` or :attr:`b` is complex, quantized, or sparse. This is a temporary restriction and
            will be relaxed in the future.
        AssertionError: If :attr:`a` and :attr:`b` do not have the same :attr:`~torch.Tensor.shape`.
        AssertionError: If :attr:`check_device`, but :attr:`a` and :attr:`b` do not live in the same
            :attr:`~torch.Tensor.device` memory.
        AssertionError: If :attr:`check_dtype`, but :attr:`a` and :attr:`b` do not have the same
            :attr:`~torch.Tensor.dtype`.
        AssertionError: If :attr:`check_stride`, but :attr:`a` and :attr:`b` do not have the same stride.
        AssertionError: If the values of :attr:`a` and :attr:`b` are not bitwise equal.

    .. seealso::

        To assert that the values in two tensors are are close but are not required to be bitwise equal, use
        :func:`assert_tensors_allclose` instead.
    """
    _assert_are_tensors(a, b)
    _check_supported(a, b)

    _assert_attributes_equal(a, b, check_device=check_device, check_dtype=check_dtype, check_stride=check_stride)
    a, b = _equalize_attributes(a, b)

    _assert_values_equal(a, b)


def assert_tensors_allclose(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    rtol: Optional[float] = None,
    atol: Optional[float] = None,
    check_device: bool = True,
    check_dtype: bool = True,
    check_stride: bool = True,
) -> None:
    """Asserts that the values of two tensors are close up to a desired tolerance.

    If both tolerances, :attr:`rtol` and :attr:`rtol`, are ``0``, asserts that :attr:`a` and :attr:`b` are bitwise
    equal. Optionally, checks that some attributes of both tensors are equal.

    Args:
        a (torch.Tensor): First tensor.
        b (torch.Tensor): Second tensor.
        rtol (Optional[float]): Relative tolerance. If specified :attr:`atol` must also be specified. If omitted,
            default values based on the :attr:`~torch.Tensor.dtype` are selected with the below table.
        atol (Optional[float]): Absolute tolerance. If specified :attr:`rtol` must also be specified. If omitted,
            default values based on the :attr:`~torch.Tensor.dtype` are selected with the below table.
        check_device (bool): If ``True`` (default), asserts that both :attr:`a` and :attr:`b` live in the same
            :attr:`~torch.Tensor.device` memory. If this check is disabled **and** :attr:`a` and :attr:`b` do not live
            in the same memory :attr:`~torch.Tensor.device`, they are moved CPU memory before their values are
            compared.
        check_dtype (bool): If ``True`` (default), asserts that both :attr:`a` and :attr:`b` have the same
            :attr:`~torch.Tensor.dtype`. If this check is disabled **and** :attr:`a` and :attr:`b` do not have the same
            :attr:`~torch.Tensor.dtype`, they are copied to the :class:`~torch.dtype` returned by
            :func:`torch.promote_types` before their values are compared.
        check_stride (bool): If ``True`` (default), asserts that both :attr:`a` and :attr:`b` have the same stride.

    Raises:
        UsageError: If :attr:`a` or :attr:`b` is complex, quantized, or sparse. This is a temporary restriction and
            will be relaxed in the future.
        AssertionError: If :attr:`a` and :attr:`b` do not have the same :attr:`~torch.Tensor.shape`.
        AssertionError: If :attr:`check_device`, but :attr:`a` and :attr:`b` do not live in the same
            :attr:`~torch.Tensor.device` memory.
        AssertionError: If :attr:`check_dtype`, but :attr:`a` and :attr:`b` do not have the same
            :attr:`~torch.Tensor.dtype`.
        AssertionError: If :attr:`check_stride`, but :attr:`a` and :attr:`b` do not have the same stride.
        AssertionError: If the values of :attr:`a` and :attr:`b` are close up to a desired tolerance.



    The following table displays the default ``rtol`` and ``atol`` for floating point :attr:`~torch.Tensor.dtype`'s.
    For integer :attr:`~torch.Tensor.dtype`'s, ``rtol = atol = 0.0`` is used.

    +===========================+============+==========+
    | :class:`~torch.dtype`     | ``rtol``   | ``atol`` |
    +===========================+============+==========+
    | :attr:`~torch.float16`    | ``1e-3``   | ``1e-5`` |
    +---------------------------+------------+----------+
    | :attr:`~torch.bfloat16`   | ``1.6e-2`` | ``1e-5`` |
    +---------------------------+------------+----------+
    | :attr:`~torch.float32`    | ``1.3e-6`` | ``1e-5`` |
    +---------------------------+------------+----------+
    | :attr:`~torch.float64`    | ``1e-7``   | ``1e-7`` |
    +---------------------------+------------+----------+
    | :attr:`~torch.complex32`  | ``1e-3``   | ``1e-5`` |
    +---------------------------+------------+----------+
    | :attr:`~torch.complex64`  | ``1.3e-6`` | ``1e-5`` |
    +---------------------------+------------+----------+
    | :attr:`~torch.complex128` | ``1e-7``   | ``1e-7`` |
    +---------------------------+------------+----------+

    .. seealso::

        To assert that the values in two tensors are bitwise equal, use :func:`assert_tensors_equal` instead.
    """
    _assert_are_tensors(a, b)
    _check_supported(a, b)

    if (rtol is None) ^ (atol is None):
        # We require both tolerance to be omitted or specified, because specifying only one might lead to surprising
        # results. Imagine setting atol=0.0 and the tensors still match because rtol>0.0.
        raise UsageError(
            f"Both 'rtol' and 'atol' must be omitted or specified, " f"but got rtol={rtol} and atol={atol} instead."
        )
    elif rtol is None:
        rtol, atol = _get_default_rtol_and_atol(a, b)

    _assert_attributes_equal(a, b, check_device=check_device, check_dtype=check_dtype, check_stride=check_stride)
    a, b = _equalize_attributes(a, b)

    if (rtol == 0.0) and (atol == 0.0):
        _assert_values_equal(a, b)
    else:
        _assert_values_allclose(a, b, rtol=rtol, atol=atol)
