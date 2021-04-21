import sys
from collections import namedtuple
from typing import Any, Optional, Tuple, Type

import torch

from ._core import _unravel_index

__all__ = ["assert_tensors_equal", "assert_tensors_close"]


# The UsageError should be raised in case the test function is not used correctly. With this the user is able to
# differentiate between a test failure (there is a bug in the tested code) and a test error (there is a bug in the
# test). If pytest is the test runner, we use the built-in UsageError instead our custom one.

try:
    # The module 'pytest' will be imported if the 'pytest' runner is used. This will only give false-positives in case
    # a previously imported module already directly or indirectly imported 'pytest', but the test is run by another
    # runner such as 'unittest'.
    # 'mypy' is not able to handle this within a type annotation
    # (see https://mypy.readthedocs.io/en/latest/common_issues.html#variables-vs-type-aliases for details). In case
    # 'UsageError' is used in an annotation, add a 'type: ignore[valid-type]' comment.
    UsageError: Type[Exception] = sys.modules["pytest"].UsageError  # type: ignore[attr-defined]
except (KeyError, AttributeError):

    class UsageError(Exception):  # type: ignore[no-redef]
        pass


# This is copy-pasted from torch.testing._internal.common_utils.TestCase.dtype_precisions. With this we avoid a
# dependency on torch.testing._internal at import. See
# https://github.com/pytorch/pytorch/pull/54769#issuecomment-813174256 for details.
# {dtype: (rtol, atol)}
_DTYPE_PRECISIONS = {
    torch.float16: (0.001, 1e-5),
    torch.bfloat16: (0.016, 1e-5),
    torch.float32: (1.3e-6, 1e-5),
    torch.float64: (1e-7, 1e-7),
    torch.complex32: (0.001, 1e-5),
    torch.complex64: (1.3e-6, 1e-5),
    torch.complex128: (1e-7, 1e-7),
}


def _get_default_rtol_and_atol(actual: torch.Tensor, expected: torch.Tensor) -> Tuple[float, float]:
    dtype = actual.dtype if actual.dtype == expected.dtype else torch.promote_types(actual.dtype, expected.dtype)
    return _DTYPE_PRECISIONS.get(dtype, (0.0, 0.0))


def _check_are_tensors(actual: Any, expected: Any) -> Optional[AssertionError]:
    """Checks if both inputs are tensors.

    Args:
        actual (Any): Actual input.
        expected (Any): Actual input.

    Returns:
        (Optional[AssertionError]): If check did not pass.
    """
    if not (isinstance(actual, torch.Tensor) and isinstance(expected, torch.Tensor)):
        return AssertionError(f"Both inputs have to be tensors, but got {type(actual)} and {type(expected)} instead.")

    return None


def _check_supported_tensors(
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> Optional[UsageError]:  # type: ignore[valid-type]
    """Checks if the tensors are supported by the current infrastructure.

    All checks are temporary and will be relaxed in the future.

    Args:
        actual (torch.Tensor): Actual tensor.
        expected (torch.Tensor): Expected tensor.

    Returns:
        (Optional[UsageError]): If check did not pass.
    """
    if any(t.dtype in (torch.complex32, torch.complex64, torch.complex128) for t in (actual, expected)):
        return UsageError("Comparison for complex tensors is not supported yet.")
    if any(t.is_quantized for t in (actual, expected)):
        return UsageError("Comparison for quantized tensors is not supported yet.")
    if any(t.is_sparse for t in (actual, expected)):
        return UsageError("Comparison for sparse tensors is not supported yet.")

    return None


def _check_attributes_equal(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    check_device: bool = True,
    check_dtype: bool = True,
    check_stride: bool = True,
) -> Optional[AssertionError]:
    """Checks if the attributes of two tensors match.

    Always checks the :attr:`~torch.Tensor.shape`. Checks for :attr:`~torch.Tensor.device`,
    :attr:`~torch.Tensor.dtype`, and :meth:`~torch.Tensor.stride` are optional and can be disabled.

    Args:
        actual (torch.Tensor): Actual tensor.
        expected (torch.Tensor): Expected tensor.
        check_device (bool): If ``True`` (default), asserts that both :attr:`actual` and :attr:`expected` are on the
            same :attr:`~torch.Tensor.device` memory.
        check_dtype (bool): If ``True`` (default), asserts that both :attr:`actual` and :attr:`expected` have the same
            :attr:`~torch.Tensor.dtype`.
        check_stride (bool): If ``True`` (default), asserts that both :attr:`actual` and :attr:`expected` have the same
            :meth:`~torch.Tensor.stride`.

    Returns:
        (Optional[AssertionError]): If check did not pass.
    """
    msg_fmtstr = "The values for attribute '{}' do not match: {} != {}."

    if actual.shape != expected.shape:
        return AssertionError(msg_fmtstr.format("shape", actual.shape, expected.shape))

    if check_device and actual.device != expected.device:
        return AssertionError(msg_fmtstr.format("device", actual.device, expected.device))

    if check_dtype and actual.dtype != expected.dtype:
        return AssertionError(msg_fmtstr.format("dtype", actual.dtype, expected.dtype))

    if check_stride and actual.stride() != expected.stride():
        return AssertionError(msg_fmtstr.format("stride()", actual.stride(), expected.stride()))

    return None


def _equalize_attributes(actual: torch.Tensor, expected: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Equalizes some attributes of two tensors for value comparison.

    If :attr:`actual` and :attr:`expected`
    - are not onn the same memory :attr:`~torch.Tensor.device`, they are moved CPU memory, and
    - do not have the same :attr:`~torch.Tensor.dtype`, they are copied to the :class:`~torch.dtype` returned by
        :func:`torch.promote_types`.

    Args:
        actual (torch.Tensor): Actual tensor.
        expected (torch.Tensor): Expected tensor.

    Returns:
        Tuple(torch.Tensor, torch.Tensor): Equalized tensors.
    """
    if actual.device != expected.device:
        actual = actual.cpu()
        expected = expected.cpu()

    if actual.dtype != expected.dtype:
        dtype = torch.promote_types(actual.dtype, expected.dtype)
        actual = actual.to(dtype)
        expected = expected.to(dtype)

    return actual, expected


_Trace = namedtuple(
    "_Trace",
    (
        "total_elements",
        "total_mismatches",
        "mismatch_ratio",
        "max_abs_diff",
        "max_abs_diff_idx",
        "max_rel_diff",
        "max_rel_diff_idx",
    ),
)


def _trace_mismatches(actual: torch.Tensor, expected: torch.Tensor, mismatches: torch.Tensor) -> _Trace:
    """Traces mismatches.

    Args:
        actual (torch.Tensor): Actual tensor.
        expected (torch.Tensor): Expected tensor.
        mismatches (torch.Tensor): Boolean mask of the same shape as :attr:`actual` and :attr:`expected` that indicates
            the location of mismatches.

    Returns:
        (NamedTuple): Mismatch diagnostics with the following fields:

            - total_elements (int): Total number of values.
            - total_mismatches (int): Total number of mismatches.
            - mismatch_ratio (float): Quotient of total mismatches and total elements.
            - max_abs_diff (Union[int, float]): Greatest absolute difference of :attr:`actual` and :attr:`expected`.
            - max_abs_diff_idx (Union[int, Tuple[int, ...]]): Index of greatest absolute difference.
            - max_rel_diff (Union[int, float]): Greatest relative difference of :attr:`actual` and :attr:`expected`.
            - max_rel_diff_idx (Union[int, Tuple[int, ...]]): Index of greatest relative difference.

            The returned type of ``max_abs_diff`` and ``max_rel_diff`` depends on the :attr:`~torch.Tensor.dtype` of
            :attr:`actual` and :attr:`expected`.
    """
    total_elements = mismatches.numel()
    total_mismatches = torch.sum(mismatches).item()
    mismatch_ratio = total_mismatches / total_elements

    dtype = torch.float64 if actual.dtype.is_floating_point else torch.int64
    a_flat = actual.flatten().to(dtype)
    b_flat = expected.flatten().to(dtype)

    abs_diff = torch.abs(a_flat - b_flat)
    max_abs_diff, max_abs_diff_flat_idx = torch.max(abs_diff, 0)

    rel_diff = abs_diff / torch.abs(b_flat)
    max_rel_diff, max_rel_diff_flat_idx = torch.max(rel_diff, 0)

    return _Trace(
        total_elements=total_elements,
        total_mismatches=total_mismatches,
        mismatch_ratio=mismatch_ratio,
        max_abs_diff=max_abs_diff.item(),
        max_abs_diff_idx=_unravel_index(max_abs_diff_flat_idx.item(), mismatches.shape),
        max_rel_diff=max_rel_diff.item(),
        max_rel_diff_idx=_unravel_index(max_rel_diff_flat_idx.item(), mismatches.shape),
    )


def _check_values_equal(actual: torch.Tensor, expected: torch.Tensor) -> Optional[AssertionError]:
    """Checks if the values of two tensors are bitwise equal.

    Args:
        actual (torch.Tensor): Actual tensor.
        expected (torch.Tensor): Expected tensor.

    Returns:
        (Optional[AssertionError]): If check did not pass.
    """
    mismatches = torch.ne(actual, expected)
    if not torch.any(mismatches):
        return None

    trace = _trace_mismatches(actual, expected, mismatches)
    return AssertionError(
        f"Tensors are not equal!\n\n"
        f"Mismatched elements: {trace.total_mismatches} / {trace.total_elements} ({trace.mismatch_ratio:.1%})\n"
        f"Greatest absolute difference: {trace.max_abs_diff} at {trace.max_abs_diff_idx}\n"
        f"Greatest relative difference: {trace.max_rel_diff} at {trace.max_rel_diff_idx}"
    )


def _check_values_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    rtol,
    atol,
) -> Optional[AssertionError]:
    """Checks if the values of two tensors are close up to a desired tolerance.

    Args:
        actual (torch.Tensor): Actual tensor.
        expected (torch.Tensor): Expected tensor.
        rtol (float): Relative tolerance.
        atol (float): Absolute tolerance.

    Returns:
        (Optional[AssertionError]): If check did not pass.
    """
    mismatches = ~torch.isclose(actual, expected, rtol=rtol, atol=atol)
    if not torch.any(mismatches):
        return None

    trace = _trace_mismatches(actual, expected, mismatches)
    return AssertionError(
        f"Tensors are not close!\n\n"
        f"Mismatched elements: {trace.total_mismatches} / {trace.total_elements} ({trace.mismatch_ratio:.1%})\n"
        f"Greatest absolute difference: {trace.max_abs_diff} at {trace.max_abs_diff_idx} (up to {atol} allowed)\n"
        f"Greatest relative difference: {trace.max_rel_diff} at {trace.max_rel_diff_idx} (up to {rtol} allowed)"
    )


def assert_tensors_equal(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    check_device: bool = True,
    check_dtype: bool = True,
    check_stride: bool = True,
) -> None:
    """Asserts that the values of two tensors are bitwise equal.

    Optionally, checks that some attributes of both tensors are equal.

    Args:
        actual (torch.Tensor): Actual tensor.
        expected (torch.Tensor): Expected tensor.
        check_device (bool): If ``True`` (default), asserts that both :attr:`actual` and :attr:`expected` are on the
            same :attr:`~torch.Tensor.device` memory. If this check is disabled **and** :attr:`actual` and
            :attr:`expected` are not on the same memory :attr:`~torch.Tensor.device`, they are moved CPU memory before
            their values are compared.
        check_dtype (bool): If ``True`` (default), asserts that both :attr:`actual` and :attr:`expected` have the same
            :attr:`~torch.Tensor.dtype`. If this check is disabled **and** :attr:`actual` and :attr:`expected` do not
            have the same :attr:`~torch.Tensor.dtype`, they are copied to the :class:`~torch.dtype` returned by
            :func:`torch.promote_types` before their values are compared.
        check_stride (bool): If ``True`` (default), asserts that both :attr:`actual` and :attr:`expected` have the same
            stride.

    Raises:
        UsageError: If :attr:`actual` or :attr:`expected` is complex, quantized, or sparse. This is a temporary
            restriction and will be relaxed in the future.
        AssertionError: If :attr:`actual` and :attr:`expected` do not have the same :attr:`~torch.Tensor.shape`.
        AssertionError: If :attr:`check_device`, but :attr:`actual` and :attr:`expected` are not on the same
            :attr:`~torch.Tensor.device` memory.
        AssertionError: If :attr:`check_dtype`, but :attr:`actual` and :attr:`expected` do not have the same
            :attr:`~torch.Tensor.dtype`.
        AssertionError: If :attr:`check_stride`, but :attr:`actual` and :attr:`expected` do not have the same stride.
        AssertionError: If the values of :attr:`actual` and :attr:`expected` are not bitwise equal.

    .. seealso::

        To assert that the values in two tensors are are close but are not required to be bitwise equal, use
        :func:`assert_tensors_close` instead.
    """
    exc: Optional[Exception] = _check_are_tensors(actual, expected)
    if exc:
        raise exc

    exc = _check_supported_tensors(actual, expected)
    if exc:
        raise exc

    exc = _check_attributes_equal(
        actual, expected, check_device=check_device, check_dtype=check_dtype, check_stride=check_stride
    )
    if exc:
        raise exc
    actual, expected = _equalize_attributes(actual, expected)

    exc = _check_values_equal(actual, expected)
    if exc:
        raise exc


def assert_tensors_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    rtol: Optional[float] = None,
    atol: Optional[float] = None,
    check_device: bool = True,
    check_dtype: bool = True,
    check_stride: bool = True,
) -> None:
    """Asserts that the values of two tensors are close up to a desired tolerance.

    If both tolerances, :attr:`rtol` and :attr:`rtol`, are ``0``, asserts that :attr:`actual` and :attr:`expected` are bitwise
    equal. Optionally, checks that some attributes of both tensors are equal.

    Args:
        actual (torch.Tensor): Actual tensor.
        expected (torch.Tensor): Expected tensor.
        rtol (Optional[float]): Relative tolerance. If specified :attr:`atol` must also be specified. If omitted,
            default values based on the :attr:`~torch.Tensor.dtype` are selected with the below table.
        atol (Optional[float]): Absolute tolerance. If specified :attr:`rtol` must also be specified. If omitted,
            default values based on the :attr:`~torch.Tensor.dtype` are selected with the below table.
        check_device (bool): If ``True`` (default), asserts that both :attr:`actual` and :attr:`expected` are on the
            same :attr:`~torch.Tensor.device` memory. If this check is disabled **and** :attr:`actual` and
            :attr:`expected` are not on the same memory :attr:`~torch.Tensor.device`, they are moved CPU memory before
            their values are compared.
        check_dtype (bool): If ``True`` (default), asserts that both :attr:`actual` and :attr:`expected` have the same
            :attr:`~torch.Tensor.dtype`. If this check is disabled **and** :attr:`actual` and :attr:`expected` do not
            have the same :attr:`~torch.Tensor.dtype`, they are copied to the :class:`~torch.dtype` returned by
            :func:`torch.promote_types` before their values are compared.
        check_stride (bool): If ``True`` (default), asserts that both :attr:`actual` and :attr:`expected` have the same
            stride.

    Raises:
        UsageError: If :attr:`actual` or :attr:`expected` is complex, quantized, or sparse. This is a temporary
            restriction and will be relaxed in the future.
        AssertionError: If :attr:`actual` and :attr:`expected` do not have the same :attr:`~torch.Tensor.shape`.
        AssertionError: If :attr:`check_device`, but :attr:`actual` and :attr:`expected` are not on the same
            :attr:`~torch.Tensor.device` memory.
        AssertionError: If :attr:`check_dtype`, but :attr:`actual` and :attr:`expected` do not have the same
            :attr:`~torch.Tensor.dtype`.
        AssertionError: If :attr:`check_stride`, but :attr:`actual` and :attr:`expected` do not have the same stride.
        AssertionError: If the values of :attr:`actual` and :attr:`expected` are close up to a desired tolerance.



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
    exc: Optional[Exception] = _check_are_tensors(actual, expected)
    if exc:
        raise exc

    exc = _check_supported_tensors(actual, expected)
    if exc:
        raise exc

    if (rtol is None) ^ (atol is None):
        # We require both tolerance to be omitted or specified, because specifying only one might lead to surprising
        # results. Imagine setting atol=0.0 and the tensors still match because rtol>0.0.
        raise UsageError(
            f"Both 'rtol' and 'atol' must be omitted or specified, " f"but got rtol={rtol} and atol={atol} instead."
        )
    elif rtol is None:
        rtol, atol = _get_default_rtol_and_atol(actual, expected)

    exc = _check_attributes_equal(
        actual, expected, check_device=check_device, check_dtype=check_dtype, check_stride=check_stride
    )
    if exc:
        raise exc
    actual, expected = _equalize_attributes(actual, expected)

    if (rtol == 0.0) and (atol == 0.0):
        exc = _check_values_equal(actual, expected)
    else:
        exc = _check_values_close(actual, expected, rtol=rtol, atol=atol)
    if exc:
        raise exc
