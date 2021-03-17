import functools
import inspect
import sys
import warnings
from typing import Any, Callable, Optional

import torch
from torch.testing._internal.common_utils import get_comparison_dtype as _get_comparison_dtype
from torch.testing._internal.common_utils import get_default_rtol_and_atol as _get_default_rtol_and_atol

__all__ = ["assert_tensors_equal", "assert_tensors_allclose"]


# The module 'pytest' will be imported if the 'pytest' runner is used. This will only give false-positives in case
# the test directly or indirectly imports 'pytest', but is run by another runner such as 'unittest'.
_RUN_BY_PYTEST = "pytest" in sys.modules


# The Usage(Error|Warning) should be raised in case the test function is not used correctly. With this the user is able
# to differentiate between a test failure (there is a bug in the tested code) and a test error (there is a bug in the
# test). If pytest is the test runner, we use the built-in UsageError instead our custom one.
if _RUN_BY_PYTEST:
    import pytest

    UsageError = pytest.UsageError
else:
    class UsageError(Exception):
        pass


class UsageWarning(Warning):
    pass


def _hide_internal_traceback_pytest(fn):
    """Decorator for user-facing function that hides the internal traceback for :mod:`pytest`.

    The decorator works by assigning ``__tracebackhide__ = True`` in each previous frame in case an
    :class:`AssertionError` is encountered. If it is set manually, ``__tracebackhide__`` is not overwritten.

    This is a :mod:`pytest`
    `feature <https://docs.pytest.org/en/stable/example/simple.html#writing-well-integrated-assertion-helpers>`_
    and thus this is a no-op if :mod:`pytest` is not present. If the :mod:`pytest` detection gives a false-positive,
    this decorator will add a layer to the traceback, but otherwise is still a no-op.
    """
    if not _RUN_BY_PYTEST:
        return fn

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except AssertionError:
            for frame_info in inspect.trace():
                frame_info.frame.f_locals.setdefault("__tracebackhide__", True)
            raise

    return wrapper


def _assert_are_tensors(a: Any, b: Any) -> None:
    """Asserts that both inputs are tensors.

    Args:
        a (Any): First input.
        b (Any): Second input.

    Raises:
        AssertionError: If :attr:`a` or :attr:`b` is not a :class:`~torch.Tensor`.
    """
    if not (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)):
        raise AssertionError(f"Both inputs have to be tensors, but got {type(a)} and {type(b)} instead.")


def _assert_attributes_equal(
    a: torch.Tensor,
    b: torch.Tensor,
    /,
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
    msg_fmtstr = "The values for attribute '{}' do not match: {} != {}."

    if a.shape != b.shape:
        raise AssertionError(msg_fmtstr.format("shape", a.shape, b.shape))

    if check_device and a.device != b.device:
        raise AssertionError(msg_fmtstr.format("device", a.device, b.device))

    if check_dtype and a.dtype != b.dtype:
        raise AssertionError(msg_fmtstr.format("dtype", a.dtype, b.dtype))

    if check_stride and a.stride() != b.stride():
        raise AssertionError(msg_fmtstr.format("stride()", a.stride(), b.stride()))


def _assert_values_equal(a: torch.Tensor, b: torch.Tensor):
    """Asserts that the values of two tensors are bitwise equal.

    Args:
        a (torch.Tensor): First tensor.
        b (torch.Tensor): Second tensor.

    Raises:
         AssertionError: If the values of :attr:`a` and :attr:`b` are not bitwise equal.

    .. seealso::

        Internally :func:`torch.eq` is used to check for bitwise equality.
    """
    if not torch.all(torch.eq(a, b)):
        raise AssertionError("ADDME")


def _assert_values_allclose(
    a: torch.Tensor, b: torch.Tensor, /, *, rtol: Optional[float] = None, atol: Optional[float] = None
) -> None:
    """Asserts that the values of two tensors are equal up to a desired precision.

    Args:
        a (torch.Tensor): First tensor.
        b (torch.Tensor): Second tensor.
        rtol (Optional[float]): Relative tolerance. If omitted, a default value based on the
            :attr:`~torch.Tensor.dtype` is selected with the table from :func:`assert_tensors_allclose`.
        atol (Optional[float]): Absolute tolerance. If omitted, a default value based on the
            :attr:`~torch.Tensor.dtype` is selected with the table from :func:`assert_tensors_allclose`.

    Raises:
         AssertionError: If the values of :attr:`a` and :attr:`b` are equal up to the desired precision.

    .. seealso::

        Internally :func:`torch.allclose` is used to check for bitwise equality.
    """
    default_rtol, default_atol = _get_default_rtol_and_atol(a, b)
    rtol = rtol if rtol is not None else default_rtol
    atol = atol if atol is not None else default_atol

    if not torch.allclose(a, b, rtol=rtol, atol=atol):
        raise AssertionError("ADDME")


def _compare_tensors(
    a: torch.Tensor,
    b: torch.Tensor,
    /,
    values_asserter: Callable[[torch.Tensor, torch.Tensor], None],
    *,
    check_device: bool = True,
    check_dtype: bool = True,
    check_stride: bool = True,
) -> None:
    """Compare values and attributes and values.

    Args:
        a (torch.Tensor): First tensor.
        b (torch.Tensor): Second tensor.
        values_asserter (Callable[[torch.Tensor, torch.Tensor], None]): Will be called with :attr:`a` and :attr:`b`
            after the attribute checks. Must raise an :class:`AssertionError` if the values do not match.
        check_device (bool): If ``True`` (default), asserts that both :attr:`a` and :attr:`b` live in the same
            :attr:`~torch.Tensor.device` memory. If this check is disabled **and** :attr:`a` and :attr:`b` do not live
            in the same memory :attr:`~torch.Tensor.device`, they are moved CPU memory before their values are
            compared.
        check_dtype (bool): If ``True`` (default), asserts that both :attr:`a` and :attr:`b` have the same
            :attr:`~torch.Tensor.dtype`. If this check is disabled **and** :attr:`a` and :attr:`b` do not have the same
            :attr:`~torch.Tensor.dtype`, the comparison :attr:`~torch.Tensor.dtype` is determined by
            :func:`torch.promote_types`.
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
        AssertionError: If the values of :attr:`a` and :attr:`b` do not match according to :attr:`values_asserter`.
        UsageError: If :attr:`a` or :attr:`b` is complex, quantized, or sparse. This is a temporary restriction and
            will be relaxed in the future.
    """
    if any(tensor.dtype in (torch.complex32, torch.complex64, torch.complex128) for tensor in (a, b)):
        raise UsageError("Comparison for complex tensors is not supported yet.")
    if any(tensor.is_quantized for tensor in (a, b)):
        raise UsageError("Comparison for quantized tensors is not supported yet.")
    if any(tensor.is_sparse for tensor in (a, b)):
        raise UsageError("Comparison for sparse tensors is not supported yet.")

    _assert_attributes_equal(a, b, check_device=check_device, check_dtype=check_dtype, check_stride=check_stride)

    if a.device != b.device:
        a = a.cpu()
        b = b.cpu()

    if a.dtype != b.dtype:
        dtype = _get_comparison_dtype(a, b)
        a = a.to(dtype)
        b = b.to(dtype)

    values_asserter(a, b)


@_hide_internal_traceback_pytest
def assert_tensors_equal(
    a: torch.Tensor,
    b: torch.Tensor,
    /,
    *,
    warn_floating_point_dtype: bool = True,
    check_device: bool = True,
    check_dtype: bool = True,
    check_stride: bool = True,
) -> None:
    """Asserts that the values of two tensors are bitwise equal.

    Optionally, checks that some attributes of both tensors are equal.

    Args:
        a (torch.Tensor): First tensor.
        b (torch.Tensor): Second tensor.
        warn_floating_point_dtype (bool): If ``True`` (default), emit a warning if :attr:`a` or :attr:`b` is a
            floating point tensor.
        check_device (bool): If ``True`` (default), asserts that both :attr:`a` and :attr:`b` live in the same
            :attr:`~torch.Tensor.device` memory. If this check is disabled **and** :attr:`a` and :attr:`b` do not live
            in the same memory :attr:`~torch.Tensor.device`, they are moved CPU memory before their values are
            compared.
        check_dtype (bool): If ``True`` (default), asserts that both :attr:`a` and :attr:`b` have the same
            :attr:`~torch.Tensor.dtype`. If this check is disabled **and** :attr:`a` and :attr:`b` do not have the same
            :attr:`~torch.Tensor.dtype`, the comparison :attr:`~torch.Tensor.dtype` is determined by
            :func:`torch.promote_types`.
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
        AssertionError: If the values of :attr:`a` and :attr:`b` are not bitwise equal.
        UsageError: If :attr:`a` or :attr:`b` is complex, quantized, or sparse. This is a temporary restriction and
            will be relaxed in the future.

    .. seealso::

        To compare tensors for value proximity, :func:`assert_tensors_allclose` can be used.
    """
    _assert_are_tensors(a, b)

    is_floating_point = a.dtype.is_floating_point or b.dtype.is_floating_point
    if is_floating_point and warn_floating_point_dtype:
        warnings.warn(
            "If one tensor has an floating-point dtype, "
            "it is recommended to use the respective 'allclose' variant instead.",
            UsageWarning,
        )

    _compare_tensors(
        a,
        b,
        values_asserter=_assert_values_equal,
        check_device=check_device,
        check_dtype=check_dtype,
        check_stride=check_stride,
    )


@_hide_internal_traceback_pytest
def assert_tensors_allclose(
    a: torch.Tensor,
    b: torch.Tensor,
    /,
    *,
    warn_integer_dtypes: bool = True,
    rtol: Optional[float] = None,
    atol: Optional[float] = None,
    check_device: bool = True,
    check_dtype: bool = True,
    check_stride: bool = True,
) -> None:
    """Asserts that the values of two tensors are equal up to a desired precision.

    Optionally, checks that some attributes of both tensors are equal.

    Args:
        a (torch.Tensor): First tensor.
        b (torch.Tensor): Second tensor.
        warn_integer_dtypes (bool): If ``True`` (default), emit a warning if :attr:`a` and :attr:`b` are integer
            tensors.
        rtol (Optional[float]): Relative tolerance. If omitted, a default value based on the
            :attr:`~torch.Tensor.dtype` is selected with the below table.
        atol (Optional[float]): Absolute tolerance. If omitted, a default value based on the
            :attr:`~torch.Tensor.dtype` is selected with the below table.
        check_device (bool): If ``True`` (default), asserts that both :attr:`a` and :attr:`b` live in the same
            :attr:`~torch.Tensor.device` memory. If this check is disabled **and** :attr:`a` and :attr:`b` do not live
            in the same memory :attr:`~torch.Tensor.device`, they are moved CPU memory before their values are
            compared.
        check_dtype (bool): If ``True`` (default), asserts that both :attr:`a` and :attr:`b` have the same
            :attr:`~torch.Tensor.dtype`. If this check is disabled **and** :attr:`a` and :attr:`b` do not have the same
            :attr:`~torch.Tensor.dtype`, the comparison :attr:`~torch.Tensor.dtype` is determined by
            :func:`torch.promote_types`.
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
        AssertionError: If the values of :attr:`a` and :attr:`b` are equal up to the desired precision.
        UsageError: If :attr:`a` or :attr:`b` is complex, quantized, or sparse. This is a temporary restriction and
            will be relaxed in the future.


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

        To compare tensors for bitwise equality, :func:`assert_tensors_equal` can be used.
    """
    _assert_are_tensors(a, b)

    are_integer_dtypes = not a.dtype.is_floating_point and not b.dtype.is_floating_point
    if are_integer_dtypes and warn_integer_dtypes:
        warnings.warn(
            "If both tensors have an integer dtype, it is recommended to use the respective 'equal' variant instead.",
            UsageWarning,
        )

    _compare_tensors(
        a,
        b,
        values_asserter=functools.partial(_assert_values_allclose, rtol=rtol, atol=atol),
        check_device=check_device,
        check_dtype=check_dtype,
        check_stride=check_stride,
    )
