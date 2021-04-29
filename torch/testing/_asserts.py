import collections.abc
import functools
import sys
from typing import Any, Callable, Dict, List, Mapping, NamedTuple, Optional, Sequence, Tuple, Type, TypeVar, Union, cast
from types import SimpleNamespace

import torch
from torch import Tensor

from ._core import _unravel_index

__all__ = ["assert_equal", "assert_close"]


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


def _get_default_rtol_and_atol(actual: Tensor, expected: Tensor) -> Tuple[float, float]:
    dtype = actual.dtype if actual.dtype == expected.dtype else torch.promote_types(actual.dtype, expected.dtype)
    return _DTYPE_PRECISIONS.get(dtype, (0.0, 0.0))


def _check_supported_tensor(
    input: Tensor,
) -> Optional[UsageError]:  # type: ignore[valid-type]
    """Checks if the tensors are supported by the current infrastructure.

    All checks are temporary and will be relaxed in the future.

    Returns:
        (Optional[UsageError]): If check did not pass.
    """
    if input.dtype in (torch.complex32, torch.complex64, torch.complex128):
        return UsageError("Comparison for complex tensors is not supported yet.")
    if input.is_quantized:
        return UsageError("Comparison for quantized tensors is not supported yet.")
    if input.is_sparse:
        return UsageError("Comparison for sparse tensors is not supported yet.")

    return None


def _check_attributes_equal(
    actual: Tensor,
    expected: Tensor,
    *,
    check_device: bool = True,
    check_dtype: bool = True,
    check_stride: bool = True,
) -> Optional[AssertionError]:
    """Checks if the attributes of two tensors match.

    Always checks the :attr:`~torch.Tensor.shape`. Checks for :attr:`~torch.Tensor.device`,
    :attr:`~torch.Tensor.dtype`, and :meth:`~torch.Tensor.stride` are optional and can be disabled.

    Args:
        actual (Tensor): Actual tensor.
        expected (Tensor): Expected tensor.
        check_device (bool): If ``True`` (default), asserts that both :attr:`actual` and :attr:`expected` are on the
            same :attr:`~torch.Tensor.device` memory.
        check_dtype (bool): If ``True`` (default), asserts that both :attr:`actual` and :attr:`expected` have the same
            :attr:`~torch.Tensor.dtype`.
        check_stride (bool): If ``True`` (default), asserts that both :attr:`actual` and :attr:`expected` have the same
            :meth:`~torch.Tensor.stride`.

    Returns:
        (Optional[AssertionError]): If checks did not pass.
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


def _equalize_attributes(actual: Tensor, expected: Tensor) -> Tuple[Tensor, Tensor]:
    """Equalizes some attributes of two tensors for value comparison.

    If :attr:`actual` and :attr:`expected`
    - are not onn the same memory :attr:`~torch.Tensor.device`, they are moved CPU memory, and
    - do not have the same :attr:`~torch.Tensor.dtype`, they are copied to the :class:`~torch.dtype` returned by
        :func:`torch.promote_types`.

    Args:
        actual (Tensor): Actual tensor.
        expected (Tensor): Expected tensor.

    Returns:
        Tuple(Tensor, Tensor): Equalized tensors.
    """
    if actual.device != expected.device:
        actual = actual.cpu()
        expected = expected.cpu()

    if actual.dtype != expected.dtype:
        dtype = torch.promote_types(actual.dtype, expected.dtype)
        actual = actual.to(dtype)
        expected = expected.to(dtype)

    return actual, expected


def _trace_mismatches(actual: Tensor, expected: Tensor, mismatches: Tensor) -> SimpleNamespace:
    """Traces mismatches and returns diagnostics.

    Args:
        actual (Tensor): Actual tensor.
        expected (Tensor): Expected tensor.
        mismatches (Tensor): Boolean mask of the same shape as :attr:`actual` and :attr:`expected` that indicates
            the location of mismatches.

    Returns:
        (SimpleNamespace): Mismatch diagnostics with the following attributes:

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

    return SimpleNamespace(
        total_elements=total_elements,
        total_mismatches=cast(int, total_mismatches),
        mismatch_ratio=mismatch_ratio,
        max_abs_diff=max_abs_diff.item(),
        max_abs_diff_idx=_unravel_index(max_abs_diff_flat_idx.item(), mismatches.shape),
        max_rel_diff=max_rel_diff.item(),
        max_rel_diff_idx=_unravel_index(max_rel_diff_flat_idx.item(), mismatches.shape),
    )


def _check_values_equal(
    actual: Tensor,
    expected: Tensor,
    *,
    msg: Optional[Union[str, Callable[[Tensor, Tensor, SimpleNamespace], str]]] = None,
) -> Optional[AssertionError]:
    """Checks if the values of two tensors are bitwise equal.

    Args:
        actual (Tensor): Actual tensor.
        expected (Tensor): Expected tensor.
        msg (Optional[Union[str, Callable[[Tensor, Tensor, SimpleNamespace], str]]]): Optional error message. Can be
            passed as callable in which case it will be called with the inputs and the result of
            :func:`_trace_mismatches`.

    Returns:
        (Optional[AssertionError]): If check did not pass.
    """
    mismatches = torch.ne(actual, expected)
    if not torch.any(mismatches):
        return None

    trace = _trace_mismatches(actual, expected, mismatches)

    if msg is None:
        msg = (
            f"Tensors are not equal!\n\n"
            f"Mismatched elements: {trace.total_mismatches} / {trace.total_elements} ({trace.mismatch_ratio:.1%})\n"
            f"Greatest absolute difference: {trace.max_abs_diff} at {trace.max_abs_diff_idx}\n"
            f"Greatest relative difference: {trace.max_rel_diff} at {trace.max_rel_diff_idx}"
        )
    elif callable(msg):
        msg = msg(actual, expected, trace)
    return AssertionError(msg)


def _check_values_close(
    actual: Tensor,
    expected: Tensor,
    *,
    rtol: float,
    atol: float,
    equal_nan: bool,
    msg: Optional[Union[str, Callable[[Tensor, Tensor, SimpleNamespace], str]]],
) -> Optional[AssertionError]:
    """Checks if the values of two tensors are close up to a desired tolerance.

    Args:
        actual (Tensor): Actual tensor.
        expected (Tensor): Expected tensor.
        rtol (float): Relative tolerance.
        atol (float): Absolute tolerance.
        equal_nan (bool): If ``True``, two ``NaN`` values will be considered equal.
        msg (Optional[Union[str, Callable[[Tensor, Tensor, SimpleNamespace], str]]]): Optional error message. Can be
            passed as callable in which case it will be called with the inputs and the result of
            :func:`_trace_mismatches`.

    Returns:
        (Optional[AssertionError]): If check did not pass.
    """

    mismatches = ~torch.isclose(actual, expected, rtol=rtol, atol=atol, equal_nan=equal_nan)
    if not torch.any(mismatches):
        return None

    trace = _trace_mismatches(actual, expected, mismatches)
    if msg is None:
        msg = (
            f"Tensors are not close!\n\n"
            f"Mismatched elements: {trace.total_mismatches} / {trace.total_elements} ({trace.mismatch_ratio:.1%})\n"
            f"Greatest absolute difference: {trace.max_abs_diff} at {trace.max_abs_diff_idx} (up to {atol} allowed)\n"
            f"Greatest relative difference: {trace.max_rel_diff} at {trace.max_rel_diff_idx} (up to {rtol} allowed)"
        )
    elif callable(msg):
        msg = msg(actual, expected, trace)
    return AssertionError(msg)


def _check_tensors_equal(
    actual: Tensor,
    expected: Tensor,
    *,
    check_device: bool = True,
    check_dtype: bool = True,
    check_stride: bool = True,
    msg: Optional[Union[str, Callable[[Tensor, Tensor, SimpleNamespace], str]]] = None,
) -> Optional[Exception]:
    """Checks that the values of two tensors are bitwise equal.

    Optionally, checks that some attributes of both tensors are equal.

    For a description of the parameters see :func:`assert_equal`.

    Returns:
        Optional[Exception]: If checks did not pass.
    """
    exc: Optional[Exception] = _check_attributes_equal(
        actual, expected, check_device=check_device, check_dtype=check_dtype, check_stride=check_stride
    )
    if exc:
        return exc
    actual, expected = _equalize_attributes(actual, expected)

    exc = _check_values_equal(actual, expected, msg=msg)
    if exc:
        return exc

    return None


def _check_tensors_close(
    actual: Tensor,
    expected: Tensor,
    *,
    rtol: Optional[float] = None,
    atol: Optional[float] = None,
    equal_nan: bool = False,
    check_device: bool = True,
    check_dtype: bool = True,
    check_stride: bool = True,
    msg: Optional[Union[str, Callable[[Tensor, Tensor, SimpleNamespace], str]]] = None,
) -> Optional[Exception]:
    r"""Checks that the values of two tensors are close.

    Closeness is defined by

    .. math::

        \lvert a - b \rvert \le \texttt{atol} + \texttt{rtol} \cdot \lvert b \rvert

    If both tolerances, :attr:`rtol` and :attr:`rtol`, are ``0``, asserts that :attr:`actual` and :attr:`expected` are
    bitwise equal.

    Optionally, checks that some attributes of both tensors are equal.

    For a description of the parameters see :func:`assert_equal`.

    Returns:
        Optional[Exception]: If checks did not pass.
    """
    if (rtol is None) ^ (atol is None):
        # We require both tolerance to be omitted or specified, because specifying only one might lead to surprising
        # results. Imagine setting atol=0.0 and the tensors still match because rtol>0.0.
        return UsageError(
            f"Both 'rtol' and 'atol' must be omitted or specified, but got rtol={rtol} and atol={atol} instead."
        )
    elif rtol is None or atol is None:
        rtol, atol = _get_default_rtol_and_atol(actual, expected)

    exc: Optional[Exception] = _check_attributes_equal(
        actual, expected, check_device=check_device, check_dtype=check_dtype, check_stride=check_stride
    )
    if exc:
        raise exc
    actual, expected = _equalize_attributes(actual, expected)

    if (rtol == 0.0) and (atol == 0.0):
        exc = _check_values_equal(actual, expected, msg=msg)
    else:
        exc = _check_values_close(actual, expected, rtol=rtol, atol=atol, equal_nan=equal_nan, msg=msg)
    if exc:
        return exc

    return None


E = TypeVar("E", bound=Exception)


def _amend_error_message(exc: E, msg_fmtstr: str) -> E:
    """Amends an exception message.

    Args:
        exc (E): Exception.
        msg_fmtstr: Format string for the amended message.

    Returns:
        (E): New exception with amended error message.
    """
    return type(exc)(msg_fmtstr.format(str(exc)))


_SEQUENCE_MSG_FMTSTR = "The failure occurred at index {} of the sequences."
_MAPPING_MSG_FMTSTR = "The failure occurred for key '{}' of the mappings."


def _check_inputs(
    actual: Union[Tensor, List[Tensor], Dict[Any, Tensor]],
    expected: Union[Tensor, List[Tensor], Dict[Any, Tensor]],
    check_tensors: Callable[[Tensor, Tensor], Optional[Exception]],
) -> Optional[Exception]:
    """Checks inputs.

    :class:`~collections.abc.Sequence`'s and :class:`~collections.abc.Mapping`'s are checked elementwise.

    Args:
        actual (Union[Tensor, List[Tensor], Dict[Any, Tensor]]): Actual input.
        expected (Union[Tensor, List[Tensor], Dict[Any, Tensor]]): Expected input.
        check_tensors (Callable[[Any, Any], Optional[Exception]]): Callable used to check if a tensor pair matches.
            In case it mismatches should return an :class:`Exception` with an expressive error message.

    Returns:
        (Optional[Exception]): Return value of :attr:`check_tensors`.
    """
    if isinstance(actual, collections.abc.Sequence) and isinstance(expected, collections.abc.Sequence):
        for idx, (actual_t, expected_t) in enumerate(zip(actual, expected)):
            exc = check_tensors(actual_t, expected_t)
            if exc:
                return _amend_error_message(exc, f"{{}}\n\n{_SEQUENCE_MSG_FMTSTR.format(idx)}")
        else:
            return None
    elif isinstance(actual, collections.abc.Mapping) and isinstance(expected, collections.abc.Mapping):
        for key in sorted(actual.keys()):
            exc = check_tensors(actual[key], expected[key])
            if exc:
                return _amend_error_message(exc, f"{{}}\n\n{_MAPPING_MSG_FMTSTR.format(key)}")
        else:
            return None
    else:
        return check_tensors(cast(Tensor, actual), cast(Tensor, expected))


class _ParsedInputs(NamedTuple):
    actual: Union[Tensor, List[Tensor], Dict[Any, Tensor]]
    expected: Union[Tensor, List[Tensor], Dict[Any, Tensor]]


def _parse_inputs(
    actual: Any,
    expected: Any,
) -> Tuple[Optional[Exception], Optional[_ParsedInputs]]:
    """Parses inputs by constructing tensors from array-or-scalar-likes.

    :class:`~collections.abc.Sequence`'s or :class:`~collections.abc.Mapping`'s are parsed elementwise.

    Args:
        actual (Any): Actual input.
        expected (Any): Expected input.

    Returns:
        (Optional[Exception], Optional[_ParsedInputs]): The two elements are orthogonal, i.e. if the first ``is None``
            the second will not and vice versa. Check :func:`_parse_array_or_scalar_like_pair`,
            :func:`_parse_sequences`, and :func:`_parse_mappings` for possible exceptions.
    """
    if isinstance(actual, collections.abc.Sequence) and isinstance(expected, collections.abc.Sequence):
        return _parse_sequences(actual, expected)
    elif isinstance(actual, collections.abc.Mapping) and isinstance(expected, collections.abc.Mapping):
        return _parse_mappings(actual, expected)
    else:
        return _parse_array_or_scalar_like_pair(actual, expected)


def _parse_array_or_scalar_like_pair(actual: Any, expected: Any) -> Tuple[Optional[Exception], Optional[_ParsedInputs]]:
    """Parses an scalar-or-array-like pair.

    Args:
        actual: Actual array-or-scalar-like.
        expected: Expected array-or-scalar-like.

    Returns:
        (Optional[Exception], Optional[_ParsedInputs]): The two elements are orthogonal, i.e. if the first ``is None``
            the second will not and vice versa. Returns a :class:`UsageError` if :attr:`actual` and :attr:`expected` do
            not have the same type or no :class:`~torch.Tensor` can be constructed from them.
    """
    exc: Optional[Exception]

    if type(actual) is not type(expected):
        exc = UsageError(
            f"Apart from a containers type equality is required, but got {type(actual)} and {type(expected)} instead."
        )
        return exc, None

    tensors = []
    for array_or_scalar_like in (actual, expected):
        try:
            tensor = torch.as_tensor(array_or_scalar_like)
        except Exception:
            exc = UsageError(f"No tensor can be constructed from type {type(array_or_scalar_like)}.")
            return exc, None

        exc = _check_supported_tensor(tensor)
        if exc:
            return exc, None

        tensors.append(tensor)

    actual_tensor, expected_tensor = tensors
    return None, _ParsedInputs(actual_tensor, expected_tensor)


def _parse_sequences(actual: Sequence, expected: Sequence) -> Tuple[Optional[Exception], Optional[_ParsedInputs]]:
    """Parses sequences of scalar-or-array-like pairs.

    Regardless of the input types, the sequences are returned as :class:`list`.

    Args:
        actual: Actual sequence array-or-scalar-likes.
        expected: Expected sequence array-or-scalar-likes.

    Returns:
        (Optional[Exception], Optional[_ParsedInputs]): The two elements are orthogonal, i.e. if the first ``is None``
            the second will not and vice versa. Returns a :class:`AssertionError` if the length of :attr:`actual` and
            :attr:`expected` does not match. Additionally, returns any exception from
            :func:`_parse_array_or_scalar_like_pair`.
    """
    exc: Optional[Exception]

    actual_len = len(actual)
    expected_len = len(expected)
    if actual_len != expected_len:
        exc = AssertionError(f"The length of the sequences mismatch: {actual_len} != {expected_len}")
        return exc, None

    actual_lst = []
    expected_lst = []
    for idx in range(actual_len):
        exc, result = _parse_array_or_scalar_like_pair(actual[idx], expected[idx])
        if exc:
            exc = _amend_error_message(exc, f"{{}}\n\n{_SEQUENCE_MSG_FMTSTR.format(idx)}")
            return exc, None

        result = cast(_ParsedInputs, result)
        actual_lst.append(cast(Tensor, result.actual))
        expected_lst.append(cast(Tensor, result.expected))

    return None, _ParsedInputs(actual_lst, expected_lst)


def _parse_mappings(actual: Mapping, expected: Mapping) -> Tuple[Optional[Exception], Optional[_ParsedInputs]]:
    """Parses sequences of scalar-or-array-like pairs.

    Regardless of the input types, the sequences are returned as :class:`dict`.

    Args:
        actual: Actual mapping array-or-scalar-likes.
        expected: Expected mapping array-or-scalar-likes.

    Returns:
        (Optional[Exception], Optional[_ParsedInputs]): The two elements are orthogonal, i.e. if the first ``is None``
            the second will not and vice versa. Returns a :class:`AssertionError` if the keys of :attr:`actual` and
            :attr:`expected` do not match. Additionally, returns any exception from
            :func:`_parse_array_or_scalar_like_pair`.
    """
    exc: Optional[Exception]

    actual_keys = set(actual.keys())
    expected_keys = set(expected.keys())
    if actual_keys != expected_keys:
        missing_keys = expected_keys - actual_keys
        additional_keys = actual_keys - expected_keys
        exc = AssertionError(
            f"The keys of the mappings do not match:\n\n"
            f"Missing keys in the actual mapping: {sorted(missing_keys)}\n"
            f"Additional keys in the actual mapping: {sorted(additional_keys)}\n"
        )
        return exc, None

    actual_dct = {}
    expected_dct = {}
    for key in sorted(actual_keys):
        exc, result = _parse_array_or_scalar_like_pair(actual[key], expected[key])
        if exc:
            exc = _amend_error_message(exc, f"{{}}\n\n{_MAPPING_MSG_FMTSTR.format(key)}")
            return exc, None

        result = cast(_ParsedInputs, result)
        actual_dct[key] = cast(Tensor, result.actual)
        expected_dct[key] = cast(Tensor, result.expected)

    return None, _ParsedInputs(actual_dct, expected_dct)


def assert_equal(
    actual: Any,
    expected: Any,
    *,
    check_device: bool = True,
    check_dtype: bool = True,
    check_stride: bool = True,
    msg: Optional[Union[str, Callable[[Tensor, Tensor, SimpleNamespace], str]]] = None,
) -> None:
    """Asserts that the values of tensor pairs are bitwise equal.

    Optionally, checks that some attributes of tensor pairs are equal.

    Also supports array-or-scalar-like inputs from which a :class:`torch.Tensor` can be constructed with
    :func:`torch.as_tensor`. Still, requires type equality, i.e. comparing a :class:`torch.Tensor` and a
    :class:`numpy.ndarray` is not supported.

    In case both inputs are :class:`~collections.abc.Sequence`'s or :class:`~collections.abc.Mapping`'s the checks are
    performed elementwise.

    Args:
        actual (Any): Actual input.
        expected (Any): Expected input.
        check_device (bool): If ``True`` (default), asserts that each tensor pair is on the same
            :attr:`~torch.Tensor.device` memory. If this check is disabled **and** it is not on the same
            :attr:`~torch.Tensor.device` memory, it is moved CPU memory before the values are compared.
        check_dtype (bool): If ``True`` (default), asserts that each tensor pair has the same
            :attr:`~torch.Tensor.dtype`. If this check is disabled it does not have the same
            :attr:`~torch.Tensor.dtype`, it is copied to the :class:`~torch.dtype` returned by
            :func:`torch.promote_types` before the values are compared.
        check_stride (bool): If ``True`` (default), asserts that each tensor pair has the same stride.
        msg (Optional[Union[str, Callable[[Tensor, Tensor, SimpleNamespace], str]]]): Optional error message to use if
            the values of a tensor pair mismatch. Can be passed as callable in which case it will be called with the
            tensor pair and a namespace of diagnostic info about the mismatches. See below for details.

    Raises:
        UsageError: If an array-or-scalar-like pair has different types.
        UsageError: If a :class:`torch.Tensor` can't be constructed from an array-or-scalar-like.
        UsageError: If any tensor is complex, quantized, or sparse. This is a temporary restriction and
            will be relaxed in the future.
        AssertionError: If the inputs are :class:`~collections.abc.Sequence`'s, but their length does not match.
        AssertionError: If the inputs are :class:`~collections.abc.Mapping`'s, but their set of keys do not match.
        AssertionError: If a tensor pair does not have the same :attr:`~torch.Tensor.shape`.
        AssertionError: If :attr:`check_device`, but a tensor pair is not on the same :attr:`~torch.Tensor.device`
            memory.
        AssertionError: If :attr:`check_dtype`, but a tensor pair does not have the same :attr:`~torch.Tensor.dtype`.
        AssertionError: If :attr:`check_stride`, but a tensor pair does not have the same stride.
        AssertionError: If the values of a tensor pair are not bitwise equal.

    The namespace that will be passed to :attr:`msg` if its a callable comprises the following attributes:

    - total_elements (int): Total number of values.
    - total_mismatches (int): Total number of mismatches.
    - mismatch_ratio (float): Quotient of total mismatches and total elements.
    - max_abs_diff (Union[int, float]): Greatest absolute difference of the inputs.
    - max_abs_diff_idx (Union[int, Tuple[int, ...]]): Index of greatest absolute difference.
    - max_rel_diff (Union[int, float]): Greatest relative difference of the inputs.
    - max_rel_diff_idx (Union[int, Tuple[int, ...]]): Index of greatest relative difference.

    For ``max_abs_diff`` and ``max_rel_diff`` the type depends on the :attr:`~torch.Tensor.dtype` of the inputs.

    .. seealso::

        To assert that the values of a tensor pair are close but are not required to be bitwise equal, use
        :func:`assert_close` instead.
    """
    exc, parse_result = _parse_inputs(actual, expected)
    if exc:
        raise exc
    actual, expected = cast(_ParsedInputs, parse_result)

    check_tensors = functools.partial(
        _check_tensors_equal,
        check_device=check_device,
        check_dtype=check_dtype,
        check_stride=check_stride,
        msg=msg,
    )
    exc = _check_inputs(actual, expected, check_tensors)
    if exc:
        raise exc


def assert_close(
    actual: Any,
    expected: Any,
    *,
    rtol: Optional[float] = None,
    atol: Optional[float] = None,
    equal_nan: bool = False,
    check_device: bool = True,
    check_dtype: bool = True,
    check_stride: bool = True,
    msg: Optional[Union[str, Callable[[Tensor, Tensor, SimpleNamespace], str]]] = None,
) -> None:
    r"""Asserts that the values of tensor pairs are bitwise close.

    Closeness is defined by

    .. math::

        \lvert a - b \rvert \le \texttt{atol} + \texttt{rtol} \cdot \lvert b \rvert

    Optionally, checks that some attributes of tensor pairs are equal.

    Also supports array-or-scalar-like inputs from which a :class:`torch.Tensor` can be constructed with
    :func:`torch.as_tensor`. Still, requires type equality, i.e. comparing a :class:`torch.Tensor` and a
    :class:`numpy.ndarray` is not supported.

    In case both inputs are :class:`~collections.abc.Sequence`'s or :class:`~collections.abc.Mapping`'s the checks are
    performed elementwise.

    Args:
        actual (Any): Actual input.
        expected (Any): Expected input.
        rtol (Optional[float]): Relative tolerance. If specified :attr:`atol` must also be specified. If omitted,
            default values based on the :attr:`~torch.Tensor.dtype` are selected with the below table.
        atol (Optional[float]): Absolute tolerance. If specified :attr:`rtol` must also be specified. If omitted,
            default values based on the :attr:`~torch.Tensor.dtype` are selected with the below table.
        equal_nan (bool): If ``True``, two ``NaN`` values will be considered equal.
        check_device (bool): If ``True`` (default), asserts that each tensor pair is on the same
            :attr:`~torch.Tensor.device` memory. If this check is disabled **and** it is not on the same
            :attr:`~torch.Tensor.device` memory, it is moved CPU memory before the values are compared.
        check_dtype (bool): If ``True`` (default), asserts that each tensor pair has the same
            :attr:`~torch.Tensor.dtype`. If this check is disabled it does not have the same
            :attr:`~torch.Tensor.dtype`, it is copied to the :class:`~torch.dtype` returned by
            :func:`torch.promote_types` before the values are compared.
        check_stride (bool): If ``True`` (default), asserts that each tensor pair has the same stride.
        msg (Optional[Union[str, Callable[[Tensor, Tensor, SimpleNamespace], str]]]): Optional error message to use if
            the values of a tensor pair mismatch. Can be passed as callable in which case it will be called with the
            tensor pair and a namespace of diagnostic info about the mismatches. See below for details.

    Raises:
        UsageError: If an array-or-scalar-like pair has different types.
        UsageError: If a :class:`torch.Tensor` can't be constructed from an array-or-scalar-like.
        UsageError: If any tensor is complex, quantized, or sparse. This is a temporary restriction and
            will be relaxed in the future.
        UsageError: If only :attr:`rtol` or :attr:`atol` is specified.
        AssertionError: If the inputs are :class:`~collections.abc.Sequence`'s, but their length does not match.
        AssertionError: If the inputs are :class:`~collections.abc.Mapping`'s, but their set of keys do not match.
        AssertionError: If a tensor pair does not have the same :attr:`~torch.Tensor.shape`.
        AssertionError: If :attr:`check_device`, but a tensor pair is not on the same :attr:`~torch.Tensor.device`
            memory.
        AssertionError: If :attr:`check_dtype`, but a tensor pair does not have the same :attr:`~torch.Tensor.dtype`.
        AssertionError: If :attr:`check_stride`, but a tensor pair does not have the same stride.
        AssertionError: If the values of a tensor pair are not bitwise equal.

    The following table displays the default ``rtol``'s and ``atol``'s. Note that the :class:`~torch.dtype` refers to
    the promoted type in case :attr:`actual` and :attr:`expected` do not have the same :attr:`~torch.Tensor.dtype`.

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
    | other                     | ``0.0``    | ``0.0``  |
    +---------------------------+------------+----------+

    The namespace that will be passed to :attr:`msg` if its a callable comprises the following attributes:

    - total_elements (int): Total number of values.
    - total_mismatches (int): Total number of mismatches.
    - mismatch_ratio (float): Quotient of total mismatches and total elements.
    - max_abs_diff (Union[int, float]): Greatest absolute difference of the inputs.
    - max_abs_diff_idx (Union[int, Tuple[int, ...]]): Index of greatest absolute difference.
    - max_rel_diff (Union[int, float]): Greatest relative difference of the inputs.
    - max_rel_diff_idx (Union[int, Tuple[int, ...]]): Index of greatest relative difference.

    For ``max_abs_diff`` and ``max_rel_diff`` the type depends on the :attr:`~torch.Tensor.dtype` of the inputs.

    .. seealso::

        To assert that the values of a tensor pair are bitwise equal, use :func:`assert_equal` instead.
    """
    exc, parse_result = _parse_inputs(actual, expected)
    if exc:
        raise exc
    actual, expected = cast(_ParsedInputs, parse_result)

    check_tensors = functools.partial(
        _check_tensors_close,
        rtol=rtol,
        atol=atol,
        equal_nan=equal_nan,
        check_device=check_device,
        check_dtype=check_dtype,
        check_stride=check_stride,
        msg=msg,
    )
    exc = _check_inputs(actual, expected, check_tensors)
    if exc:
        raise exc
