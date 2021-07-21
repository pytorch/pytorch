import collections.abc
import functools
import numbers
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple, Type, Union, cast
from types import SimpleNamespace as Diagnostics

import torch
from torch import Tensor

from ._core import _unravel_index

__all__ = ["assert_close"]


class _TestingErrorMeta(NamedTuple):
    type: Type[Exception]
    msg: str

    def amend_msg(self, prefix: str = "", postfix: str = "") -> "_TestingErrorMeta":
        return self._replace(msg=f"{prefix}{self.msg}{postfix}")

    def to_error(self) -> Exception:
        return self.type(self.msg)


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
    actual_rtol, actual_atol = _DTYPE_PRECISIONS.get(actual.dtype, (0.0, 0.0))
    expected_rtol, expected_atol = _DTYPE_PRECISIONS.get(expected.dtype, (0.0, 0.0))
    return max(actual_rtol, expected_rtol), max(actual_atol, expected_atol)


def _check_complex_components_individually(
    check_tensors: Callable[..., Optional[_TestingErrorMeta]]
) -> Callable[..., Optional[_TestingErrorMeta]]:
    """Decorates real-valued tensor check functions to handle complex components individually.

    If the inputs are not complex, this decorator is a no-op.

    Args:
        check_tensors (Callable[[Tensor, Tensor], Optional[_TestingErrorMeta]]): Tensor check function for real-valued
        tensors.
    """

    @functools.wraps(check_tensors)
    def wrapper(
        actual: Tensor, expected: Tensor, *, equal_nan: Union[str, bool], **kwargs: Any
    ) -> Optional[_TestingErrorMeta]:
        if equal_nan == "relaxed":
            relaxed_complex_nan = True
            equal_nan = True
        else:
            relaxed_complex_nan = False

        if actual.dtype not in (torch.complex32, torch.complex64, torch.complex128):
            return check_tensors(actual, expected, equal_nan=equal_nan, **kwargs)

        if relaxed_complex_nan:
            actual, expected = [
                t.clone().masked_fill(
                    t.real.isnan() | t.imag.isnan(), complex(float("NaN"), float("NaN"))  # type: ignore[call-overload]
                )
                for t in (actual, expected)
            ]

        error_meta = check_tensors(actual.real, expected.real, equal_nan=equal_nan, **kwargs)
        if error_meta:
            return error_meta

        error_meta = check_tensors(actual.imag, expected.imag, equal_nan=equal_nan, **kwargs)
        if error_meta:
            return error_meta

        return None

    return wrapper


def _check_sparse_coo_members_individually(
    check_tensors: Callable[..., Optional[_TestingErrorMeta]]
) -> Callable[..., Optional[_TestingErrorMeta]]:
    """Decorates strided tensor check functions to individually handle sparse COO members.

    If the inputs are not sparse COO, this decorator is a no-op.

    Args:
        check_tensors (Callable[[Tensor, Tensor], Optional[Exception]]): Tensor check function for strided tensors.
    """

    @functools.wraps(check_tensors)
    def wrapper(actual: Tensor, expected: Tensor, **kwargs: Any) -> Optional[_TestingErrorMeta]:
        if not actual.is_sparse:
            return check_tensors(actual, expected, **kwargs)

        if actual._nnz() != expected._nnz():
            return _TestingErrorMeta(
                AssertionError, f"The number of specified values does not match: {actual._nnz()} != {expected._nnz()}"
            )

        kwargs_equal = dict(kwargs, rtol=0, atol=0)
        error_meta = check_tensors(actual._indices(), expected._indices(), **kwargs_equal)
        if error_meta:
            return error_meta.amend_msg(postfix="\n\nThe failure occurred for the indices.")

        error_meta = check_tensors(actual._values(), expected._values(), **kwargs)
        if error_meta:
            return error_meta.amend_msg(postfix="\n\nThe failure occurred for the values.")

        return None

    return wrapper


def _check_sparse_csr_members_individually(
    check_tensors: Callable[..., Optional[_TestingErrorMeta]]
) -> Callable[..., Optional[_TestingErrorMeta]]:
    """Decorates strided tensor check functions to individually handle sparse CSR members.

    If the inputs are not sparse CSR, this decorator is a no-op.

    Args:
        check_tensors (Callable[[Tensor, Tensor], Optional[Exception]]): Tensor check function for strided
        tensors.
    """

    @functools.wraps(check_tensors)
    def wrapper(actual: Tensor, expected: Tensor, **kwargs: Any) -> Optional[_TestingErrorMeta]:
        if not actual.is_sparse_csr:
            return check_tensors(actual, expected, **kwargs)

        kwargs_equal = dict(kwargs, rtol=0, atol=0)
        error_meta = check_tensors(actual.crow_indices(), expected.crow_indices(), **kwargs_equal)
        if error_meta:
            return error_meta.amend_msg(postfix="\n\nThe failure occurred for the crow_indices.")

        error_meta = check_tensors(actual.col_indices(), expected.col_indices(), **kwargs_equal)
        if error_meta:
            return error_meta.amend_msg(postfix="\n\nThe failure occurred for the col_indices.")

        error_meta = check_tensors(actual.values(), expected.values(), **kwargs)
        if error_meta:
            return error_meta.amend_msg(postfix="\n\nThe failure occurred for the values.")

        return None

    return wrapper


def _check_quantized(
    check_tensor_values: Callable[..., Optional[_TestingErrorMeta]]
) -> Callable[..., Optional[_TestingErrorMeta]]:
    """Decorates non-quantized tensor check functions to handle quantized tensors.

    If the inputs are not quantized, this decorator is a no-op.

    Args:
        check_tensor_values (Callable[..., Optional[_TestingErrorMeta]]): Tensor check function for continuous tensors.

    Returns:
        Optional[_TestingErrorMeta]: Return value of :attr:`check_tensors`.
    """

    @functools.wraps(check_tensor_values)
    def wrapper(actual: Tensor, expected: Tensor, **kwargs: Any) -> Optional[_TestingErrorMeta]:
        if not actual.is_quantized:
            return check_tensor_values(actual, expected, **kwargs)

        return check_tensor_values(actual.dequantize(), expected.dequantize(), **kwargs)

    return wrapper


def _check_supported_tensor(input: Tensor) -> Optional[_TestingErrorMeta]:
    """Checks if the tensor is supported by the current infrastructure.

    Returns:
        (Optional[_TestingErrorMeta]): If check did not pass.
    """
    if input.layout not in {torch.strided, torch.sparse_coo, torch.sparse_csr}:  # type: ignore[attr-defined]
        return _TestingErrorMeta(ValueError, f"Unsupported tensor layout {input.layout}")

    return None


def _check_attributes_equal(
    actual: Tensor,
    expected: Tensor,
    *,
    check_device: bool = True,
    check_dtype: bool = True,
    check_stride: bool = True,
    check_is_coalesced: bool = True,
) -> Optional[_TestingErrorMeta]:
    """Checks if the attributes of two tensors match.

    Always checks the :attr:`~torch.Tensor.shape` and :attr:`~torch.Tensor.layout`. Checks for
    :attr:`~torch.Tensor.device`, :attr:`~torch.Tensor.dtype`, :meth:`~torch.Tensor.stride` if the tensors are strided,
    and :meth:`~torch.tensor.is_coalesced` if the tensors are sparse COO are optional and can be disabled.

    Args:
        actual (Tensor): Actual tensor.
        expected (Tensor): Expected tensor.
        check_device (bool): If ``True`` (default), checks that both :attr:`actual` and :attr:`expected` are on the
            same :attr:`~torch.Tensor.device`.
        check_dtype (bool): If ``True`` (default), checks that both :attr:`actual` and :attr:`expected` have the same
            ``dtype``.
        check_stride (bool): If ``True`` (default) and the tensors are strided, checks that both :attr:`actual` and
            :attr:`expected` have the same stride.
        check_is_coalesced (bool): If ``True`` (default) and the tensors are sparse COO, checks that both
            :attr:`actual` and :attr:`expected` are either coalesced or uncoalesced.

    Returns:
        (Optional[_TestingErrorMeta]): If checks did not pass.
    """
    msg_fmtstr = "The values for attribute '{}' do not match: {} != {}."

    if actual.shape != expected.shape:
        return _TestingErrorMeta(AssertionError, msg_fmtstr.format("shape", actual.shape, expected.shape))

    if actual.layout != expected.layout:
        return _TestingErrorMeta(AssertionError, msg_fmtstr.format("layout", actual.layout, expected.layout))
    elif actual.layout == torch.strided and check_stride and actual.stride() != expected.stride():
        return _TestingErrorMeta(AssertionError, msg_fmtstr.format("stride()", actual.stride(), expected.stride()))
    elif actual.layout == torch.sparse_coo and check_is_coalesced and actual.is_coalesced() != expected.is_coalesced():
        return _TestingErrorMeta(
            AssertionError, msg_fmtstr.format("is_coalesced()", actual.is_coalesced(), expected.is_coalesced())
        )

    if check_device and actual.device != expected.device:
        return _TestingErrorMeta(AssertionError, msg_fmtstr.format("device", actual.device, expected.device))

    if actual.is_quantized != expected.is_quantized:
        return _TestingErrorMeta(
            AssertionError, msg_fmtstr.format("is_quantized", actual.is_quantized, expected.is_quantized)
        )
    elif actual.is_quantized and actual.qscheme() != expected.qscheme():
        return _TestingErrorMeta(AssertionError, msg_fmtstr.format("qscheme()", actual.qscheme(), expected.qscheme()))

    if check_dtype and actual.dtype != expected.dtype:
        return _TestingErrorMeta(AssertionError, msg_fmtstr.format("dtype", actual.dtype, expected.dtype))

    return None


def _equalize_attributes(actual: Tensor, expected: Tensor) -> Tuple[Tensor, Tensor]:
    """Equalizes some attributes of two tensors for value comparison.

    If :attr:`actual` and :attr:`expected`
    - are not on the same :attr:`~torch.Tensor.device`, they are moved CPU memory, and
    - do not have the same ``dtype``, they are promoted  to a common ``dtype`` (according to
        :func:`torch.promote_types`)

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

    if actual.is_sparse and actual.is_coalesced() != expected.is_coalesced():
        actual = actual.coalesce()
        expected = expected.coalesce()

    return actual, expected


def _trace_mismatches(actual: Tensor, expected: Tensor, mismatches: Tensor, *, rtol: float, atol: float) -> Diagnostics:
    """Traces mismatches and returns diagnostic information.

    Args:
        actual (Tensor): Actual tensor.
        expected (Tensor): Expected tensor.
        mismatches (Tensor): Boolean mask of the same shape as :attr:`actual` and :attr:`expected` that indicates
            the location of mismatches.

    Returns:
        (Diagnostics): Mismatch diagnostics with the following attributes:

            - ``number_of_elements`` (int): Number of elements in each tensor being compared.
            - ``total_mismatches`` (int): Total number of mismatches.
            - ``max_abs_diff`` (Union[int, float]): Greatest absolute difference of the inputs.
            - ``max_abs_diff_idx`` (Union[int, Tuple[int, ...]]): Index of greatest absolute difference.
            - ``atol`` (float): Allowed absolute tolerance.
            - ``max_rel_diff`` (Union[int, float]): Greatest relative difference of the inputs.
            - ``max_rel_diff_idx`` (Union[int, Tuple[int, ...]]): Index of greatest relative difference.
            - ``rtol`` (float): Allowed relative tolerance.

            For ``max_abs_diff`` and ``max_rel_diff`` the type depends on the :attr:`~torch.Tensor.dtype` of the inputs.
    """
    number_of_elements = mismatches.numel()
    total_mismatches = torch.sum(mismatches).item()

    a_flat = actual.flatten()
    b_flat = expected.flatten()
    matches_flat = ~mismatches.flatten()

    abs_diff = torch.abs(a_flat - b_flat)
    # Ensure that only mismatches are used for the max_abs_diff computation
    abs_diff[matches_flat] = 0
    max_abs_diff, max_abs_diff_flat_idx = torch.max(abs_diff, 0)

    rel_diff = abs_diff / torch.abs(b_flat)
    # Ensure that only mismatches are used for the max_rel_diff computation
    rel_diff[matches_flat] = 0
    max_rel_diff, max_rel_diff_flat_idx = torch.max(rel_diff, 0)

    return Diagnostics(
        number_of_elements=number_of_elements,
        total_mismatches=cast(int, total_mismatches),
        max_abs_diff=max_abs_diff.item(),
        max_abs_diff_idx=_unravel_index(max_abs_diff_flat_idx.item(), mismatches.shape),
        atol=atol,
        max_rel_diff=max_rel_diff.item(),
        max_rel_diff_idx=_unravel_index(max_rel_diff_flat_idx.item(), mismatches.shape),
        rtol=rtol,
    )


def _make_mismatch_msg(
    actual: Tensor,
    expected: Tensor,
    diagnostics: Diagnostics,
    *,
    identifier: Optional[Union[str, Callable[[str], str]]] = None,
) -> str:
    scalar_comparison = actual.size() == torch.Size([])
    equality = diagnostics.rtol == 0 and diagnostics.atol == 0

    def append_difference(msg: str, *, type: str, difference: float, index: Tuple[int, ...], tolerance: float) -> str:
        if scalar_comparison:
            msg += f"{type.title()} difference: {difference}"
        else:
            msg += f"Greatest {type} difference: {difference} at index {index}"
        if not equality:
            msg += f" (up to {tolerance} allowed)"
        msg += "\n"
        return msg

    default_identifier = "Scalars" if scalar_comparison else "Tensor-likes"
    if identifier is None:
        identifier = default_identifier
    elif callable(identifier):
        identifier = identifier(default_identifier)

    msg = f"{identifier} are not {'equal' if equality else 'close'}!\n\n"

    if not scalar_comparison:
        msg += (
            f"Mismatched elements: {diagnostics.total_mismatches} / {diagnostics.number_of_elements} "
            f"({diagnostics.total_mismatches / diagnostics.number_of_elements:.1%})\n"
        )

    msg = append_difference(
        msg,
        type="absolute",
        difference=diagnostics.max_abs_diff,
        index=diagnostics.max_abs_diff_idx,
        tolerance=diagnostics.atol,
    )
    msg = append_difference(
        msg,
        type="relative",
        difference=diagnostics.max_rel_diff,
        index=diagnostics.max_rel_diff_idx,
        tolerance=diagnostics.rtol,
    )

    return msg.strip()


@_check_quantized
@_check_sparse_coo_members_individually
@_check_sparse_csr_members_individually
@_check_complex_components_individually
def _check_values_close(
    actual: Tensor,
    expected: Tensor,
    *,
    rtol: float,
    atol: float,
    equal_nan: bool,
    msg: Optional[Union[str, Callable[[Tensor, Tensor, Diagnostics], str]]],
) -> Optional[_TestingErrorMeta]:
    """Checks if the values of two tensors are close up to a desired tolerance.

    Args:
        actual (Tensor): Actual tensor.
        expected (Tensor): Expected tensor.
        rtol (float): Relative tolerance.
        atol (float): Absolute tolerance.
        equal_nan (bool): If ``True``, two ``NaN`` values will be considered equal.
        msg (Optional[Union[str, Callable[[Tensor, Tensor, Diagnostics], str]]]): Optional error message. Can be passed
            as callable in which case it will be called with the inputs and the result of :func:`_trace_mismatches`.

    Returns:
        (Optional[AssertionError]): If check did not pass.
    """
    dtype = torch.float64 if actual.dtype.is_floating_point else torch.int64
    actual = actual.to(dtype)
    expected = expected.to(dtype)
    mismatches = ~torch.isclose(actual, expected, rtol=rtol, atol=atol, equal_nan=equal_nan)
    if not torch.any(mismatches):
        return None

    diagnostics = _trace_mismatches(actual, expected, mismatches, rtol=rtol, atol=atol)
    if msg is None:
        msg = _make_mismatch_msg
    if callable(msg):
        msg = msg(actual, expected, diagnostics)
    return _TestingErrorMeta(AssertionError, msg)


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
    check_is_coalesced: bool = True,
    msg: Union[str, Callable[[Tensor, Tensor, Diagnostics], str]],
) -> Optional[_TestingErrorMeta]:
    r"""Checks that the values of :attr:`actual` and :attr:`expected` are close.

    If :attr:`actual` and :attr:`expected` are real-valued and finite, they are considered close if

    .. code::

        torch.abs(actual - expected) <= (atol + rtol * expected)

    and they have the same device (if :attr:`check_device` is ``True``), same dtype (if :attr:`check_dtype` is
    ``True``), and the same stride (if :attr:`check_stride` is ``True``). Non-finite values (``-inf`` and ``inf``) are
    only considered close if and only if they are equal. ``NaN``'s are only considered equal to each other if
    :attr:`equal_nan` is ``True``.

    For a description of the parameters see :func:`assert_close`.

    Returns:
        Optional[_TestingErrorMeta]: If checks did not pass.
    """
    if rtol is None or atol is None:
        rtol, atol = _get_default_rtol_and_atol(actual, expected)

    error_meta = _check_attributes_equal(
        actual,
        expected,
        check_device=check_device,
        check_dtype=check_dtype,
        check_stride=check_stride,
        check_is_coalesced=check_is_coalesced,
    )
    if error_meta:
        return error_meta
    actual, expected = _equalize_attributes(actual, expected)

    if rtol is None or atol is None:
        rtol, atol = _DTYPE_PRECISIONS.get(actual.dtype, (0.0, 0.0))

    error_meta = _check_values_close(actual, expected, rtol=rtol, atol=atol, equal_nan=equal_nan, msg=msg)
    if error_meta:
        return error_meta

    return None


class _TensorPair(NamedTuple):
    actual: Tensor
    expected: Tensor


_SEQUENCE_MSG_FMTSTR = "The failure occurred at index {} of the sequences."
_MAPPING_MSG_FMTSTR = "The failure occurred for key '{}' of the mappings."


def _check_pair_close(
    pair: Union[_TensorPair, List, Dict],
    **kwargs: Any,
) -> Optional[_TestingErrorMeta]:
    """Checks input pairs.

    :class:`list`'s or :class:`dict`'s are checked elementwise. Checking is performed recursively and thus nested
    containers are supported.

    Args:
        pair (Union[_TensorPair, List, Dict]): Input pair.
        **kwargs (Any): Keyword arguments passed to :func:`__check_tensors_close`.

    Returns:
        (Optional[_TestingErrorMeta]): Return value of :attr:`check_tensors`.
    """
    if isinstance(pair, list):
        for idx, pair_item in enumerate(pair):
            error_meta = _check_pair_close(pair_item, **kwargs)
            if error_meta:
                return error_meta.amend_msg(postfix=f"\n\n{_SEQUENCE_MSG_FMTSTR.format(idx)}")
        else:
            return None
    elif isinstance(pair, dict):
        for key, pair_item in pair.items():
            error_meta = _check_pair_close(pair_item, **kwargs)
            if error_meta:
                return error_meta.amend_msg(postfix=f"\n\n{_MAPPING_MSG_FMTSTR.format(key)}")
        else:
            return None
    else:  # isinstance(pair, TensorPair)
        return _check_tensors_close(pair.actual, pair.expected, **kwargs)


def _to_tensor(tensor_or_scalar_like: Any) -> Tuple[Optional[_TestingErrorMeta], Optional[Tensor]]:
    """Converts a tensor-or-scalar-like to a :class:`~torch.Tensor`.
    Args:
        tensor_or_scalar_like (Any): Tensor-or-scalar-like.
    Returns:

        (Tuple[Optional[_TestingErrorMeta], Optional[Tensor]]): The two elements are orthogonal, i.e. if the first is
            ``None`` the second will be valid and vice versa. Returns :class:`_TestingErrorMeta` if no tensor can be
            constructed from :attr:`actual` or :attr:`expected`. Additionally, returns any error meta from
            :func:`_check_supported_tensor`.
    """
    error_meta: Optional[_TestingErrorMeta]

    if isinstance(tensor_or_scalar_like, Tensor):
        tensor = tensor_or_scalar_like
    else:
        try:
            tensor = torch.as_tensor(tensor_or_scalar_like)
        except Exception:
            error_meta = _TestingErrorMeta(
                ValueError, f"No tensor can be constructed from type {type(tensor_or_scalar_like)}."
            )
            return error_meta, None

    error_meta = _check_supported_tensor(tensor)
    if error_meta:
        return error_meta, None

    return None, tensor


def _check_types(actual: Any, expected: Any, *, allow_subclasses: bool) -> Optional[_TestingErrorMeta]:
    # We exclude numbers here, since numbers of different type, e.g. int vs. float, should be treated the same as
    # tensors with different dtypes. Without user input, passing numbers of different types will still fail, but this
    # can be disabled by setting `check_dtype=False`.
    if isinstance(actual, numbers.Number) and isinstance(expected, numbers.Number):
        return None

    msg_fmtstr = f"Except for Python scalars, {{}}, but got {type(actual)} and {type(expected)} instead."
    directly_related = isinstance(actual, type(expected)) or isinstance(expected, type(actual))
    if not directly_related:
        return _TestingErrorMeta(AssertionError, msg_fmtstr.format("input types need to be directly related"))

    if allow_subclasses or type(actual) is type(expected):
        return None

    return _TestingErrorMeta(AssertionError, msg_fmtstr.format("type equality is required if allow_subclasses=False"))


def _to_tensor_pair(
    actual: Any, expected: Any, *, allow_subclasses: bool
) -> Tuple[Optional[_TestingErrorMeta], Optional[_TensorPair]]:
    """Converts a tensor-or-scalar-like pair to a :class:`_TensorPair`.

    Args:
        actual (Any): Actual tensor-or-scalar-like.
        expected (Any): Expected tensor-or-scalar-like.
        allow_subclasses (bool): If ``True`` (default) and except for Python scalars, inputs of directly related types
            are allowed. Otherwise type equality is required.

    Returns:
        (Optional[_TestingErrorMeta], Optional[_TensorPair]): The two elements are orthogonal, i.e. if the first is
            ``None`` the second will not and vice versa. Returns :class:`_TestingErrorMeta` if :attr:`actual` and
            :attr:`expected` are not scalars and do not have the same type. Additionally, returns any error meta from
            :func:`_to_tensor`.
    """
    error_meta = _check_types(actual, expected, allow_subclasses=allow_subclasses)
    if error_meta:
        return error_meta, None

    error_meta, actual = _to_tensor(actual)
    if error_meta:
        return error_meta, None

    error_meta, expected = _to_tensor(expected)
    if error_meta:
        return error_meta, None

    return None, _TensorPair(actual, expected)


def _parse_inputs(
    actual: Any, expected: Any, *, allow_subclasses: bool
) -> Tuple[Optional[_TestingErrorMeta], Optional[Union[_TensorPair, List, Dict]]]:
    """Parses the positional inputs by constructing :class:`_TensorPair`'s from corresponding tensor-or-scalar-likes.


    :class:`~collections.abc.Sequence`'s or :class:`~collections.abc.Mapping`'s are parsed elementwise. Parsing is
    performed recursively and thus nested containers are supported. The hierarchy of the containers is preserved, but
    sequences are returned as :class:`list` and mappings as :class:`dict`.

    Args:
        actual (Any): Actual input.
        expected (Any): Expected input.
        allow_subclasses (bool): If ``True`` (default) and except for Python scalars, inputs of directly related types
            are allowed. Otherwise type equality is required.

    Returns:
        (Tuple[Optional[_TestingErrorMeta], Optional[Union[_TensorPair, List, Dict]]]): The two elements are
            orthogonal, i.e. if the first is ``None`` the second will be valid and vice versa. Returns
            :class:`_TestingErrorMeta` if the length of two sequences or the keys of two mappings do not match.
            Additionally, returns any error meta from :func:`_to_tensor_pair`.

    """
    error_meta: Optional[_TestingErrorMeta]

    # We explicitly exclude str's here since they are self-referential and would cause an infinite recursion loop:
    # "a" == "a"[0][0]...
    if (
        isinstance(actual, collections.abc.Sequence)
        and not isinstance(actual, str)
        and isinstance(expected, collections.abc.Sequence)
        and not isinstance(expected, str)
    ):
        actual_len = len(actual)
        expected_len = len(expected)
        if actual_len != expected_len:
            error_meta = _TestingErrorMeta(
                AssertionError, f"The length of the sequences mismatch: {actual_len} != {expected_len}"
            )
            return error_meta, None

        pair_list = []
        for idx in range(actual_len):
            error_meta, pair = _parse_inputs(actual[idx], expected[idx], allow_subclasses=allow_subclasses)
            if error_meta:
                error_meta = error_meta.amend_msg(postfix=f"\n\n{_SEQUENCE_MSG_FMTSTR.format(idx)}")
                return error_meta, None

            pair_list.append(pair)
        else:
            return None, pair_list

    elif isinstance(actual, collections.abc.Mapping) and isinstance(expected, collections.abc.Mapping):
        actual_keys = set(actual.keys())
        expected_keys = set(expected.keys())
        if actual_keys != expected_keys:
            missing_keys = expected_keys - actual_keys
            additional_keys = actual_keys - expected_keys
            error_meta = _TestingErrorMeta(
                AssertionError,
                f"The keys of the mappings do not match:\n"
                f"Missing keys in the actual mapping: {sorted(missing_keys)}\n"
                f"Additional keys in the actual mapping: {sorted(additional_keys)}",
            )
            return error_meta, None

        pair_dict = {}
        for key in sorted(actual_keys):
            error_meta, pair = _parse_inputs(actual[key], expected[key], allow_subclasses=allow_subclasses)
            if error_meta:
                error_meta = error_meta.amend_msg(postfix=f"\n\n{_MAPPING_MSG_FMTSTR.format(key)}")
                return error_meta, None

            pair_dict[key] = pair
        else:
            return None, pair_dict

    else:
        return _to_tensor_pair(actual, expected, allow_subclasses=allow_subclasses)


def assert_close(
    actual: Any,
    expected: Any,
    *,
    allow_subclasses: bool = True,
    rtol: Optional[float] = None,
    atol: Optional[float] = None,
    equal_nan: Union[bool, str] = False,
    check_device: bool = True,
    check_dtype: bool = True,
    check_stride: bool = False,
    check_is_coalesced: bool = True,
    msg: Optional[Union[str, Callable[[Tensor, Tensor, Diagnostics], str]]] = None,
) -> None:
    r"""Asserts that :attr:`actual` and :attr:`expected` are close.

    If :attr:`actual` and :attr:`expected` are strided, non-quantized, real-valued, and finite, they are considered
    close if

    .. math::

        \lvert \text{actual} - \text{expected} \rvert \le \texttt{atol} + \texttt{rtol} \cdot \lvert \text{expected} \rvert

    and they have the same :attr:`~torch.Tensor.device` (if :attr:`check_device` is ``True``), same ``dtype`` (if
    :attr:`check_dtype` is ``True``), and the same stride (if :attr:`check_stride` is ``True``). Non-finite values
    (``-inf`` and ``inf``) are only considered close if and only if they are equal. ``NaN``'s are only considered equal
    to each other if :attr:`equal_nan` is ``True``.

    If :attr:`actual` and :attr:`expected` are complex-valued, they are considered close if both their real and
    imaginary components are considered close according to the definition above.

    If :attr:`actual` and :attr:`expected` are sparse (either having COO or CSR layout), their strided members are
    checked individually. Indices, namely ``indices`` for COO or ``crow_indices``  and ``col_indices`` for CSR layout,
    are always checked for equality whereas the values are checked for closeness according to the definition above.
    Sparse COO tensors are only considered close if both are either coalesced or uncoalesced (if
    :attr:`check_is_coalesced` is ``True``).

    If :attr:`actual` and :attr:`expected` are quantized, they are considered close if they have the same
    :meth:`~torch.Tensor.qscheme` and the result of :meth:`~torch.Tensor.dequantize` is close according to the
    definition above.

    :attr:`actual` and :attr:`expected` can be :class:`~torch.Tensor`'s or any tensor-or-scalar-likes from which
    :class:`torch.Tensor`'s can be constructed with :func:`torch.as_tensor`. Except for Python scalars the input types
    have to be directly related. In addition, :attr:`actual` and :attr:`expected` can be
    :class:`~collections.abc.Sequence`'s or :class:`~collections.abc.Mapping`'s in which case they are considered close
    if their structure matches and all their elements are considered close according to the above definition.

    .. note::

        Python scalars are an exception to the type relation requirement, because their :func:`type`, i.e.
        :class:`int`, :class:`float`, and :class:`complex`, is equivalent to the ``dtype`` of a tensor-like. Thus,
        Python scalars of different types can be checked, but require :attr:`check_dtype` to be set to ``False``.

    Args:
        actual (Any): Actual input.
        expected (Any): Expected input.
        allow_subclasses (bool): If ``True`` (default) and except for Python scalars, inputs of directly related types
            are allowed. Otherwise type equality is required.
        rtol (Optional[float]): Relative tolerance. If specified :attr:`atol` must also be specified. If omitted,
            default values based on the :attr:`~torch.Tensor.dtype` are selected with the below table.
        atol (Optional[float]): Absolute tolerance. If specified :attr:`rtol` must also be specified. If omitted,
            default values based on the :attr:`~torch.Tensor.dtype` are selected with the below table.
        equal_nan (Union[bool, str]): If ``True``, two ``NaN`` values will be considered equal. If ``"relaxed"``,
            complex values are considered as ``NaN`` if either the real **or** imaginary component is ``NaN``.
        check_device (bool): If ``True`` (default), asserts that corresponding tensors are on the same
            :attr:`~torch.Tensor.device`. If this check is disabled, tensors on different
            :attr:`~torch.Tensor.device`'s are moved to the CPU before being compared.
        check_dtype (bool): If ``True`` (default), asserts that corresponding tensors have the same ``dtype``. If this
            check is disabled, tensors with different ``dtype``'s are promoted  to a common ``dtype`` (according to
            :func:`torch.promote_types`) before being compared.
        check_stride (bool): If ``True`` and corresponding tensors are strided, asserts that they have the same stride.
        check_is_coalesced (bool): If ``True`` (default) and corresponding tensors are sparse COO, checks that both
            :attr:`actual` and :attr:`expected` are either coalesced or uncoalesced. If this check is disabled,
            tensors are :meth:`~torch.Tensor.coalesce`'ed before being compared.
        msg (Optional[Union[str, Callable[[Tensor, Tensor, Diagnostics], str]]]): Optional error message to use if the
            values of corresponding tensors mismatch. Can be passed as callable in which case it will be called with
            the mismatching tensors and a namespace of diagnostics about the mismatches. See below for details.

    Raises:
        ValueError: If no :class:`torch.Tensor` can be constructed from an input.
        ValueError: If only :attr:`rtol` or :attr:`atol` is specified.
        AssertionError: If corresponding inputs are not Python scalars and are not directly related.
        AssertionError: If :attr:`allow_subclasses` is ``False``, but corresponding inputs are not Python scalars and
            have different types.
        AssertionError: If the inputs are :class:`~collections.abc.Sequence`'s, but their length does not match.
        AssertionError: If the inputs are :class:`~collections.abc.Mapping`'s, but their set of keys do not match.
        AssertionError: If corresponding tensors do not have the same :attr:`~torch.Tensor.shape`.
        AssertionError: If corresponding tensors do not have the same :attr:`~torch.Tensor.layout`.
        AssertionError: If corresponding tensors are quantized, but have different :meth:`~torch.Tensor.qscheme`'s.
        AssertionError: If :attr:`check_device` is ``True``, but corresponding tensors are not on the same
            :attr:`~torch.Tensor.device`.
        AssertionError: If :attr:`check_dtype` is ``True``, but corresponding tensors do not have the same ``dtype``.
        AssertionError: If :attr:`check_stride` is ``True``, but corresponding strided tensors do not have the same
            stride.
        AssertionError: If :attr:`check_is_coalesced`  is ``True``, but corresponding sparse COO tensors are not both
            either coalesced or uncoalesced.
        AssertionError: If the values of corresponding tensors are not close according to the definition above.

    The following table displays the default ``rtol`` and ``atol`` for different ``dtype``'s. In case of mismatching
    ``dtype``'s, the maximum of both tolerances is used.

    +---------------------------+------------+----------+
    | ``dtype``                 | ``rtol``   | ``atol`` |
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

    The namespace of diagnostics that will be passed to :attr:`msg` if its a callable has the following attributes:

    - ``number_of_elements`` (int): Number of elements in each tensor being compared.
    - ``total_mismatches`` (int): Total number of mismatches.
    - ``max_abs_diff`` (Union[int, float]): Greatest absolute difference of the inputs.
    - ``max_abs_diff_idx`` (Union[int, Tuple[int, ...]]): Index of greatest absolute difference.
    - ``atol`` (float): Allowed absolute tolerance.
    - ``max_rel_diff`` (Union[int, float]): Greatest relative difference of the inputs.
    - ``max_rel_diff_idx`` (Union[int, Tuple[int, ...]]): Index of greatest relative difference.
    - ``rtol`` (float): Allowed relative tolerance.

    For ``max_abs_diff`` and ``max_rel_diff`` the type depends on the :attr:`~torch.Tensor.dtype` of the inputs.

    .. note::

        :func:`~torch.testing.assert_close` is highly configurable with strict default settings. Users are encouraged
        to :func:`~functools.partial` it to fit their use case. For example, if an equality check is needed, one might
        define an ``assert_equal`` that uses zero tolrances for every ``dtype`` by default:

        >>> import functools
        >>> assert_equal = functools.partial(torch.testing.assert_close, rtol=0, atol=0)
        >>> assert_equal(1e-9, 1e-10)
        Traceback (most recent call last):
        ...
        AssertionError: Scalars are not equal!
        <BLANKLINE>
        Absolute difference: 8.999999703829253e-10
        Relative difference: 8.999999583666371

    Examples:
        >>> # tensor to tensor comparison
        >>> expected = torch.tensor([1e0, 1e-1, 1e-2])
        >>> actual = torch.acos(torch.cos(expected))
        >>> torch.testing.assert_close(actual, expected)

        >>> # scalar to scalar comparison
        >>> import math
        >>> expected = math.sqrt(2.0)
        >>> actual = 2.0 / math.sqrt(2.0)
        >>> torch.testing.assert_close(actual, expected)

        >>> # numpy array to numpy array comparison
        >>> import numpy as np
        >>> expected = np.array([1e0, 1e-1, 1e-2])
        >>> actual = np.arccos(np.cos(expected))
        >>> torch.testing.assert_close(actual, expected)

        >>> # sequence to sequence comparison
        >>> import numpy as np
        >>> # The types of the sequences do not have to match. They only have to have the same
        >>> # length and their elements have to match.
        >>> expected = [torch.tensor([1.0]), 2.0, np.array(3.0)]
        >>> actual = tuple(expected)
        >>> torch.testing.assert_close(actual, expected)

        >>> # mapping to mapping comparison
        >>> from collections import OrderedDict
        >>> import numpy as np
        >>> foo = torch.tensor(1.0)
        >>> bar = 2.0
        >>> baz = np.array(3.0)
        >>> # The types and a possible ordering of mappings do not have to match. They only
        >>> # have to have the same set of keys and their elements have to match.
        >>> expected = OrderedDict([("foo", foo), ("bar", bar), ("baz", baz)])
        >>> actual = {"baz": baz, "bar": bar, "foo": foo}
        >>> torch.testing.assert_close(actual, expected)

        >>> expected = torch.tensor([1.0, 2.0, 3.0])
        >>> actual = expected.clone()
        >>> # By default, directly related instances can be compared
        >>> torch.testing.assert_close(torch.nn.Parameter(actual), expected)
        >>> # This check can be made more strict with allow_subclasses=False
        >>> torch.testing.assert_close(
        ...     torch.nn.Parameter(actual), expected, allow_subclasses=False
        ... )
        Traceback (most recent call last):
        ...
        AssertionError: Except for Python scalars, type equality is required if
        allow_subclasses=False, but got <class 'torch.nn.parameter.Parameter'> and
        <class 'torch.Tensor'> instead.
        >>> # If the inputs are not directly related, they are never considered close
        >>> torch.testing.assert_close(actual.numpy(), expected)
        Traceback (most recent call last):
        ...
        AssertionError: Except for Python scalars, input types need to be directly
        related, but got <class 'numpy.ndarray'> and <class 'torch.Tensor'> instead.
        >>> # Exceptions to these rules are Python scalars. They can be checked regardless of
        >>> # their type if check_dtype=False.
        >>> torch.testing.assert_close(1.0, 1, check_dtype=False)

        >>> # NaN != NaN by default.
        >>> expected = torch.tensor(float("Nan"))
        >>> actual = expected.clone()
        >>> torch.testing.assert_close(actual, expected)
        Traceback (most recent call last):
        ...
        AssertionError: Scalars are not close!
        <BLANKLINE>
        Absolute difference: nan (up to 1e-05 allowed)
        Relative difference: nan (up to 1.3e-06 allowed)
        >>> torch.testing.assert_close(actual, expected, equal_nan=True)

        >>> # If equal_nan=True, the real and imaginary NaN's of complex inputs have to match.
        >>> expected = torch.tensor(complex(float("NaN"), 0))
        >>> actual = torch.tensor(complex(0, float("NaN")))
        >>> torch.testing.assert_close(actual, expected, equal_nan=True)
        Traceback (most recent call last):
        ...
        AssertionError: Scalars are not close!
        <BLANKLINE>
        Absolute difference: nan (up to 1e-05 allowed)
        Relative difference: nan (up to 1.3e-06 allowed)
        >>> # If equal_nan="relaxed", however, then complex numbers are treated as NaN if any
        >>> # of the real or imaginary components is NaN.
        >>> torch.testing.assert_close(actual, expected, equal_nan="relaxed")

        >>> expected = torch.tensor([1.0, 2.0, 3.0])
        >>> actual = torch.tensor([1.0, 4.0, 5.0])
        >>> # The default mismatch message can be overwritten.
        >>> torch.testing.assert_close(actual, expected, msg="Argh, the tensors are not close!")
        Traceback (most recent call last):
        ...
        AssertionError: Argh, the tensors are not close!
        >>> # The error message can also created at runtime by passing a callable.
        >>> def custom_msg(actual, expected, diagnostics):
        ...     ratio = diagnostics.total_mismatches / diagnostics.number_of_elements
        ...     return (
        ...         f"Argh, we found {diagnostics.total_mismatches} mismatches! "
        ...         f"That is {ratio:.1%}!"
        ...     )
        >>> torch.testing.assert_close(actual, expected, msg=custom_msg)
        Traceback (most recent call last):
        ...
        AssertionError: Argh, we found 2 mismatches! That is 66.7%!
    """
    # Hide this function from `pytest`'s traceback
    __tracebackhide__ = True

    if (rtol is None) ^ (atol is None):
        # We require both tolerance to be omitted or specified, because specifying only one might lead to surprising
        # results. Imagine setting atol=0.0 and the tensors still match because rtol>0.0.
        raise ValueError(
            f"Both 'rtol' and 'atol' must be either specified or omitted, "
            f"but got no {'rtol' if rtol is None else 'atol'}.",
        )

    error_meta, pair = _parse_inputs(actual, expected, allow_subclasses=allow_subclasses)
    if error_meta:
        raise error_meta.to_error()
    else:
        pair = cast(Union[_TensorPair, List, Dict], pair)

    error_meta = _check_pair_close(
        pair,
        rtol=rtol,
        atol=atol,
        equal_nan=equal_nan,
        check_device=check_device,
        check_dtype=check_dtype,
        check_stride=check_stride,
        check_is_coalesced=check_is_coalesced,
        msg=msg,
    )
    if error_meta:
        raise error_meta.to_error()
