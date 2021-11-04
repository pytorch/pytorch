import abc
import cmath
import collections.abc
from typing import NamedTuple, Callable, Sequence, List, Union, Optional, Type, Tuple, Any, cast

import torch

from ._core import _unravel_index


class ErrorMeta(NamedTuple):
    type: Type[Exception]
    msg: str
    id: Tuple[Any, ...] = ()

    def to_error(self) -> Exception:
        msg = self.msg
        if self.id:
            msg += f"\n\nThe failure occurred for item {''.join(str([item]) for item in self.id)}"
        return self.type(msg)


_DTYPE_PRECISIONS = {
    torch.float16: (0.001, 1e-5),
    torch.bfloat16: (0.016, 1e-5),
    torch.float32: (1.3e-6, 1e-5),
    torch.float64: (1e-7, 1e-7),
    torch.complex32: (0.001, 1e-5),
    torch.complex64: (1.3e-6, 1e-5),
    torch.complex128: (1e-7, 1e-7),
}


def default_tolerances(*inputs: Union[torch.Tensor, torch.dtype]) -> Tuple[float, float]:
    dtypes = []
    for input in inputs:
        if isinstance(input, torch.Tensor):
            dtypes.append(input.dtype)
        elif isinstance(input, torch.dtype):
            dtypes.append(input)
        else:
            raise TypeError()
    rtols, atols = zip(*[_DTYPE_PRECISIONS.get(dtype, (0.0, 0.0)) for dtype in dtypes])
    return max(rtols), max(atols)


def parse_tolerances(
    *inputs: Union[torch.Tensor, torch.dtype], rtol: Optional[float], atol: Optional[float]
) -> Tuple[Optional[ErrorMeta], Optional[Tuple[float, float]]]:
    if (rtol is None) ^ (atol is None):
        # We require both tolerance to be omitted or specified, because specifying only one might lead to surprising
        # results. Imagine setting atol=0.0 and the tensors still match because rtol>0.0.
        error_meta = ErrorMeta(
            ValueError,
            f"Both 'rtol' and 'atol' must be either specified or omitted, "
            f"but got no {'rtol' if rtol is None else 'atol'}.",
        )
        return error_meta, None
    elif rtol is not None and atol is not None:
        return None, (rtol, atol)
    else:
        return None, default_tolerances(*inputs)


def _make_mismatch_msg(
    *,
    identifier: Optional[Union[str, Callable[[str], str]]],
    default_identifier: str,
    extra: Optional[str] = None,
    abs_diff: float,
    abs_diff_idx: Optional[Union[int, Tuple[int, ...]]] = None,
    atol: float,
    rel_diff: float,
    rel_diff_idx: Optional[Union[int, Tuple[int, ...]]] = None,
    rtol: float,
) -> str:
    equality = rtol == 0 and atol == 0

    def make_diff_msg(*, type: str, diff: float, idx: Optional[Union[int, Tuple[int, ...]]], tol: float) -> str:
        if idx is None:
            msg = f"{type.title()} difference: {diff}"
        else:
            msg = f"Greatest {type} difference: {diff} at index {idx}"
        if not equality:
            msg += f" (up to {tol} allowed)"
        return msg + "\n"

    if identifier is None:
        identifier = default_identifier
    elif callable(identifier):
        identifier = identifier(default_identifier)

    msg = f"{identifier} are not {'equal' if equality else 'close'}!\n\n"

    if extra:
        msg += f"{extra.strip()}\n"

    msg += make_diff_msg(type="absolute", diff=abs_diff, idx=abs_diff_idx, tol=atol)
    msg += make_diff_msg(type="relative", diff=rel_diff, idx=rel_diff_idx, tol=rtol)

    return msg


def make_scalar_mismatch_msg(
    actual: Union[int, float, complex],
    expected: Union[int, float, complex],
    *,
    rtol: float,
    atol: float,
    identifier: Optional[Union[str, Callable[[str], str]]] = None,
) -> str:
    abs_diff = abs(actual - expected)
    rel_diff = abs_diff / abs(expected)
    return _make_mismatch_msg(
        identifier=identifier,
        default_identifier="Scalars",
        abs_diff=abs_diff,
        atol=atol,
        rel_diff=rel_diff,
        rtol=rtol,
    )


def make_tensor_mismatch_msg(
    actual: torch.Tensor,
    expected: torch.Tensor,
    mismatches: torch.Tensor,
    *,
    rtol: float,
    atol: float,
    identifier: Optional[Union[str, Callable[[str], str]]] = None,
):
    number_of_elements = mismatches.numel()
    total_mismatches = torch.sum(mismatches).item()
    extra = (
        f"Mismatched elements: {total_mismatches} / {number_of_elements} "
        f"({total_mismatches / number_of_elements:.1%})"
    )

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
    return _make_mismatch_msg(
        identifier=identifier,
        default_identifier="Tensor-likes",
        extra=extra,
        abs_diff=max_abs_diff.item(),
        abs_diff_idx=_unravel_index(max_abs_diff_flat_idx.item(), mismatches.shape),
        atol=atol,
        rel_diff=max_rel_diff.item(),
        rel_diff_idx=_unravel_index(max_rel_diff_flat_idx.item(), mismatches.shape),
        rtol=rtol,
    )


class UnsupportedInputs(Exception):  # noqa: B903
    def __init__(self, meta: Optional[ErrorMeta] = None) -> None:
        self.meta = meta


class Pair(abc.ABC):
    def __init__(
        self,
        actual: Any,
        expected: Any,
        *,
        id: Tuple[Any, ...] = (),
        msg: Optional[str] = None,
        **unknown_parameters: Any,
    ) -> None:
        self.actual = actual
        self.expected = expected
        self.id = id
        self.msg = msg
        self._unknown_parameters = unknown_parameters

    @staticmethod
    def _check_inputs_isinstance(*inputs: Any, cls: Union[Type, Tuple[Type, ...]]):
        if not all(isinstance(input, cls) for input in inputs):
            raise UnsupportedInputs()

    @staticmethod
    def _apply_unary(
        fn: Callable[[Any], Union[Any, Tuple[Optional[ErrorMeta], Any]]],
        *inputs: Any,
        id: Tuple[Any, ...] = (),
    ) -> Tuple[Optional[ErrorMeta], Optional[Tuple[Any, ...]]]:
        outputs = []
        for input in inputs:
            output: Any = fn(input)
            if isinstance(output, ErrorMeta) or output is None:
                error_meta = output
                output = []
            else:
                error_meta, *output = output
            if error_meta:
                return error_meta._replace(id=id) if id else error_meta, None  # type: ignore[misc]

            outputs.append(output[0] if len(output) == 1 else output)
        return None, tuple(outputs) if outputs else None

    def _make_error_meta(self, type: Type[Exception], msg: str) -> ErrorMeta:
        return ErrorMeta(type, self.msg or msg, self.id)

    @abc.abstractmethod
    def compare(self) -> Optional[ErrorMeta]:
        pass


class ObjectPair(Pair):
    def compare(self) -> Optional[ErrorMeta]:
        try:
            equal = self.actual == self.expected
        except Exception as error:
            return ErrorMeta(
                ValueError,
                f"Comparing {self.actual} and {self.expected} for equality failed with:\n{error}.",
            )

        if equal:
            return None

        return self._make_error_meta(AssertionError, f"{self.actual} != {self.expected}")


class NonePair(Pair):
    def __init__(self, actual: Any, expected: Any, **other_parameters: Any) -> None:
        if not (actual is None and expected is None):
            raise UnsupportedInputs()

        super().__init__(actual, expected, **other_parameters)

    def compare(self) -> Optional[ErrorMeta]:
        # At instantiation we already checked that both actual and expected are None, so there is nothing left to do
        # here
        return None


class BooleanPair(Pair):
    def __init__(self, actual: Any, expected: Any, **other_parameters: Any) -> None:
        self._check_inputs_isinstance(actual, expected, cls=bool)
        super().__init__(actual, expected, **other_parameters)

    def compare(self) -> Optional[ErrorMeta]:
        if self.actual is self.expected:
            return None

        return self._make_error_meta(AssertionError, f"Booleans mismatch: {self.actual} is not {self.expected}")


class NumberPair(Pair):
    _TYPE_TO_DTYPE = {
        int: torch.int64,
        float: torch.float64,
        complex: torch.complex128,
    }

    def __init__(
        self,
        actual: Any,
        expected: Any,
        *,
        id: Tuple[Any, ...] = (),
        rtol: Optional[float] = None,
        atol: Optional[float] = None,
        equal_nan: bool = False,
        check_dtype: bool = False,
        **other_parameters: Any,
    ) -> None:
        self._check_inputs_isinstance(actual, expected, cls=tuple(self._TYPE_TO_DTYPE.keys()))
        error_meta, tolerances = parse_tolerances(
            *[self._TYPE_TO_DTYPE.get(type(input), torch.float64) for input in (actual, expected)], rtol=rtol, atol=atol
        )
        if error_meta:
            raise UnsupportedInputs(error_meta._replace(id=id))
        self.rtol, self.atol = cast(Tuple[float, float], tolerances)

        super().__init__(actual, expected, id=id, **other_parameters)
        self.equal_nan = equal_nan
        self.check_dtype = check_dtype

    def compare(self) -> Optional[ErrorMeta]:
        if self.check_dtype and type(self.actual) is not type(self.expected):
            return self._make_error_meta(
                AssertionError,
                f"The (d)types do not match: {type(self.actual)} != {type(self.expected)}.",
            )

        if cmath.isnan(self.actual) and cmath.isnan(self.expected) and self.equal_nan:
            return None

        diff = abs(self.actual - self.expected)
        tolerance = self.atol + self.rtol * abs(self.expected)

        if diff <= tolerance:
            return None

        return self._make_error_meta(
            AssertionError, make_scalar_mismatch_msg(self.actual, self.expected, rtol=self.rtol, atol=self.atol)
        )


class TensorLikePair(Pair):
    def __init__(
        self,
        actual: Any,
        expected: Any,
        *,
        id: Tuple[Any, ...] = (),
        allow_subclasses: bool = True,
        rtol: Optional[float] = None,
        atol: Optional[float] = None,
        equal_nan: bool = False,
        check_device: bool = True,
        check_dtype: bool = True,
        check_layout: bool = True,
        check_stride: bool = False,
        check_is_coalesced: bool = True,
        **other_parameters: Any,
    ):
        actual, expected = self._check_supported_inputs(actual, expected, id=id, allow_subclasses=allow_subclasses)

        error_meta, tolerances = parse_tolerances(actual, expected, rtol=rtol, atol=atol)
        if error_meta:
            raise UnsupportedInputs(error_meta._replace(id=id))
        self.rtol, self.atol = cast(Tuple[float, float], tolerances)

        super().__init__(actual, expected, id=id, **other_parameters)

        self.equal_nan = equal_nan
        self.check_device = check_device
        self.check_dtype = check_dtype
        self.check_layout = check_layout
        self.check_stride = check_stride
        self.check_is_coalesced = check_is_coalesced

    def _check_supported_inputs(
        self, actual: Any, expected: Any, *, id: Tuple[Any, ...], allow_subclasses: bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        directly_related = isinstance(actual, type(expected)) or isinstance(expected, type(actual))
        if not directly_related:
            raise UnsupportedInputs()

        if not allow_subclasses and type(actual) is not type(expected):
            raise UnsupportedInputs()

        error_meta, tensors = self._apply_unary(self._to_tensor, actual, expected, id=id)
        if error_meta:
            raise UnsupportedInputs(error_meta)
        actual, expected = cast(Tuple[torch.Tensor, torch.Tensor], tensors)

        error_meta, _ = self._apply_unary(self._check_supported, actual, expected, id=id)
        if error_meta:
            raise UnsupportedInputs(error_meta)

        return actual, expected

    def _to_tensor(self, tensor_like: Any) -> Tuple[Optional[ErrorMeta], Optional[torch.Tensor]]:
        if isinstance(tensor_like, torch.Tensor):
            tensor = tensor_like
        else:
            try:
                tensor = torch.as_tensor(tensor_like)
            except Exception as error:
                error_meta = ErrorMeta(
                    ValueError,
                    f"Constructing a tensor from {type(tensor_like)} failed with \n{error}.",
                )
                return error_meta, None

        return None, tensor

    def _check_supported(self, tensor: torch.Tensor) -> Optional[ErrorMeta]:
        if tensor.layout not in {torch.strided, torch.sparse_coo, torch.sparse_csr}:  # type: ignore[attr-defined]
            return ErrorMeta(ValueError, f"Unsupported tensor layout {tensor.layout}")

        return None

    def compare(self) -> Optional[ErrorMeta]:
        actual, expected = self.actual, self.expected

        error_meta = self._compare_attributes(actual, expected)
        if error_meta:
            return error_meta

        actual, expected = self._equalize_attributes(actual, expected)
        error_meta = self._compare_values(actual, expected)
        if error_meta:
            return error_meta

        return None

    def _compare_attributes(
        self,
        actual: torch.Tensor,
        expected: torch.Tensor,
    ) -> Optional[ErrorMeta]:
        def error_meta(attribute_name: str, actual_value: Any, expected_value: Any) -> ErrorMeta:
            return self._make_error_meta(
                AssertionError,
                f"The values for attribute '{attribute_name}' do not match: {actual_value} != {expected_value}.",
            )

        if actual.shape != expected.shape:
            return error_meta("shape", actual.shape, expected.shape)

        if actual.layout != expected.layout:
            if self.check_layout:
                return error_meta("layout", actual.layout, expected.layout)
        else:
            if actual.layout == torch.strided and self.check_stride and actual.stride() != expected.stride():
                return error_meta("stride()", actual.stride(), expected.stride())
            elif (
                actual.layout == torch.sparse_coo
                and self.check_is_coalesced
                and actual.is_coalesced() != expected.is_coalesced()
            ):
                return error_meta("is_coalesced()", actual.is_coalesced(), expected.is_coalesced())

        if self.check_device and actual.device != expected.device:
            return error_meta("device", actual.device, expected.device)

        if actual.is_quantized != expected.is_quantized:
            return error_meta("is_quantized", actual.is_quantized, expected.is_quantized)
        elif actual.is_quantized and actual.qscheme() != expected.qscheme():
            return error_meta("qscheme()", actual.qscheme(), expected.qscheme())

        if self.check_dtype and actual.dtype != expected.dtype:
            return error_meta("dtype", actual.dtype, expected.dtype)

        return None

    def _equalize_attributes(self, actual: torch.Tensor, expected: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Equalizes some attributes of two tensors for value comparison.

        If ``actual`` and ``expected`` are ...

        - ... not on the same :attr:`~torch.Tensor.device`, they are moved CPU memory.
        - ... not of the same ``dtype``, they are promoted  to a common ``dtype`` (according to
            :func:`torch.promote_types`).
        - ... not of the same ``layout``, they are converted to strided tensors.
        - ... both sparse COO tensors but only one is coalesced, the other one is coalesced.

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

        if actual.layout != expected.layout:
            # These checks are needed, since Tensor.to_dense() fails on tensors that are already strided
            actual = actual.to_dense() if actual.layout != torch.strided else actual
            expected = expected.to_dense() if expected.layout != torch.strided else expected
        elif actual.is_sparse and actual.is_coalesced() != expected.is_coalesced():
            actual = actual.coalesce()
            expected = expected.coalesce()

        return actual, expected

    def _compare_values(self, actual: torch.Tensor, expected: torch.Tensor) -> Optional[ErrorMeta]:
        if actual.is_quantized:
            compare_fn = self._compare_quantized_values
        elif actual.is_sparse:
            compare_fn = self._compare_sparse_coo_values
        elif actual.is_sparse_csr:
            compare_fn = self._compare_sparse_csr_values
        else:
            compare_fn = self._compare_regular_values

        return compare_fn(actual, expected, rtol=self.rtol, atol=self.atol, equal_nan=self.equal_nan)

    def _compare_quantized_values(
        self, actual: torch.Tensor, expected: torch.Tensor, *, rtol: float, atol: float, equal_nan: bool
    ) -> Optional[ErrorMeta]:
        return self._compare_regular_values(
            actual.dequantize(), expected.dequantize(), rtol=rtol, atol=atol, equal_nan=equal_nan
        )

    def _compare_sparse_coo_values(
        self, actual: torch.Tensor, expected: torch.Tensor, *, rtol: float, atol: float, equal_nan: bool
    ) -> Optional[ErrorMeta]:
        if actual._nnz() != expected._nnz():
            return self._make_error_meta(
                AssertionError,
                (
                    f"The number of specified values in sparse COO tensors does not match: "
                    f"{actual._nnz()} != {expected._nnz()}"
                ),
            )

        error_meta = self._compare_regular_values(
            actual._indices(),
            expected._indices(),
            rtol=0,
            atol=0,
            equal_nan=False,
            identifier="Sparse COO indices",
        )
        if error_meta:
            return error_meta

        return self._compare_regular_values(
            actual._values(),
            expected._values(),
            rtol=rtol,
            atol=atol,
            equal_nan=equal_nan,
            identifier="Sparse COO values",
        )

    def _compare_sparse_csr_values(
        self, actual: torch.Tensor, expected: torch.Tensor, *, rtol: float, atol: float, equal_nan: bool
    ) -> Optional[ErrorMeta]:
        if actual._nnz() != expected._nnz():
            return self._make_error_meta(
                AssertionError,
                (
                    f"The number of specified values in sparse CSR tensors does not match: "
                    f"{actual._nnz()} != {expected._nnz()}"
                ),
            )

        error_meta = self._compare_regular_values(
            actual.crow_indices(),
            expected.crow_indices(),
            rtol=0,
            atol=0,
            equal_nan=False,
            identifier="Sparse CSR crow_indices",
        )
        if error_meta:
            return error_meta

        error_meta = self._compare_regular_values(
            actual.col_indices(),
            expected.col_indices(),
            rtol=0,
            atol=0,
            equal_nan=False,
            identifier="Sparse CSR col_indices",
        )
        if error_meta:
            return error_meta

        return self._compare_regular_values(
            actual.values(),
            expected.values(),
            rtol=rtol,
            atol=atol,
            equal_nan=equal_nan,
            identifier="Sparse CSR values",
        )

    def _compare_regular_values(
        self,
        actual: torch.Tensor,
        expected: torch.Tensor,
        *,
        rtol: float,
        atol: float,
        equal_nan: bool,
        identifier: Optional[Union[str, Callable[[str], str]]] = None,
    ) -> Optional[ErrorMeta]:
        actual, expected = TensorLikePair._promote_for_comparison(actual, expected)
        mismatches = ~torch.isclose(actual, expected, rtol=rtol, atol=atol, equal_nan=equal_nan)
        if not torch.any(mismatches):
            return None

        if actual.shape == torch.Size([]):
            msg = make_scalar_mismatch_msg(actual.item(), expected.item(), rtol=rtol, atol=atol, identifier=identifier)
        else:
            msg = make_tensor_mismatch_msg(actual, expected, mismatches, rtol=rtol, atol=atol, identifier=identifier)
        return self._make_error_meta(AssertionError, msg)

    @staticmethod
    def _promote_for_comparison(actual: torch.Tensor, expected: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Selects the comparison dtype based on the input dtype.

        Returns:
            Highest precision dtype of the same dtype category as the input. :class:`torch.bool` is treated as integral
            dtype.
        """
        # This is called after self._equalize_attributes() and thus `actual` and `expected` already have the same dtype.
        if actual.dtype.is_complex:
            dtype = torch.complex128
        elif actual.dtype.is_floating_point:
            dtype = torch.float64
        else:
            dtype = torch.int64
        return actual.to(dtype), expected.to(dtype)


def originate_pairs(
    actual: Any,
    expected: Any,
    *,
    pair_types: Sequence[Type[Pair]],
    id: Tuple[Any, ...] = (),
    **options: Any,
) -> Tuple[Optional[ErrorMeta], Optional[List[Pair]]]:
    error_meta: Optional[ErrorMeta]
    pairs: List[Pair] = []
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
            error_meta = ErrorMeta(
                AssertionError, f"The length of the sequences mismatch: {actual_len} != {expected_len}", id
            )
            return error_meta, None

        for idx in range(actual_len):
            error_meta, partial_pairs = originate_pairs(
                actual[idx], expected[idx], pair_types=pair_types, id=(*id, idx), **options
            )
            if error_meta:
                return error_meta, None

            pairs.extend(cast(List[Pair], partial_pairs))

        return None, pairs

    elif isinstance(actual, collections.abc.Mapping) and isinstance(expected, collections.abc.Mapping):
        actual_keys = set(actual.keys())
        expected_keys = set(expected.keys())
        if actual_keys != expected_keys:
            missing_keys = expected_keys - actual_keys
            additional_keys = actual_keys - expected_keys
            error_meta = ErrorMeta(
                AssertionError,
                (
                    f"The keys of the mappings do not match:\n"
                    f"Missing keys in the actual mapping: {sorted(missing_keys)}\n"
                    f"Additional keys in the actual mapping: {sorted(additional_keys)}"
                ),
                id,
            )
            return error_meta, None

        for key in sorted(actual_keys):
            error_meta, partial_pairs = originate_pairs(
                actual[key], expected[key], pair_types=pair_types, id=(*id, key), **options
            )
            if error_meta:
                return error_meta, None

            pairs.extend(cast(List[Pair], partial_pairs))

        return None, pairs

    else:
        for pair_type in pair_types:
            try:
                return None, [pair_type(actual, expected, id=id, **options)]
            except UnsupportedInputs as error:
                # In case we encounter an error during origination, we abort
                if error.meta:
                    return error.meta, None

        else:
            return (
                ErrorMeta(
                    TypeError,
                    f"No comparison pair was able to handle inputs of type {type(actual)} and {type(expected)}.",
                    id=id,
                ),
                None,
            )


def assert_equal(
    actual: Any, expected: Any, *, pair_types: Sequence[Type[Pair]] = (ObjectPair,), **options: Any
) -> None:
    # Hide this function from `pytest`'s traceback
    __tracebackhide__ = True

    error_meta, pairs = originate_pairs(actual, expected, pair_types=pair_types, **options)
    if error_meta:
        raise error_meta.to_error()

    error_metas: List[ErrorMeta] = []
    for pair in cast(List[Pair], pairs):
        error_meta = pair.compare()
        if error_meta:
            error_metas.append(error_meta)

    if not error_metas:
        return

    # TODO: compose all metas into one AssertionError
    raise error_metas[0].to_error()


def assert_close(
    actual: Any,
    expected: Any,
    *,
    allow_subclasses: bool = True,
    rtol: Optional[float] = None,
    atol: Optional[float] = None,
    equal_nan: bool = False,
    check_device: bool = True,
    check_dtype: bool = True,
    check_layout: bool = True,
    check_stride: bool = False,
    check_is_coalesced: bool = True,
    msg: Optional[str] = None,
):
    r"""Asserts that ``actual`` and ``expected`` are close.

    If ``actual`` and ``expected`` are strided, non-quantized, real-valued, and finite, they are considered close if

    .. math::

        \lvert \text{actual} - \text{expected} \rvert \le \texttt{atol} + \texttt{rtol} \cdot \lvert \text{expected} \rvert

    and they have the same :attr:`~torch.Tensor.device` (if ``check_device`` is ``True``), same ``dtype`` (if
    ``check_dtype`` is ``True``), and the same stride (if ``check_stride`` is ``True``). Non-finite values
    (``-inf`` and ``inf``) are only considered close if and only if they are equal. ``NaN``'s are only considered equal
    to each other if ``equal_nan`` is ``True``.

    If ``actual`` and ``expected`` are sparse (either having COO or CSR layout), their strided members are
    checked individually. Indices, namely ``indices`` for COO or ``crow_indices``  and ``col_indices`` for CSR layout,
    are always checked for equality whereas the values are checked for closeness according to the definition above.
    Sparse COO tensors are only considered close if both are either coalesced or uncoalesced (if
    ``check_is_coalesced`` is ``True``).

    If ``actual`` and ``expected`` are quantized, they are considered close if they have the same
    :meth:`~torch.Tensor.qscheme` and the result of :meth:`~torch.Tensor.dequantize` is close according to the
    definition above.

    ``actual`` and ``expected`` can be :class:`~torch.Tensor`'s or any tensor-or-scalar-likes from which
    :class:`torch.Tensor`'s can be constructed with :func:`torch.as_tensor`. Except for Python scalars the input types
    have to be directly related. In addition, ``actual`` and ``expected`` can be :class:`~collections.abc.Sequence`'s
    or :class:`~collections.abc.Mapping`'s in which case they are considered close if their structure matches and all
    their elements are considered close according to the above definition.

    .. note::

        Python scalars are an exception to the type relation requirement, because their :func:`type`, i.e.
        :class:`int`, :class:`float`, and :class:`complex`, is equivalent to the ``dtype`` of a tensor-like. Thus,
        Python scalars of different types can be checked, but require ``check_dtype=False``.

    Args:
        actual (Any): Actual input.
        expected (Any): Expected input.
        allow_subclasses (bool): If ``True`` (default) and except for Python scalars, inputs of directly related types
            are allowed. Otherwise type equality is required.
        rtol (Optional[float]): Relative tolerance. If specified ``atol`` must also be specified. If omitted, default
            values based on the :attr:`~torch.Tensor.dtype` are selected with the below table.
        atol (Optional[float]): Absolute tolerance. If specified ``rtol`` must also be specified. If omitted, default
            values based on the :attr:`~torch.Tensor.dtype` are selected with the below table.
        equal_nan (Union[bool, str]): If ``True``, two ``NaN`` values will be considered equal.
        check_device (bool): If ``True`` (default), asserts that corresponding tensors are on the same
            :attr:`~torch.Tensor.device`. If this check is disabled, tensors on different
            :attr:`~torch.Tensor.device`'s are moved to the CPU before being compared.
        check_dtype (bool): If ``True`` (default), asserts that corresponding tensors have the same ``dtype``. If this
            check is disabled, tensors with different ``dtype``'s are promoted  to a common ``dtype`` (according to
            :func:`torch.promote_types`) before being compared.
        check_layout (bool): If ``True`` (default), asserts that corresponding tensors have the same ``layout``. If this
            check is disabled, tensors with different ``layout``'s are converted to strided tensors before being
            compared.
        check_stride (bool): If ``True`` and corresponding tensors are strided, asserts that they have the same stride.
        check_is_coalesced (bool): If ``True`` (default) and corresponding tensors are sparse COO, checks that both
            ``actual`` and ``expected`` are either coalesced or uncoalesced. If this check is disabled, tensors are
            :meth:`~torch.Tensor.coalesce`'ed before being compared.
        msg (Optional[str]): Optional error message to use in case a failure occurs during the comparison.

    Raises:
        ValueError: If no :class:`torch.Tensor` can be constructed from an input.
        ValueError: If only ``rtol`` or ``atol`` is specified.
        AssertionError: If corresponding inputs are not Python scalars and are not directly related.
        AssertionError: If ``allow_subclasses`` is ``False``, but corresponding inputs are not Python scalars and have
            different types.
        AssertionError: If the inputs are :class:`~collections.abc.Sequence`'s, but their length does not match.
        AssertionError: If the inputs are :class:`~collections.abc.Mapping`'s, but their set of keys do not match.
        AssertionError: If corresponding tensors do not have the same :attr:`~torch.Tensor.shape`.
        AssertionError: If ``check_layout`` is ``True``, but corresponding tensors do not have the same
            :attr:`~torch.Tensor.layout`.
        AssertionError: If corresponding tensors are quantized, but have different :meth:`~torch.Tensor.qscheme`'s.
        AssertionError: If ``check_device`` is ``True``, but corresponding tensors are not on the same
            :attr:`~torch.Tensor.device`.
        AssertionError: If ``check_dtype`` is ``True``, but corresponding tensors do not have the same ``dtype``.
        AssertionError: If ``check_stride`` is ``True``, but corresponding strided tensors do not have the same stride.
        AssertionError: If ``check_is_coalesced``  is ``True``, but corresponding sparse COO tensors are not both
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
        Absolute difference: 9.000000000000001e-10
        Relative difference: 9.0

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
        TypeError: No comparison pair was able to handle inputs of type
        <class 'torch.nn.parameter.Parameter'> and <class 'torch.Tensor'>.
        >>> # If the inputs are not directly related, they are never considered close
        >>> torch.testing.assert_close(actual.numpy(), expected)
        Traceback (most recent call last):
        ...
        TypeError: No comparison pair was able to handle inputs of type <class 'numpy.ndarray'>
        and <class 'torch.Tensor'>.
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

        >>> expected = torch.tensor([1.0, 2.0, 3.0])
        >>> actual = torch.tensor([1.0, 4.0, 5.0])
        >>> # The default error message can be overwritten.
        >>> torch.testing.assert_close(actual, expected, msg="Argh, the tensors are not close!")
        Traceback (most recent call last):
        ...
        AssertionError: Argh, the tensors are not close!
    """
    # Hide this function from `pytest`'s traceback
    __tracebackhide__ = True

    assert_equal(
        actual,
        expected,
        pair_types=(
            NonePair,
            BooleanPair,
            NumberPair,
            TensorLikePair,
        ),
        allow_subclasses=allow_subclasses,
        rtol=rtol,
        atol=atol,
        equal_nan=equal_nan,
        check_device=check_device,
        check_dtype=check_dtype,
        check_layout=check_layout,
        check_stride=check_stride,
        check_is_coalesced=check_is_coalesced,
        msg=msg,
    )
