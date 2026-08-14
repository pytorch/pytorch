r"""Operations over lists of tensors.

Each function applies the corresponding ordinary PyTorch operation to every
position in one or more tensor lists. This API is in beta and may change based
on user feedback.

The functions may use an accelerated multi-tensor implementation when their
inputs meet its requirements. Otherwise they use a semantically equivalent
per-tensor fallback. Calling a function in this module does not guarantee a
single or fused kernel.
"""

from typing import TYPE_CHECKING

from torch._C import _add_docstr, _foreach  # type: ignore[attr-defined]


__all__ = [
    "abs",
    "abs_",
    "acos",
    "acos_",
    "add",
    "add_",
    "addcdiv",
    "addcdiv_",
    "addcmul",
    "addcmul_",
    "asin",
    "asin_",
    "atan",
    "atan_",
    "ceil",
    "ceil_",
    "clamp_max",
    "clamp_max_",
    "clamp_min",
    "clamp_min_",
    "clone",
    "copy_",
    "cos",
    "cos_",
    "cosh",
    "cosh_",
    "div",
    "div_",
    "erf",
    "erf_",
    "erfc",
    "erfc_",
    "exp",
    "exp_",
    "expm1",
    "expm1_",
    "floor",
    "floor_",
    "frac",
    "frac_",
    "lerp",
    "lerp_",
    "lgamma",
    "lgamma_",
    "log",
    "log10",
    "log10_",
    "log1p",
    "log1p_",
    "log2",
    "log2_",
    "log_",
    "max",
    "maximum",
    "maximum_",
    "minimum",
    "minimum_",
    "mm",
    "mul",
    "mul_",
    "neg",
    "neg_",
    "norm",
    "pow",
    "pow_",
    "reciprocal",
    "reciprocal_",
    "round",
    "round_",
    "rsqrt",
    "rsqrt_",
    "sigmoid",
    "sigmoid_",
    "sign",
    "sign_",
    "sin",
    "sin_",
    "sinh",
    "sinh_",
    "sqrt",
    "sqrt_",
    "sub",
    "sub_",
    "tan",
    "tan_",
    "tanh",
    "tanh_",
    "trunc",
    "trunc_",
    "zero_",
]


def _result(in_place: bool) -> str:
    if in_place:
        return "the exact input list or tuple"
    return "a tuple containing one result tensor for each input tensor"


def _common(reference: str, in_place: bool) -> str:
    mutation = (
        "Mutates every tensor in ``inputs`` and returns the exact input "
        "container object."
        if in_place
        else "Does not mutate ``inputs`` and returns a tuple of result tensors."
    )
    return rf"""
This is semantically equivalent to applying :func:`{reference}` independently
at every list position. {mutation}

The input tensor lists must be non-empty. Corresponding tensor or scalar lists
must have the same length. An accelerated multi-tensor implementation is used
only when supported by the inputs; otherwise the operation falls back to
per-tensor execution.
"""


def _unary_doc(name: str, reference: str, *, inplace: bool) -> str:
    suffix = "tuple[Tensor, ...] | list[Tensor]" if inplace else "tuple[Tensor, ...]"
    return rf"""
{name}(inputs) -> {suffix}

Applies :func:`{reference}` to each tensor in ``inputs``.

{_common(reference, inplace)}

Args:
    inputs (list or tuple of Tensor): tensors to transform.

Returns:
    {_result(inplace)}.
"""


def _binary_doc(
    name: str,
    reference: str,
    forms: str,
    *,
    inplace: bool,
    alpha: str | None = None,
    operand: str = "other",
    shared_tensor: bool = False,
) -> str:
    suffix = "tuple[Tensor, ...] | list[Tensor]" if inplace else "tuple[Tensor, ...]"
    alpha_sig = ", *, alpha=1" if alpha is not None else ""
    alpha_doc = "" if alpha is None else f"\n    alpha (Number, optional): {alpha}"
    operand_type = "Number, list or tuple of Number, or list or tuple of Tensor"
    if shared_tensor:
        operand_type = f"{operand_type}, or Tensor"
    return rf"""
{name}(inputs, {operand}{alpha_sig}) -> {suffix}

Applies :func:`{reference}` to every tensor in ``inputs``.

{_common(reference, inplace)}

Supported forms for ``{operand}``: {forms}

Args:
    inputs (list or tuple of Tensor): tensors to transform.
    {operand} ({operand_type}): operand shared across positions or supplied per
        position.{alpha_doc}

Returns:
    {_result(inplace)}.
"""


def _pointwise_doc(name: str, reference: str, *, inplace: bool) -> str:
    suffix = "tuple[Tensor, ...] | list[Tensor]" if inplace else "tuple[Tensor, ...]"
    return rf"""
{name}(inputs, tensor1s, tensor2s, *, value=1) -> {suffix}

Applies :func:`{reference}` to corresponding tensors from the three input
lists.

{_common(reference, inplace)}

``value`` may be one shared scalar, a scalar list or tuple, or a packed 1-D
CPU tensor containing one scalar per list position.

Args:
    inputs (list or tuple of Tensor): tensors to transform.
    tensor1s (list or tuple of Tensor): first multiplicative or divisive operands.
    tensor2s (list or tuple of Tensor): second multiplicative or divisive operands.
    value (Number, list or tuple of Number, or Tensor, optional): scale values.
        Default: ``1``.

Returns:
    {_result(inplace)}.
"""


if TYPE_CHECKING:
    from typing import overload, TypeVar

    from torch import memory_format, Tensor
    from torch.types import _bool, _complex, _dtype, Number, PySymType

    _ScalarT = TypeVar("_ScalarT", bound=Number | _complex | PySymType)
    _ScalarList = tuple[_ScalarT, ...] | list[_ScalarT]

    def abs(inputs: tuple[Tensor, ...] | list[Tensor]) -> tuple[Tensor, ...]: ...
    def abs_(inputs: tuple[Tensor, ...] | list[Tensor]) -> tuple[Tensor, ...] | list[Tensor]: ...
    def acos(inputs: tuple[Tensor, ...] | list[Tensor]) -> tuple[Tensor, ...]: ...
    def acos_(inputs: tuple[Tensor, ...] | list[Tensor]) -> tuple[Tensor, ...] | list[Tensor]: ...
    @overload
    def add(
        inputs: tuple[Tensor, ...] | list[Tensor],
        other: Number | _complex | PySymType,
    ) -> tuple[Tensor, ...]: ...
    @overload
    def add(
        inputs: tuple[Tensor, ...] | list[Tensor],
        other: _ScalarList[_ScalarT],
    ) -> tuple[Tensor, ...]: ...
    @overload
    def add(
        inputs: tuple[Tensor, ...] | list[Tensor],
        other: Tensor,
        *,
        alpha: Number | _complex | PySymType = 1,
    ) -> tuple[Tensor, ...]: ...
    @overload
    def add(
        inputs: tuple[Tensor, ...] | list[Tensor],
        other: tuple[Tensor, ...] | list[Tensor],
        *,
        alpha: Number | _complex | PySymType = 1,
    ) -> tuple[Tensor, ...]: ...
    def add(*args: object, **kwargs: object) -> tuple[Tensor, ...]: ...
    @overload
    def add_(
        inputs: tuple[Tensor, ...] | list[Tensor],
        other: Number | _complex | PySymType,
    ) -> tuple[Tensor, ...] | list[Tensor]: ...
    @overload
    def add_(
        inputs: tuple[Tensor, ...] | list[Tensor],
        other: _ScalarList[_ScalarT],
    ) -> tuple[Tensor, ...] | list[Tensor]: ...
    @overload
    def add_(
        inputs: tuple[Tensor, ...] | list[Tensor],
        other: Tensor,
        *,
        alpha: Number | _complex | PySymType = 1,
    ) -> tuple[Tensor, ...] | list[Tensor]: ...
    @overload
    def add_(
        inputs: tuple[Tensor, ...] | list[Tensor],
        other: tuple[Tensor, ...] | list[Tensor],
        *,
        alpha: Number | _complex | PySymType = 1,
    ) -> tuple[Tensor, ...] | list[Tensor]: ...
    def add_(*args: object, **kwargs: object) -> tuple[Tensor, ...] | list[Tensor]: ...
    @overload
    def addcdiv(
        inputs: tuple[Tensor, ...] | list[Tensor],
        tensor1s: tuple[Tensor, ...] | list[Tensor],
        tensor2s: tuple[Tensor, ...] | list[Tensor],
        *,
        value: _ScalarList[_ScalarT],
    ) -> tuple[Tensor, ...]: ...
    @overload
    def addcdiv(
        inputs: tuple[Tensor, ...] | list[Tensor],
        tensor1s: tuple[Tensor, ...] | list[Tensor],
        tensor2s: tuple[Tensor, ...] | list[Tensor],
        *,
        value: Tensor,
    ) -> tuple[Tensor, ...]: ...
    @overload
    def addcdiv(
        inputs: tuple[Tensor, ...] | list[Tensor],
        tensor1s: tuple[Tensor, ...] | list[Tensor],
        tensor2s: tuple[Tensor, ...] | list[Tensor],
        *,
        value: Number | _complex | PySymType = 1,
    ) -> tuple[Tensor, ...]: ...
    def addcdiv(*args: object, **kwargs: object) -> tuple[Tensor, ...]: ...
    @overload
    def addcdiv_(
        inputs: tuple[Tensor, ...] | list[Tensor],
        tensor1s: tuple[Tensor, ...] | list[Tensor],
        tensor2s: tuple[Tensor, ...] | list[Tensor],
        *,
        value: _ScalarList[_ScalarT],
    ) -> tuple[Tensor, ...] | list[Tensor]: ...
    @overload
    def addcdiv_(
        inputs: tuple[Tensor, ...] | list[Tensor],
        tensor1s: tuple[Tensor, ...] | list[Tensor],
        tensor2s: tuple[Tensor, ...] | list[Tensor],
        *,
        value: Tensor,
    ) -> tuple[Tensor, ...] | list[Tensor]: ...
    @overload
    def addcdiv_(
        inputs: tuple[Tensor, ...] | list[Tensor],
        tensor1s: tuple[Tensor, ...] | list[Tensor],
        tensor2s: tuple[Tensor, ...] | list[Tensor],
        *,
        value: Number | _complex | PySymType = 1,
    ) -> tuple[Tensor, ...] | list[Tensor]: ...
    def addcdiv_(*args: object, **kwargs: object) -> tuple[Tensor, ...] | list[Tensor]: ...
    @overload
    def addcmul(
        inputs: tuple[Tensor, ...] | list[Tensor],
        tensor1s: tuple[Tensor, ...] | list[Tensor],
        tensor2s: tuple[Tensor, ...] | list[Tensor],
        *,
        value: _ScalarList[_ScalarT],
    ) -> tuple[Tensor, ...]: ...
    @overload
    def addcmul(
        inputs: tuple[Tensor, ...] | list[Tensor],
        tensor1s: tuple[Tensor, ...] | list[Tensor],
        tensor2s: tuple[Tensor, ...] | list[Tensor],
        *,
        value: Tensor,
    ) -> tuple[Tensor, ...]: ...
    @overload
    def addcmul(
        inputs: tuple[Tensor, ...] | list[Tensor],
        tensor1s: tuple[Tensor, ...] | list[Tensor],
        tensor2s: tuple[Tensor, ...] | list[Tensor],
        *,
        value: Number | _complex | PySymType = 1,
    ) -> tuple[Tensor, ...]: ...
    def addcmul(*args: object, **kwargs: object) -> tuple[Tensor, ...]: ...
    @overload
    def addcmul_(
        inputs: tuple[Tensor, ...] | list[Tensor],
        tensor1s: tuple[Tensor, ...] | list[Tensor],
        tensor2s: tuple[Tensor, ...] | list[Tensor],
        *,
        value: _ScalarList[_ScalarT],
    ) -> tuple[Tensor, ...] | list[Tensor]: ...
    @overload
    def addcmul_(
        inputs: tuple[Tensor, ...] | list[Tensor],
        tensor1s: tuple[Tensor, ...] | list[Tensor],
        tensor2s: tuple[Tensor, ...] | list[Tensor],
        *,
        value: Tensor,
    ) -> tuple[Tensor, ...] | list[Tensor]: ...
    @overload
    def addcmul_(
        inputs: tuple[Tensor, ...] | list[Tensor],
        tensor1s: tuple[Tensor, ...] | list[Tensor],
        tensor2s: tuple[Tensor, ...] | list[Tensor],
        *,
        value: Number | _complex | PySymType = 1,
    ) -> tuple[Tensor, ...] | list[Tensor]: ...
    def addcmul_(*args: object, **kwargs: object) -> tuple[Tensor, ...] | list[Tensor]: ...
    def asin(inputs: tuple[Tensor, ...] | list[Tensor]) -> tuple[Tensor, ...]: ...
    def asin_(inputs: tuple[Tensor, ...] | list[Tensor]) -> tuple[Tensor, ...] | list[Tensor]: ...
    def atan(inputs: tuple[Tensor, ...] | list[Tensor]) -> tuple[Tensor, ...]: ...
    def atan_(inputs: tuple[Tensor, ...] | list[Tensor]) -> tuple[Tensor, ...] | list[Tensor]: ...
    def ceil(inputs: tuple[Tensor, ...] | list[Tensor]) -> tuple[Tensor, ...]: ...
    def ceil_(inputs: tuple[Tensor, ...] | list[Tensor]) -> tuple[Tensor, ...] | list[Tensor]: ...
    @overload
    def clamp_max(
        inputs: tuple[Tensor, ...] | list[Tensor],
        max: Number | _complex | PySymType,
    ) -> tuple[Tensor, ...]: ...
    @overload
    def clamp_max(
        inputs: tuple[Tensor, ...] | list[Tensor],
        max: _ScalarList[_ScalarT],
    ) -> tuple[Tensor, ...]: ...
    @overload
    def clamp_max(
        inputs: tuple[Tensor, ...] | list[Tensor],
        max: tuple[Tensor, ...] | list[Tensor],
    ) -> tuple[Tensor, ...]: ...
    def clamp_max(*args: object, **kwargs: object) -> tuple[Tensor, ...]: ...
    @overload
    def clamp_max_(
        inputs: tuple[Tensor, ...] | list[Tensor],
        max: Number | _complex | PySymType,
    ) -> tuple[Tensor, ...] | list[Tensor]: ...
    @overload
    def clamp_max_(
        inputs: tuple[Tensor, ...] | list[Tensor],
        max: _ScalarList[_ScalarT],
    ) -> tuple[Tensor, ...] | list[Tensor]: ...
    @overload
    def clamp_max_(
        inputs: tuple[Tensor, ...] | list[Tensor],
        max: tuple[Tensor, ...] | list[Tensor],
    ) -> tuple[Tensor, ...] | list[Tensor]: ...
    def clamp_max_(*args: object, **kwargs: object) -> tuple[Tensor, ...] | list[Tensor]: ...
    @overload
    def clamp_min(
        inputs: tuple[Tensor, ...] | list[Tensor],
        min: Number | _complex | PySymType,
    ) -> tuple[Tensor, ...]: ...
    @overload
    def clamp_min(
        inputs: tuple[Tensor, ...] | list[Tensor],
        min: _ScalarList[_ScalarT],
    ) -> tuple[Tensor, ...]: ...
    @overload
    def clamp_min(
        inputs: tuple[Tensor, ...] | list[Tensor],
        min: tuple[Tensor, ...] | list[Tensor],
    ) -> tuple[Tensor, ...]: ...
    def clamp_min(*args: object, **kwargs: object) -> tuple[Tensor, ...]: ...
    @overload
    def clamp_min_(
        inputs: tuple[Tensor, ...] | list[Tensor],
        min: Number | _complex | PySymType,
    ) -> tuple[Tensor, ...] | list[Tensor]: ...
    @overload
    def clamp_min_(
        inputs: tuple[Tensor, ...] | list[Tensor],
        min: _ScalarList[_ScalarT],
    ) -> tuple[Tensor, ...] | list[Tensor]: ...
    @overload
    def clamp_min_(
        inputs: tuple[Tensor, ...] | list[Tensor],
        min: tuple[Tensor, ...] | list[Tensor],
    ) -> tuple[Tensor, ...] | list[Tensor]: ...
    def clamp_min_(*args: object, **kwargs: object) -> tuple[Tensor, ...] | list[Tensor]: ...
    def clone(
        inputs: tuple[Tensor, ...] | list[Tensor],
        *,
        memory_format: memory_format | None = None,
    ) -> tuple[Tensor, ...]: ...
    def copy_(
        inputs: tuple[Tensor, ...] | list[Tensor],
        srcs: tuple[Tensor, ...] | list[Tensor],
        non_blocking: _bool = False,
    ) -> tuple[Tensor, ...] | list[Tensor]: ...
    def cos(inputs: tuple[Tensor, ...] | list[Tensor]) -> tuple[Tensor, ...]: ...
    def cos_(inputs: tuple[Tensor, ...] | list[Tensor]) -> tuple[Tensor, ...] | list[Tensor]: ...
    def cosh(inputs: tuple[Tensor, ...] | list[Tensor]) -> tuple[Tensor, ...]: ...
    def cosh_(inputs: tuple[Tensor, ...] | list[Tensor]) -> tuple[Tensor, ...] | list[Tensor]: ...
    @overload
    def div(
        inputs: tuple[Tensor, ...] | list[Tensor],
        other: _ScalarList[_ScalarT],
    ) -> tuple[Tensor, ...]: ...
    @overload
    def div(
        inputs: tuple[Tensor, ...] | list[Tensor],
        other: Tensor,
    ) -> tuple[Tensor, ...]: ...
    @overload
    def div(
        inputs: tuple[Tensor, ...] | list[Tensor],
        other: tuple[Tensor, ...] | list[Tensor],
    ) -> tuple[Tensor, ...]: ...
    @overload
    def div(
        inputs: tuple[Tensor, ...] | list[Tensor],
        other: Number | _complex | PySymType,
    ) -> tuple[Tensor, ...]: ...
    def div(*args: object, **kwargs: object) -> tuple[Tensor, ...]: ...
    @overload
    def div_(
        inputs: tuple[Tensor, ...] | list[Tensor],
        other: _ScalarList[_ScalarT],
    ) -> tuple[Tensor, ...] | list[Tensor]: ...
    @overload
    def div_(inputs: tuple[Tensor, ...] | list[Tensor], other: Tensor) -> tuple[Tensor, ...] | list[Tensor]: ...
    @overload
    def div_(
        inputs: tuple[Tensor, ...] | list[Tensor],
        other: tuple[Tensor, ...] | list[Tensor],
    ) -> tuple[Tensor, ...] | list[Tensor]: ...
    @overload
    def div_(
        inputs: tuple[Tensor, ...] | list[Tensor],
        other: Number | _complex | PySymType,
    ) -> tuple[Tensor, ...] | list[Tensor]: ...
    def div_(*args: object, **kwargs: object) -> tuple[Tensor, ...] | list[Tensor]: ...
    def erf(inputs: tuple[Tensor, ...] | list[Tensor]) -> tuple[Tensor, ...]: ...
    def erf_(inputs: tuple[Tensor, ...] | list[Tensor]) -> tuple[Tensor, ...] | list[Tensor]: ...
    def erfc(inputs: tuple[Tensor, ...] | list[Tensor]) -> tuple[Tensor, ...]: ...
    def erfc_(inputs: tuple[Tensor, ...] | list[Tensor]) -> tuple[Tensor, ...] | list[Tensor]: ...
    def exp(inputs: tuple[Tensor, ...] | list[Tensor]) -> tuple[Tensor, ...]: ...
    def exp_(inputs: tuple[Tensor, ...] | list[Tensor]) -> tuple[Tensor, ...] | list[Tensor]: ...
    def expm1(inputs: tuple[Tensor, ...] | list[Tensor]) -> tuple[Tensor, ...]: ...
    def expm1_(inputs: tuple[Tensor, ...] | list[Tensor]) -> tuple[Tensor, ...] | list[Tensor]: ...
    def floor(inputs: tuple[Tensor, ...] | list[Tensor]) -> tuple[Tensor, ...]: ...
    def floor_(inputs: tuple[Tensor, ...] | list[Tensor]) -> tuple[Tensor, ...] | list[Tensor]: ...
    def frac(inputs: tuple[Tensor, ...] | list[Tensor]) -> tuple[Tensor, ...]: ...
    def frac_(inputs: tuple[Tensor, ...] | list[Tensor]) -> tuple[Tensor, ...] | list[Tensor]: ...
    @overload
    def lerp(
        inputs: tuple[Tensor, ...] | list[Tensor],
        ends: tuple[Tensor, ...] | list[Tensor],
        weight: Number | _complex | PySymType,
    ) -> tuple[Tensor, ...]: ...
    @overload
    def lerp(
        inputs: tuple[Tensor, ...] | list[Tensor],
        ends: tuple[Tensor, ...] | list[Tensor],
        weight: _ScalarList[_ScalarT],
    ) -> tuple[Tensor, ...]: ...
    @overload
    def lerp(
        inputs: tuple[Tensor, ...] | list[Tensor],
        ends: tuple[Tensor, ...] | list[Tensor],
        weight: tuple[Tensor, ...] | list[Tensor],
    ) -> tuple[Tensor, ...]: ...
    def lerp(*args: object, **kwargs: object) -> tuple[Tensor, ...]: ...
    @overload
    def lerp_(
        inputs: tuple[Tensor, ...] | list[Tensor],
        ends: tuple[Tensor, ...] | list[Tensor],
        weight: Number | _complex | PySymType,
    ) -> tuple[Tensor, ...] | list[Tensor]: ...
    @overload
    def lerp_(
        inputs: tuple[Tensor, ...] | list[Tensor],
        ends: tuple[Tensor, ...] | list[Tensor],
        weight: _ScalarList[_ScalarT],
    ) -> tuple[Tensor, ...] | list[Tensor]: ...
    @overload
    def lerp_(
        inputs: tuple[Tensor, ...] | list[Tensor],
        ends: tuple[Tensor, ...] | list[Tensor],
        weight: tuple[Tensor, ...] | list[Tensor],
    ) -> tuple[Tensor, ...] | list[Tensor]: ...
    def lerp_(*args: object, **kwargs: object) -> tuple[Tensor, ...] | list[Tensor]: ...
    def lgamma(inputs: tuple[Tensor, ...] | list[Tensor]) -> tuple[Tensor, ...]: ...
    def lgamma_(inputs: tuple[Tensor, ...] | list[Tensor]) -> tuple[Tensor, ...] | list[Tensor]: ...
    def log(inputs: tuple[Tensor, ...] | list[Tensor]) -> tuple[Tensor, ...]: ...
    def log10(inputs: tuple[Tensor, ...] | list[Tensor]) -> tuple[Tensor, ...]: ...
    def log10_(inputs: tuple[Tensor, ...] | list[Tensor]) -> tuple[Tensor, ...] | list[Tensor]: ...
    def log1p(inputs: tuple[Tensor, ...] | list[Tensor]) -> tuple[Tensor, ...]: ...
    def log1p_(inputs: tuple[Tensor, ...] | list[Tensor]) -> tuple[Tensor, ...] | list[Tensor]: ...
    def log2(inputs: tuple[Tensor, ...] | list[Tensor]) -> tuple[Tensor, ...]: ...
    def log2_(inputs: tuple[Tensor, ...] | list[Tensor]) -> tuple[Tensor, ...] | list[Tensor]: ...
    def log_(inputs: tuple[Tensor, ...] | list[Tensor]) -> tuple[Tensor, ...] | list[Tensor]: ...
    def max(inputs: tuple[Tensor, ...] | list[Tensor]) -> tuple[Tensor, ...]: ...
    @overload
    def maximum(
        inputs: tuple[Tensor, ...] | list[Tensor],
        other: Number | _complex | PySymType,
    ) -> tuple[Tensor, ...]: ...
    @overload
    def maximum(
        inputs: tuple[Tensor, ...] | list[Tensor],
        other: _ScalarList[_ScalarT],
    ) -> tuple[Tensor, ...]: ...
    @overload
    def maximum(
        inputs: tuple[Tensor, ...] | list[Tensor],
        other: tuple[Tensor, ...] | list[Tensor],
    ) -> tuple[Tensor, ...]: ...
    def maximum(*args: object, **kwargs: object) -> tuple[Tensor, ...]: ...
    @overload
    def maximum_(
        inputs: tuple[Tensor, ...] | list[Tensor],
        other: Number | _complex | PySymType,
    ) -> tuple[Tensor, ...] | list[Tensor]: ...
    @overload
    def maximum_(
        inputs: tuple[Tensor, ...] | list[Tensor],
        other: _ScalarList[_ScalarT],
    ) -> tuple[Tensor, ...] | list[Tensor]: ...
    @overload
    def maximum_(
        inputs: tuple[Tensor, ...] | list[Tensor],
        other: tuple[Tensor, ...] | list[Tensor],
    ) -> tuple[Tensor, ...] | list[Tensor]: ...
    def maximum_(*args: object, **kwargs: object) -> tuple[Tensor, ...] | list[Tensor]: ...
    @overload
    def minimum(
        inputs: tuple[Tensor, ...] | list[Tensor],
        other: Number | _complex | PySymType,
    ) -> tuple[Tensor, ...]: ...
    @overload
    def minimum(
        inputs: tuple[Tensor, ...] | list[Tensor],
        other: _ScalarList[_ScalarT],
    ) -> tuple[Tensor, ...]: ...
    @overload
    def minimum(
        inputs: tuple[Tensor, ...] | list[Tensor],
        other: tuple[Tensor, ...] | list[Tensor],
    ) -> tuple[Tensor, ...]: ...
    def minimum(*args: object, **kwargs: object) -> tuple[Tensor, ...]: ...
    @overload
    def minimum_(
        inputs: tuple[Tensor, ...] | list[Tensor],
        other: Number | _complex | PySymType,
    ) -> tuple[Tensor, ...] | list[Tensor]: ...
    @overload
    def minimum_(
        inputs: tuple[Tensor, ...] | list[Tensor],
        other: _ScalarList[_ScalarT],
    ) -> tuple[Tensor, ...] | list[Tensor]: ...
    @overload
    def minimum_(
        inputs: tuple[Tensor, ...] | list[Tensor],
        other: tuple[Tensor, ...] | list[Tensor],
    ) -> tuple[Tensor, ...] | list[Tensor]: ...
    def minimum_(*args: object, **kwargs: object) -> tuple[Tensor, ...] | list[Tensor]: ...
    def mm(
        inputs: tuple[Tensor, ...] | list[Tensor],
        mat2s: tuple[Tensor, ...] | list[Tensor],
    ) -> tuple[Tensor, ...]: ...
    @overload
    def mul(
        inputs: tuple[Tensor, ...] | list[Tensor],
        other: _ScalarList[_ScalarT],
    ) -> tuple[Tensor, ...]: ...
    @overload
    def mul(
        inputs: tuple[Tensor, ...] | list[Tensor],
        other: Tensor,
    ) -> tuple[Tensor, ...]: ...
    @overload
    def mul(
        inputs: tuple[Tensor, ...] | list[Tensor],
        other: tuple[Tensor, ...] | list[Tensor],
    ) -> tuple[Tensor, ...]: ...
    @overload
    def mul(
        inputs: tuple[Tensor, ...] | list[Tensor],
        other: Number | _complex | PySymType,
    ) -> tuple[Tensor, ...]: ...
    def mul(*args: object, **kwargs: object) -> tuple[Tensor, ...]: ...
    @overload
    def mul_(
        inputs: tuple[Tensor, ...] | list[Tensor],
        other: _ScalarList[_ScalarT],
    ) -> tuple[Tensor, ...] | list[Tensor]: ...
    @overload
    def mul_(inputs: tuple[Tensor, ...] | list[Tensor], other: Tensor) -> tuple[Tensor, ...] | list[Tensor]: ...
    @overload
    def mul_(
        inputs: tuple[Tensor, ...] | list[Tensor],
        other: tuple[Tensor, ...] | list[Tensor],
    ) -> tuple[Tensor, ...] | list[Tensor]: ...
    @overload
    def mul_(
        inputs: tuple[Tensor, ...] | list[Tensor],
        other: Number | _complex | PySymType,
    ) -> tuple[Tensor, ...] | list[Tensor]: ...
    def mul_(*args: object, **kwargs: object) -> tuple[Tensor, ...] | list[Tensor]: ...
    def neg(inputs: tuple[Tensor, ...] | list[Tensor]) -> tuple[Tensor, ...]: ...
    def neg_(inputs: tuple[Tensor, ...] | list[Tensor]) -> tuple[Tensor, ...] | list[Tensor]: ...
    def norm(
        inputs: tuple[Tensor, ...] | list[Tensor],
        ord: Number | _complex | PySymType = 2,
        *,
        dtype: _dtype | None = None,
    ) -> tuple[Tensor, ...]: ...
    @overload
    def pow(
        input: Number | _complex | PySymType,
        exponent: tuple[Tensor, ...] | list[Tensor],
    ) -> tuple[Tensor, ...]: ...
    @overload
    def pow(
        input: tuple[Tensor, ...] | list[Tensor],
        exponent: Number | _complex | PySymType,
    ) -> tuple[Tensor, ...]: ...
    @overload
    def pow(
        input: tuple[Tensor, ...] | list[Tensor],
        exponent: _ScalarList[_ScalarT],
    ) -> tuple[Tensor, ...]: ...
    @overload
    def pow(
        input: tuple[Tensor, ...] | list[Tensor],
        exponent: tuple[Tensor, ...] | list[Tensor],
    ) -> tuple[Tensor, ...]: ...
    def pow(*args: object, **kwargs: object) -> tuple[Tensor, ...]: ...
    @overload
    def pow_(
        inputs: tuple[Tensor, ...] | list[Tensor],
        exponent: Number | _complex | PySymType,
    ) -> tuple[Tensor, ...] | list[Tensor]: ...
    @overload
    def pow_(
        inputs: tuple[Tensor, ...] | list[Tensor],
        exponent: _ScalarList[_ScalarT],
    ) -> tuple[Tensor, ...] | list[Tensor]: ...
    @overload
    def pow_(
        inputs: tuple[Tensor, ...] | list[Tensor],
        exponent: tuple[Tensor, ...] | list[Tensor],
    ) -> tuple[Tensor, ...] | list[Tensor]: ...
    def pow_(*args: object, **kwargs: object) -> tuple[Tensor, ...] | list[Tensor]: ...
    def reciprocal(
        inputs: tuple[Tensor, ...] | list[Tensor],
    ) -> tuple[Tensor, ...]: ...
    def reciprocal_(inputs: tuple[Tensor, ...] | list[Tensor]) -> tuple[Tensor, ...] | list[Tensor]: ...
    def round(inputs: tuple[Tensor, ...] | list[Tensor]) -> tuple[Tensor, ...]: ...
    def round_(inputs: tuple[Tensor, ...] | list[Tensor]) -> tuple[Tensor, ...] | list[Tensor]: ...
    def rsqrt(inputs: tuple[Tensor, ...] | list[Tensor]) -> tuple[Tensor, ...]: ...
    def rsqrt_(inputs: tuple[Tensor, ...] | list[Tensor]) -> tuple[Tensor, ...] | list[Tensor]: ...
    def sigmoid(
        inputs: tuple[Tensor, ...] | list[Tensor],
    ) -> tuple[Tensor, ...]: ...
    def sigmoid_(inputs: tuple[Tensor, ...] | list[Tensor]) -> tuple[Tensor, ...] | list[Tensor]: ...
    def sign(inputs: tuple[Tensor, ...] | list[Tensor]) -> tuple[Tensor, ...]: ...
    def sign_(inputs: tuple[Tensor, ...] | list[Tensor]) -> tuple[Tensor, ...] | list[Tensor]: ...
    def sin(inputs: tuple[Tensor, ...] | list[Tensor]) -> tuple[Tensor, ...]: ...
    def sin_(inputs: tuple[Tensor, ...] | list[Tensor]) -> tuple[Tensor, ...] | list[Tensor]: ...
    def sinh(inputs: tuple[Tensor, ...] | list[Tensor]) -> tuple[Tensor, ...]: ...
    def sinh_(inputs: tuple[Tensor, ...] | list[Tensor]) -> tuple[Tensor, ...] | list[Tensor]: ...
    def sqrt(inputs: tuple[Tensor, ...] | list[Tensor]) -> tuple[Tensor, ...]: ...
    def sqrt_(inputs: tuple[Tensor, ...] | list[Tensor]) -> tuple[Tensor, ...] | list[Tensor]: ...
    @overload
    def sub(
        inputs: tuple[Tensor, ...] | list[Tensor],
        other: Number | _complex | PySymType,
    ) -> tuple[Tensor, ...]: ...
    @overload
    def sub(
        inputs: tuple[Tensor, ...] | list[Tensor],
        other: _ScalarList[_ScalarT],
    ) -> tuple[Tensor, ...]: ...
    @overload
    def sub(
        inputs: tuple[Tensor, ...] | list[Tensor],
        other: tuple[Tensor, ...] | list[Tensor],
        *,
        alpha: Number | _complex | PySymType = 1,
    ) -> tuple[Tensor, ...]: ...
    def sub(*args: object, **kwargs: object) -> tuple[Tensor, ...]: ...
    @overload
    def sub_(
        inputs: tuple[Tensor, ...] | list[Tensor],
        other: Number | _complex | PySymType,
    ) -> tuple[Tensor, ...] | list[Tensor]: ...
    @overload
    def sub_(
        inputs: tuple[Tensor, ...] | list[Tensor],
        other: _ScalarList[_ScalarT],
    ) -> tuple[Tensor, ...] | list[Tensor]: ...
    @overload
    def sub_(
        inputs: tuple[Tensor, ...] | list[Tensor],
        other: tuple[Tensor, ...] | list[Tensor],
        *,
        alpha: Number | _complex | PySymType = 1,
    ) -> tuple[Tensor, ...] | list[Tensor]: ...
    def sub_(*args: object, **kwargs: object) -> tuple[Tensor, ...] | list[Tensor]: ...
    def tan(inputs: tuple[Tensor, ...] | list[Tensor]) -> tuple[Tensor, ...]: ...
    def tan_(inputs: tuple[Tensor, ...] | list[Tensor]) -> tuple[Tensor, ...] | list[Tensor]: ...
    def tanh(inputs: tuple[Tensor, ...] | list[Tensor]) -> tuple[Tensor, ...]: ...
    def tanh_(inputs: tuple[Tensor, ...] | list[Tensor]) -> tuple[Tensor, ...] | list[Tensor]: ...
    def trunc(inputs: tuple[Tensor, ...] | list[Tensor]) -> tuple[Tensor, ...]: ...
    def trunc_(inputs: tuple[Tensor, ...] | list[Tensor]) -> tuple[Tensor, ...] | list[Tensor]: ...
    def zero_(inputs: tuple[Tensor, ...] | list[Tensor]) -> tuple[Tensor, ...] | list[Tensor]: ...
else:
    abs = _add_docstr(_foreach.abs, _unary_doc("abs", "torch.abs", inplace=False))
    abs_ = _add_docstr(_foreach.abs_, _unary_doc("abs_", "torch.abs", inplace=True))
    acos = _add_docstr(_foreach.acos, _unary_doc("acos", "torch.acos", inplace=False))
    acos_ = _add_docstr(_foreach.acos_, _unary_doc("acos_", "torch.acos", inplace=True))
    asin = _add_docstr(_foreach.asin, _unary_doc("asin", "torch.asin", inplace=False))
    asin_ = _add_docstr(_foreach.asin_, _unary_doc("asin_", "torch.asin", inplace=True))
    atan = _add_docstr(_foreach.atan, _unary_doc("atan", "torch.atan", inplace=False))
    atan_ = _add_docstr(_foreach.atan_, _unary_doc("atan_", "torch.atan", inplace=True))
    ceil = _add_docstr(_foreach.ceil, _unary_doc("ceil", "torch.ceil", inplace=False))
    ceil_ = _add_docstr(_foreach.ceil_, _unary_doc("ceil_", "torch.ceil", inplace=True))
    cos = _add_docstr(_foreach.cos, _unary_doc("cos", "torch.cos", inplace=False))
    cos_ = _add_docstr(_foreach.cos_, _unary_doc("cos_", "torch.cos", inplace=True))
    cosh = _add_docstr(_foreach.cosh, _unary_doc("cosh", "torch.cosh", inplace=False))
    cosh_ = _add_docstr(_foreach.cosh_, _unary_doc("cosh_", "torch.cosh", inplace=True))
    erf = _add_docstr(_foreach.erf, _unary_doc("erf", "torch.erf", inplace=False))
    erf_ = _add_docstr(_foreach.erf_, _unary_doc("erf_", "torch.erf", inplace=True))
    erfc = _add_docstr(_foreach.erfc, _unary_doc("erfc", "torch.erfc", inplace=False))
    erfc_ = _add_docstr(_foreach.erfc_, _unary_doc("erfc_", "torch.erfc", inplace=True))
    exp = _add_docstr(_foreach.exp, _unary_doc("exp", "torch.exp", inplace=False))
    exp_ = _add_docstr(_foreach.exp_, _unary_doc("exp_", "torch.exp", inplace=True))
    expm1 = _add_docstr(_foreach.expm1, _unary_doc("expm1", "torch.expm1", inplace=False))
    expm1_ = _add_docstr(_foreach.expm1_, _unary_doc("expm1_", "torch.expm1", inplace=True))
    floor = _add_docstr(_foreach.floor, _unary_doc("floor", "torch.floor", inplace=False))
    floor_ = _add_docstr(_foreach.floor_, _unary_doc("floor_", "torch.floor", inplace=True))
    frac = _add_docstr(_foreach.frac, _unary_doc("frac", "torch.frac", inplace=False))
    frac_ = _add_docstr(_foreach.frac_, _unary_doc("frac_", "torch.frac", inplace=True))
    lgamma = _add_docstr(_foreach.lgamma, _unary_doc("lgamma", "torch.lgamma", inplace=False))
    lgamma_ = _add_docstr(_foreach.lgamma_, _unary_doc("lgamma_", "torch.lgamma", inplace=True))
    log = _add_docstr(_foreach.log, _unary_doc("log", "torch.log", inplace=False))
    log_ = _add_docstr(_foreach.log_, _unary_doc("log_", "torch.log", inplace=True))
    log10 = _add_docstr(_foreach.log10, _unary_doc("log10", "torch.log10", inplace=False))
    log10_ = _add_docstr(_foreach.log10_, _unary_doc("log10_", "torch.log10", inplace=True))
    log1p = _add_docstr(_foreach.log1p, _unary_doc("log1p", "torch.log1p", inplace=False))
    log1p_ = _add_docstr(_foreach.log1p_, _unary_doc("log1p_", "torch.log1p", inplace=True))
    log2 = _add_docstr(_foreach.log2, _unary_doc("log2", "torch.log2", inplace=False))
    log2_ = _add_docstr(_foreach.log2_, _unary_doc("log2_", "torch.log2", inplace=True))
    neg = _add_docstr(_foreach.neg, _unary_doc("neg", "torch.neg", inplace=False))
    neg_ = _add_docstr(_foreach.neg_, _unary_doc("neg_", "torch.neg", inplace=True))
    reciprocal = _add_docstr(
        _foreach.reciprocal,
        _unary_doc("reciprocal", "torch.reciprocal", inplace=False),
    )
    reciprocal_ = _add_docstr(
        _foreach.reciprocal_,
        _unary_doc("reciprocal_", "torch.reciprocal", inplace=True),
    )
    round = _add_docstr(_foreach.round, _unary_doc("round", "torch.round", inplace=False))
    round_ = _add_docstr(_foreach.round_, _unary_doc("round_", "torch.round", inplace=True))
    rsqrt = _add_docstr(_foreach.rsqrt, _unary_doc("rsqrt", "torch.rsqrt", inplace=False))
    rsqrt_ = _add_docstr(_foreach.rsqrt_, _unary_doc("rsqrt_", "torch.rsqrt", inplace=True))
    sigmoid = _add_docstr(_foreach.sigmoid, _unary_doc("sigmoid", "torch.sigmoid", inplace=False))
    sigmoid_ = _add_docstr(_foreach.sigmoid_, _unary_doc("sigmoid_", "torch.sigmoid", inplace=True))
    sign = _add_docstr(_foreach.sign, _unary_doc("sign", "torch.sign", inplace=False))
    sign_ = _add_docstr(_foreach.sign_, _unary_doc("sign_", "torch.sign", inplace=True))
    sin = _add_docstr(_foreach.sin, _unary_doc("sin", "torch.sin", inplace=False))
    sin_ = _add_docstr(_foreach.sin_, _unary_doc("sin_", "torch.sin", inplace=True))
    sinh = _add_docstr(_foreach.sinh, _unary_doc("sinh", "torch.sinh", inplace=False))
    sinh_ = _add_docstr(_foreach.sinh_, _unary_doc("sinh_", "torch.sinh", inplace=True))
    sqrt = _add_docstr(_foreach.sqrt, _unary_doc("sqrt", "torch.sqrt", inplace=False))
    sqrt_ = _add_docstr(_foreach.sqrt_, _unary_doc("sqrt_", "torch.sqrt", inplace=True))
    tan = _add_docstr(_foreach.tan, _unary_doc("tan", "torch.tan", inplace=False))
    tan_ = _add_docstr(_foreach.tan_, _unary_doc("tan_", "torch.tan", inplace=True))
    tanh = _add_docstr(_foreach.tanh, _unary_doc("tanh", "torch.tanh", inplace=False))
    tanh_ = _add_docstr(_foreach.tanh_, _unary_doc("tanh_", "torch.tanh", inplace=True))
    trunc = _add_docstr(_foreach.trunc, _unary_doc("trunc", "torch.trunc", inplace=False))
    trunc_ = _add_docstr(_foreach.trunc_, _unary_doc("trunc_", "torch.trunc", inplace=True))

    clone = _add_docstr(
        _foreach.clone,
        rf"""
clone(inputs, *, memory_format=torch.preserve_format) -> tuple[Tensor, ...]

Clones every tensor in ``inputs``.

{_common("torch.clone", False)}

Args:
    inputs (list or tuple of Tensor): tensors to clone.
    memory_format (:class:`torch.memory_format`, optional): desired memory
        format. Default: :attr:`torch.preserve_format`.

Returns:
    a tuple containing the cloned tensors.
""",
    )

    add = _add_docstr(
        _foreach.add,
        _binary_doc(
            "add",
            "torch.add",
            "a shared scalar, a scalar list or tuple, a tensor list, or a shared 0-D scalar tensor",
            inplace=False,
            alpha=(
                "supported only when ``other`` is a tensor list or a shared 0-D "
                "scalar tensor. Default: ``1``."
            ),
            shared_tensor=True,
        ),
    )
    add_ = _add_docstr(
        _foreach.add_,
        _binary_doc(
            "add_",
            "torch.add",
            "a shared scalar, a scalar list or tuple, a tensor list, or a shared 0-D scalar tensor",
            inplace=True,
            alpha=(
                "supported only when ``other`` is a tensor list or a shared 0-D "
                "scalar tensor. Default: ``1``."
            ),
            shared_tensor=True,
        ),
    )
    sub = _add_docstr(
        _foreach.sub,
        _binary_doc(
            "sub",
            "torch.sub",
            "a shared scalar, a scalar list or tuple, or a tensor list",
            inplace=False,
            alpha="supported only when ``other`` is a tensor list. Default: ``1``.",
        ),
    )
    sub_ = _add_docstr(
        _foreach.sub_,
        _binary_doc(
            "sub_",
            "torch.sub",
            "a shared scalar, a scalar list or tuple, or a tensor list",
            inplace=True,
            alpha="supported only when ``other`` is a tensor list. Default: ``1``.",
        ),
    )
    mul = _add_docstr(
        _foreach.mul,
        _binary_doc(
            "mul",
            "torch.mul",
            "a shared scalar, a scalar list or tuple, a tensor list, or a shared 0-D scalar tensor",
            inplace=False,
            shared_tensor=True,
        ),
    )
    mul_ = _add_docstr(
        _foreach.mul_,
        _binary_doc(
            "mul_",
            "torch.mul",
            "a shared scalar, a scalar list or tuple, a tensor list, or a shared 0-D scalar tensor",
            inplace=True,
            shared_tensor=True,
        ),
    )
    div = _add_docstr(
        _foreach.div,
        _binary_doc(
            "div",
            "torch.div",
            "a shared scalar, a scalar list or tuple, a tensor list, or a shared 0-D scalar tensor",
            inplace=False,
            shared_tensor=True,
        ),
    )
    div_ = _add_docstr(
        _foreach.div_,
        _binary_doc(
            "div_",
            "torch.div",
            "a shared scalar, a scalar list or tuple, a tensor list, or a shared 0-D scalar tensor",
            inplace=True,
            shared_tensor=True,
        ),
    )

    clamp_min = _add_docstr(
        _foreach.clamp_min,
        _binary_doc(
            "clamp_min",
            "torch.clamp_min",
            "a shared scalar, a scalar list or tuple, or a tensor list",
            inplace=False,
            operand="min",
        ),
    )
    clamp_min_ = _add_docstr(
        _foreach.clamp_min_,
        _binary_doc(
            "clamp_min_",
            "torch.clamp_min",
            "a shared scalar, a scalar list or tuple, or a tensor list",
            inplace=True,
            operand="min",
        ),
    )
    clamp_max = _add_docstr(
        _foreach.clamp_max,
        _binary_doc(
            "clamp_max",
            "torch.clamp_max",
            "a shared scalar, a scalar list or tuple, or a tensor list",
            inplace=False,
            operand="max",
        ),
    )
    clamp_max_ = _add_docstr(
        _foreach.clamp_max_,
        _binary_doc(
            "clamp_max_",
            "torch.clamp_max",
            "a shared scalar, a scalar list or tuple, or a tensor list",
            inplace=True,
            operand="max",
        ),
    )
    minimum = _add_docstr(
        _foreach.minimum,
        _binary_doc(
            "minimum",
            "torch.minimum",
            (
                "a shared scalar or a scalar list or tuple (with ``torch.clamp_max`` "
                "semantics), or a tensor list"
            ),
            inplace=False,
        ),
    )
    minimum_ = _add_docstr(
        _foreach.minimum_,
        _binary_doc(
            "minimum_",
            "torch.minimum",
            (
                "a shared scalar or a scalar list or tuple (with ``torch.clamp_max`` "
                "semantics), or a tensor list"
            ),
            inplace=True,
        ),
    )
    maximum = _add_docstr(
        _foreach.maximum,
        _binary_doc(
            "maximum",
            "torch.maximum",
            (
                "a shared scalar or a scalar list or tuple (with ``torch.clamp_min`` "
                "semantics), or a tensor list"
            ),
            inplace=False,
        ),
    )
    maximum_ = _add_docstr(
        _foreach.maximum_,
        _binary_doc(
            "maximum_",
            "torch.maximum",
            (
                "a shared scalar or a scalar list or tuple (with ``torch.clamp_min`` "
                "semantics), or a tensor list"
            ),
            inplace=True,
        ),
    )

    pow = _add_docstr(
        _foreach.pow,
        rf"""
pow(input, exponent) -> tuple[Tensor, ...]

Applies :func:`torch.pow` at each list position.

{_common("torch.pow", False)}

``input`` may be a tensor list and ``exponent`` may be a shared scalar,
a scalar list or tuple, or a tensor list. Alternatively, ``input`` may be a
shared scalar when ``exponent`` is a tensor list.

Args:
    input (Number, list of Tensor, or tuple of Tensor): bases.
    exponent (Number, list or tuple of Number, or list or tuple of Tensor): exponents.

Returns:
    a tuple containing one result tensor per list position.
""",
    )
    pow_ = _add_docstr(
        _foreach.pow_,
        rf"""
pow_(inputs, exponent) -> tuple[Tensor, ...] | list[Tensor]

In-place version of :func:`torch.foreach.pow`.

{_common("torch.pow", True)}

``inputs`` must be a tensor list. ``exponent`` may be a shared scalar, a
scalar list or tuple, or a tensor list. The scalar-left form has no in-place variant.

Args:
    inputs (list or tuple of Tensor): bases to mutate.
    exponent (Number, list or tuple of Number, or list or tuple of Tensor): exponents.

Returns:
    the exact ``inputs`` list or tuple.
""",
    )

    addcmul = _add_docstr(
        _foreach.addcmul,
        _pointwise_doc("addcmul", "torch.addcmul", inplace=False),
    )
    addcmul_ = _add_docstr(
        _foreach.addcmul_,
        _pointwise_doc("addcmul_", "torch.addcmul", inplace=True),
    )
    addcdiv = _add_docstr(
        _foreach.addcdiv,
        _pointwise_doc("addcdiv", "torch.addcdiv", inplace=False),
    )
    addcdiv_ = _add_docstr(
        _foreach.addcdiv_,
        _pointwise_doc("addcdiv_", "torch.addcdiv", inplace=True),
    )

    lerp = _add_docstr(
        _foreach.lerp,
        rf"""
lerp(inputs, ends, weight) -> tuple[Tensor, ...]

Applies :func:`torch.lerp` to corresponding tensors in ``inputs`` and
``ends``.

{_common("torch.lerp", False)}

``weight`` may be one shared scalar, a scalar list or tuple, or a tensor list.

Args:
    inputs (list or tuple of Tensor): starting tensors.
    ends (list or tuple of Tensor): ending tensors.
    weight (Number, list or tuple of Number, or list or tuple of Tensor): interpolation
        weights.

Returns:
    a tuple containing one result tensor per list position.
""",
    )
    lerp_ = _add_docstr(
        _foreach.lerp_,
        rf"""
lerp_(inputs, ends, weight) -> tuple[Tensor, ...] | list[Tensor]

In-place version of :func:`torch.foreach.lerp`.

{_common("torch.lerp", True)}

``weight`` may be one shared scalar, a scalar list or tuple, or a tensor list.

Args:
    inputs (list or tuple of Tensor): starting tensors to mutate.
    ends (list or tuple of Tensor): ending tensors.
    weight (Number, list or tuple of Number, or list or tuple of Tensor): interpolation
        weights.

Returns:
    the exact ``inputs`` list or tuple.
""",
    )

    max = _add_docstr(
        _foreach.max,
        rf"""
max(inputs) -> tuple[Tensor, ...]

Returns the maximum value of each tensor in ``inputs``.

{_common("torch.max", False)}

This operation reduces every input tensor over all dimensions. It does not
accept a dimension or return indices.

Args:
    inputs (list or tuple of Tensor): tensors to reduce.

Returns:
    a tuple of scalar tensors.
""",
    )
    norm = _add_docstr(
        _foreach.norm,
        rf"""
norm(inputs, ord=2, *, dtype=None) -> tuple[Tensor, ...]

Returns the vector norm of each tensor in ``inputs``.

{_common("torch.linalg.vector_norm", False)}

Every input tensor is reduced over all dimensions. The ``dim`` and
``keepdim`` options of :func:`torch.linalg.vector_norm` are not supported.

Args:
    inputs (list or tuple of Tensor): tensors to reduce.
    ord (Number, optional): norm order. Default: ``2``.
    dtype (:class:`torch.dtype`, optional): dtype used for the computation.

Returns:
    a tuple containing one norm tensor per input tensor.
""",
    )

    copy_ = _add_docstr(
        _foreach.copy_,
        r"""
copy_(inputs, srcs, non_blocking=False) -> tuple[Tensor, ...] | list[Tensor]

Copies each tensor in ``srcs`` into the corresponding tensor in
``inputs``, following :meth:`torch.Tensor.copy_`.

Both lists must be non-empty and have the same length. The operation mutates
the tensors in ``inputs`` and returns the exact ``inputs`` container.

Args:
    inputs (list or tuple of Tensor): destination tensors.
    srcs (list or tuple of Tensor): source tensors.
    non_blocking (bool, optional): allows asynchronous host/device copies when
        supported. Default: ``False``.

Returns:
    the exact ``inputs`` list or tuple.
""",
    )
    zero_ = _add_docstr(
        _foreach.zero_,
        r"""
zero_(inputs) -> tuple[Tensor, ...] | list[Tensor]

Fills every tensor in ``inputs`` with zero.

The operation is equivalent to calling :meth:`torch.Tensor.zero_` on every
tensor and returns the exact ``inputs`` container.

Args:
    inputs (list or tuple of Tensor): tensors to zero.

Returns:
    the exact ``inputs`` list or tuple.
""",
    )
    mm = _add_docstr(
        _foreach.mm,
        r"""
mm(inputs, mat2s) -> tuple[Tensor, ...]

Multiplies corresponding matrices from ``inputs`` and ``mat2s`` using
:func:`torch.mm`.

Both lists must be non-empty and have the same length. Compatible calls may use
a grouped matrix multiplication implementation; all other supported calls use a
loop of :func:`torch.mm`.

Args:
    inputs (list or tuple of Tensor): first matrices.
    mat2s (list or tuple of Tensor): second matrices.

Returns:
    a tuple containing one matrix product per list position.
""",
    )
