r"""Operations over lists of tensors.

Each function applies the corresponding ordinary PyTorch operation to every
position in one or more tensor lists. This API is in beta and may change based
on user feedback.

The functions may use an accelerated multi-tensor implementation when their
inputs meet its requirements. Otherwise they use a semantically equivalent
per-tensor fallback. Calling a function in this module does not guarantee a
single or fused kernel.
"""

import inspect
from typing import TYPE_CHECKING

import torch
from torch.overrides import wrap_torch_function


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


def _unary_doc(
    name: str,
    reference: str,
    *,
    inplace: bool,
    note: str | None = None,
) -> str:
    suffix = "tuple[Tensor, ...] | list[Tensor]" if inplace else "tuple[Tensor, ...]"
    note_text = "" if note is None else f"\n{note}\n"
    return rf"""
{name}(inputs) -> {suffix}

Applies :func:`{reference}` to each tensor in ``inputs``.

{_common(reference, inplace)}
{note_text}

Args:
    inputs (list or tuple of Tensor): tensors to transform.

Returns:
    {_result(inplace)}.
"""


def _binary_doc(
    name: str,
    reference: str,
    signatures: tuple[str, ...],
    *,
    inplace: bool,
    alpha: str | None = None,
    operand: str = "other",
    shared_tensor: bool = False,
    operand_note: str | None = None,
) -> str:
    suffix = "tuple[Tensor, ...] | list[Tensor]" if inplace else "tuple[Tensor, ...]"
    alpha_sig = ", *, alpha=1" if alpha is not None else ""
    alpha_doc = "" if alpha is None else f"\n    alpha (Number, optional): {alpha}"
    signature_lines = (f"    {name}({signature})" for signature in signatures)
    supported_signatures = "\n".join(signature_lines)
    shared_tensor_note = (
        "\nA shared ``Tensor`` operand must be a 0-D scalar tensor.\n"
        if shared_tensor
        else ""
    )
    operand_note_text = "" if operand_note is None else f"\n{operand_note}\n"
    operand_type = "Number, list or tuple of Number, or list or tuple of Tensor"
    if shared_tensor:
        operand_type = f"{operand_type}, or Tensor"
    return rf"""
{name}(inputs, {operand}{alpha_sig}) -> {suffix}

Applies :func:`{reference}` to every tensor in ``inputs``.

{_common(reference, inplace)}

Supported signatures::

{supported_signatures}

``TensorList`` and ``ScalarList`` denote a list or tuple of tensors and
scalars, respectively.
{shared_tensor_note}{operand_note_text}

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

    def abs_(
        inputs: tuple[Tensor, ...] | list[Tensor],
    ) -> tuple[Tensor, ...] | list[Tensor]: ...

    def acos(inputs: tuple[Tensor, ...] | list[Tensor]) -> tuple[Tensor, ...]: ...

    def acos_(
        inputs: tuple[Tensor, ...] | list[Tensor],
    ) -> tuple[Tensor, ...] | list[Tensor]: ...

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

    def addcdiv_(
        *args: object, **kwargs: object
    ) -> tuple[Tensor, ...] | list[Tensor]: ...

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

    def addcmul_(
        *args: object, **kwargs: object
    ) -> tuple[Tensor, ...] | list[Tensor]: ...

    def asin(inputs: tuple[Tensor, ...] | list[Tensor]) -> tuple[Tensor, ...]: ...

    def asin_(
        inputs: tuple[Tensor, ...] | list[Tensor],
    ) -> tuple[Tensor, ...] | list[Tensor]: ...

    def atan(inputs: tuple[Tensor, ...] | list[Tensor]) -> tuple[Tensor, ...]: ...

    def atan_(
        inputs: tuple[Tensor, ...] | list[Tensor],
    ) -> tuple[Tensor, ...] | list[Tensor]: ...

    def ceil(inputs: tuple[Tensor, ...] | list[Tensor]) -> tuple[Tensor, ...]: ...

    def ceil_(
        inputs: tuple[Tensor, ...] | list[Tensor],
    ) -> tuple[Tensor, ...] | list[Tensor]: ...

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

    def clamp_max_(
        *args: object, **kwargs: object
    ) -> tuple[Tensor, ...] | list[Tensor]: ...

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

    def clamp_min_(
        *args: object, **kwargs: object
    ) -> tuple[Tensor, ...] | list[Tensor]: ...

    def clone(
        inputs: tuple[Tensor, ...] | list[Tensor],
        *,
        memory_format: memory_format | None = torch.preserve_format,
    ) -> tuple[Tensor, ...]: ...

    def copy_(
        inputs: tuple[Tensor, ...] | list[Tensor],
        srcs: tuple[Tensor, ...] | list[Tensor],
        non_blocking: _bool = False,
    ) -> tuple[Tensor, ...] | list[Tensor]: ...

    def cos(inputs: tuple[Tensor, ...] | list[Tensor]) -> tuple[Tensor, ...]: ...

    def cos_(
        inputs: tuple[Tensor, ...] | list[Tensor],
    ) -> tuple[Tensor, ...] | list[Tensor]: ...

    def cosh(inputs: tuple[Tensor, ...] | list[Tensor]) -> tuple[Tensor, ...]: ...

    def cosh_(
        inputs: tuple[Tensor, ...] | list[Tensor],
    ) -> tuple[Tensor, ...] | list[Tensor]: ...

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
    def div_(
        inputs: tuple[Tensor, ...] | list[Tensor], other: Tensor
    ) -> tuple[Tensor, ...] | list[Tensor]: ...

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

    def erf_(
        inputs: tuple[Tensor, ...] | list[Tensor],
    ) -> tuple[Tensor, ...] | list[Tensor]: ...

    def erfc(inputs: tuple[Tensor, ...] | list[Tensor]) -> tuple[Tensor, ...]: ...

    def erfc_(
        inputs: tuple[Tensor, ...] | list[Tensor],
    ) -> tuple[Tensor, ...] | list[Tensor]: ...

    def exp(inputs: tuple[Tensor, ...] | list[Tensor]) -> tuple[Tensor, ...]: ...

    def exp_(
        inputs: tuple[Tensor, ...] | list[Tensor],
    ) -> tuple[Tensor, ...] | list[Tensor]: ...

    def expm1(inputs: tuple[Tensor, ...] | list[Tensor]) -> tuple[Tensor, ...]: ...

    def expm1_(
        inputs: tuple[Tensor, ...] | list[Tensor],
    ) -> tuple[Tensor, ...] | list[Tensor]: ...

    def floor(inputs: tuple[Tensor, ...] | list[Tensor]) -> tuple[Tensor, ...]: ...

    def floor_(
        inputs: tuple[Tensor, ...] | list[Tensor],
    ) -> tuple[Tensor, ...] | list[Tensor]: ...

    def frac(inputs: tuple[Tensor, ...] | list[Tensor]) -> tuple[Tensor, ...]: ...

    def frac_(
        inputs: tuple[Tensor, ...] | list[Tensor],
    ) -> tuple[Tensor, ...] | list[Tensor]: ...

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

    def lgamma_(
        inputs: tuple[Tensor, ...] | list[Tensor],
    ) -> tuple[Tensor, ...] | list[Tensor]: ...

    def log(inputs: tuple[Tensor, ...] | list[Tensor]) -> tuple[Tensor, ...]: ...

    def log10(inputs: tuple[Tensor, ...] | list[Tensor]) -> tuple[Tensor, ...]: ...

    def log10_(
        inputs: tuple[Tensor, ...] | list[Tensor],
    ) -> tuple[Tensor, ...] | list[Tensor]: ...

    def log1p(inputs: tuple[Tensor, ...] | list[Tensor]) -> tuple[Tensor, ...]: ...

    def log1p_(
        inputs: tuple[Tensor, ...] | list[Tensor],
    ) -> tuple[Tensor, ...] | list[Tensor]: ...

    def log2(inputs: tuple[Tensor, ...] | list[Tensor]) -> tuple[Tensor, ...]: ...

    def log2_(
        inputs: tuple[Tensor, ...] | list[Tensor],
    ) -> tuple[Tensor, ...] | list[Tensor]: ...

    def log_(
        inputs: tuple[Tensor, ...] | list[Tensor],
    ) -> tuple[Tensor, ...] | list[Tensor]: ...

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

    def maximum_(
        *args: object, **kwargs: object
    ) -> tuple[Tensor, ...] | list[Tensor]: ...

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

    def minimum_(
        *args: object, **kwargs: object
    ) -> tuple[Tensor, ...] | list[Tensor]: ...

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
    def mul_(
        inputs: tuple[Tensor, ...] | list[Tensor], other: Tensor
    ) -> tuple[Tensor, ...] | list[Tensor]: ...

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

    def neg_(
        inputs: tuple[Tensor, ...] | list[Tensor],
    ) -> tuple[Tensor, ...] | list[Tensor]: ...

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

    def reciprocal_(
        inputs: tuple[Tensor, ...] | list[Tensor],
    ) -> tuple[Tensor, ...] | list[Tensor]: ...

    def round(inputs: tuple[Tensor, ...] | list[Tensor]) -> tuple[Tensor, ...]: ...

    def round_(
        inputs: tuple[Tensor, ...] | list[Tensor],
    ) -> tuple[Tensor, ...] | list[Tensor]: ...

    def rsqrt(inputs: tuple[Tensor, ...] | list[Tensor]) -> tuple[Tensor, ...]: ...

    def rsqrt_(
        inputs: tuple[Tensor, ...] | list[Tensor],
    ) -> tuple[Tensor, ...] | list[Tensor]: ...

    def sigmoid(
        inputs: tuple[Tensor, ...] | list[Tensor],
    ) -> tuple[Tensor, ...]: ...

    def sigmoid_(
        inputs: tuple[Tensor, ...] | list[Tensor],
    ) -> tuple[Tensor, ...] | list[Tensor]: ...

    def sign(inputs: tuple[Tensor, ...] | list[Tensor]) -> tuple[Tensor, ...]: ...

    def sign_(
        inputs: tuple[Tensor, ...] | list[Tensor],
    ) -> tuple[Tensor, ...] | list[Tensor]: ...

    def sin(inputs: tuple[Tensor, ...] | list[Tensor]) -> tuple[Tensor, ...]: ...

    def sin_(
        inputs: tuple[Tensor, ...] | list[Tensor],
    ) -> tuple[Tensor, ...] | list[Tensor]: ...

    def sinh(inputs: tuple[Tensor, ...] | list[Tensor]) -> tuple[Tensor, ...]: ...

    def sinh_(
        inputs: tuple[Tensor, ...] | list[Tensor],
    ) -> tuple[Tensor, ...] | list[Tensor]: ...

    def sqrt(inputs: tuple[Tensor, ...] | list[Tensor]) -> tuple[Tensor, ...]: ...

    def sqrt_(
        inputs: tuple[Tensor, ...] | list[Tensor],
    ) -> tuple[Tensor, ...] | list[Tensor]: ...

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

    def tan_(
        inputs: tuple[Tensor, ...] | list[Tensor],
    ) -> tuple[Tensor, ...] | list[Tensor]: ...

    def tanh(inputs: tuple[Tensor, ...] | list[Tensor]) -> tuple[Tensor, ...]: ...

    def tanh_(
        inputs: tuple[Tensor, ...] | list[Tensor],
    ) -> tuple[Tensor, ...] | list[Tensor]: ...

    def trunc(inputs: tuple[Tensor, ...] | list[Tensor]) -> tuple[Tensor, ...]: ...

    def trunc_(
        inputs: tuple[Tensor, ...] | list[Tensor],
    ) -> tuple[Tensor, ...] | list[Tensor]: ...

    def zero_(
        inputs: tuple[Tensor, ...] | list[Tensor],
    ) -> tuple[Tensor, ...] | list[Tensor]: ...
else:

    class _DefaultAlpha:
        def __repr__(self):
            return "1"

    _DEFAULT_ALPHA = _DefaultAlpha()

    def _wrap(func, name):
        func.__name__ = name
        func.__qualname__ = name
        signature = inspect.signature(func)

        def dispatcher(*args, **kwargs):
            try:
                bound = signature.bind(*args, **kwargs)
            except TypeError:
                # Preserve normal Python call errors before override dispatch.
                return func(*args, **kwargs)
            relevant_args = []
            for arg in bound.arguments.values():
                if isinstance(arg, (list, tuple)):
                    relevant_args.extend(arg)
                else:
                    relevant_args.append(arg)
            return relevant_args

        parameters = [
            parameter.replace(default=1)
            if parameter.default is _DEFAULT_ALPHA
            else parameter
            for parameter in signature.parameters.values()
        ]
        wrapped = wrap_torch_function(dispatcher)(func)
        wrapped.__signature__ = signature.replace(parameters=parameters)
        return wrapped

    def _set_doc(func, doc):
        func.__doc__ = doc
        return func

    def _make_unary(name):
        private = getattr(torch, f"_foreach_{name}")

        def func(inputs):
            result = private(inputs)
            return inputs if name.endswith("_") else result

        return _wrap(func, name)

    def _make_binary(name):
        private = getattr(torch, f"_foreach_{name}")

        def func(inputs, other):
            result = private(inputs, other)
            return inputs if name.endswith("_") else result

        return _wrap(func, name)

    def _make_add(name):
        private = getattr(torch, f"_foreach_{name}")

        def func(inputs, other, *, alpha=_DEFAULT_ALPHA):
            if alpha is _DEFAULT_ALPHA:
                result = private(inputs, other)
            else:
                result = private(inputs, other, alpha=alpha)
            return inputs if name.endswith("_") else result

        return _wrap(func, name)

    def _make_pointwise(name):
        private = getattr(torch, f"_foreach_{name}")

        def func(inputs, tensor1s, tensor2s, *, value=1):
            result = private(inputs, tensor1s, tensor2s, value)
            return inputs if name.endswith("_") else result

        return _wrap(func, name)

    def _make_lerp(name):
        private = getattr(torch, f"_foreach_{name}")

        def func(inputs, ends, weight):
            result = private(inputs, ends, weight)
            return inputs if name.endswith("_") else result

        return _wrap(func, name)

    def _make_pow(name, *, inplace):
        private = getattr(torch, f"_foreach_{name}")

        if inplace:

            def func(inputs, exponent):
                private(inputs, exponent)
                return inputs

        else:

            def func(input, exponent):
                return private(input, exponent)

        return _wrap(func, name)

    abs = _make_unary("abs")
    abs_ = _make_unary("abs_")
    acos = _make_unary("acos")
    acos_ = _make_unary("acos_")
    asin = _make_unary("asin")
    asin_ = _make_unary("asin_")
    atan = _make_unary("atan")
    atan_ = _make_unary("atan_")
    ceil = _make_unary("ceil")
    ceil_ = _make_unary("ceil_")
    cos = _make_unary("cos")
    cos_ = _make_unary("cos_")
    cosh = _make_unary("cosh")
    cosh_ = _make_unary("cosh_")
    erf = _make_unary("erf")
    erf_ = _make_unary("erf_")
    erfc = _make_unary("erfc")
    erfc_ = _make_unary("erfc_")
    exp = _make_unary("exp")
    exp_ = _make_unary("exp_")
    expm1 = _make_unary("expm1")
    expm1_ = _make_unary("expm1_")
    floor = _make_unary("floor")
    floor_ = _make_unary("floor_")
    frac = _make_unary("frac")
    frac_ = _make_unary("frac_")
    lgamma = _make_unary("lgamma")
    lgamma_ = _make_unary("lgamma_")
    log = _make_unary("log")
    log_ = _make_unary("log_")
    log10 = _make_unary("log10")
    log10_ = _make_unary("log10_")
    log1p = _make_unary("log1p")
    log1p_ = _make_unary("log1p_")
    log2 = _make_unary("log2")
    log2_ = _make_unary("log2_")
    neg = _make_unary("neg")
    neg_ = _make_unary("neg_")
    reciprocal = _make_unary("reciprocal")
    reciprocal_ = _make_unary("reciprocal_")
    round = _make_unary("round")
    round_ = _make_unary("round_")
    rsqrt = _make_unary("rsqrt")
    rsqrt_ = _make_unary("rsqrt_")
    sigmoid = _make_unary("sigmoid")
    sigmoid_ = _make_unary("sigmoid_")
    sign = _make_unary("sign")
    sign_ = _make_unary("sign_")
    sin = _make_unary("sin")
    sin_ = _make_unary("sin_")
    sinh = _make_unary("sinh")
    sinh_ = _make_unary("sinh_")
    sqrt = _make_unary("sqrt")
    sqrt_ = _make_unary("sqrt_")
    tan = _make_unary("tan")
    tan_ = _make_unary("tan_")
    tanh = _make_unary("tanh")
    tanh_ = _make_unary("tanh_")
    trunc = _make_unary("trunc")
    trunc_ = _make_unary("trunc_")
    max = _make_unary("max")
    zero_ = _make_unary("zero_")

    add = _make_add("add")
    add_ = _make_add("add_")
    sub = _make_add("sub")
    sub_ = _make_add("sub_")
    addcdiv = _make_pointwise("addcdiv")
    addcdiv_ = _make_pointwise("addcdiv_")
    addcmul = _make_pointwise("addcmul")
    addcmul_ = _make_pointwise("addcmul_")
    div = _make_binary("div")
    div_ = _make_binary("div_")
    lerp = _make_lerp("lerp")
    lerp_ = _make_lerp("lerp_")
    maximum = _make_binary("maximum")
    maximum_ = _make_binary("maximum_")
    minimum = _make_binary("minimum")
    minimum_ = _make_binary("minimum_")
    mul = _make_binary("mul")
    mul_ = _make_binary("mul_")
    pow = _make_pow("pow", inplace=False)
    pow_ = _make_pow("pow_", inplace=True)

    def clamp_max(inputs, max):
        return torch._foreach_clamp_max(inputs, max)

    clamp_max = _wrap(clamp_max, "clamp_max")

    def clamp_max_(inputs, max):
        torch._foreach_clamp_max_(inputs, max)
        return inputs

    clamp_max_ = _wrap(clamp_max_, "clamp_max_")

    def clamp_min(inputs, min):
        return torch._foreach_clamp_min(inputs, min)

    clamp_min = _wrap(clamp_min, "clamp_min")

    def clamp_min_(inputs, min):
        torch._foreach_clamp_min_(inputs, min)
        return inputs

    clamp_min_ = _wrap(clamp_min_, "clamp_min_")

    def clone(inputs, *, memory_format=torch.preserve_format):
        return torch._foreach_clone(inputs, memory_format=memory_format)

    clone = _wrap(clone, "clone")

    def copy_(inputs, srcs, non_blocking=False):
        torch._foreach_copy_(inputs, srcs, non_blocking)
        return inputs

    copy_ = _wrap(copy_, "copy_")

    def mm(inputs, mat2s):
        return torch._foreach_mm(inputs, mat2s)

    mm = _wrap(mm, "mm")

    def norm(inputs, ord=2, *, dtype=None):
        return torch._foreach_norm(inputs, ord, dtype=dtype)

    norm = _wrap(norm, "norm")

    abs = _set_doc(abs, _unary_doc("abs", "torch.abs", inplace=False))
    abs_ = _set_doc(abs_, _unary_doc("abs_", "torch.abs", inplace=True))
    acos = _set_doc(acos, _unary_doc("acos", "torch.acos", inplace=False))
    acos_ = _set_doc(acos_, _unary_doc("acos_", "torch.acos", inplace=True))
    asin = _set_doc(asin, _unary_doc("asin", "torch.asin", inplace=False))
    asin_ = _set_doc(asin_, _unary_doc("asin_", "torch.asin", inplace=True))
    atan = _set_doc(atan, _unary_doc("atan", "torch.atan", inplace=False))
    atan_ = _set_doc(atan_, _unary_doc("atan_", "torch.atan", inplace=True))
    ceil = _set_doc(ceil, _unary_doc("ceil", "torch.ceil", inplace=False))
    ceil_ = _set_doc(ceil_, _unary_doc("ceil_", "torch.ceil", inplace=True))
    cos = _set_doc(cos, _unary_doc("cos", "torch.cos", inplace=False))
    cos_ = _set_doc(cos_, _unary_doc("cos_", "torch.cos", inplace=True))
    cosh = _set_doc(cosh, _unary_doc("cosh", "torch.cosh", inplace=False))
    cosh_ = _set_doc(cosh_, _unary_doc("cosh_", "torch.cosh", inplace=True))
    erf = _set_doc(erf, _unary_doc("erf", "torch.erf", inplace=False))
    erf_ = _set_doc(erf_, _unary_doc("erf_", "torch.erf", inplace=True))
    erfc = _set_doc(erfc, _unary_doc("erfc", "torch.erfc", inplace=False))
    erfc_ = _set_doc(erfc_, _unary_doc("erfc_", "torch.erfc", inplace=True))
    exp = _set_doc(exp, _unary_doc("exp", "torch.exp", inplace=False))
    exp_ = _set_doc(exp_, _unary_doc("exp_", "torch.exp", inplace=True))
    expm1 = _set_doc(expm1, _unary_doc("expm1", "torch.expm1", inplace=False))
    expm1_ = _set_doc(expm1_, _unary_doc("expm1_", "torch.expm1", inplace=True))
    floor = _set_doc(floor, _unary_doc("floor", "torch.floor", inplace=False))
    floor_ = _set_doc(floor_, _unary_doc("floor_", "torch.floor", inplace=True))
    frac = _set_doc(frac, _unary_doc("frac", "torch.frac", inplace=False))
    frac_ = _set_doc(frac_, _unary_doc("frac_", "torch.frac", inplace=True))
    lgamma = _set_doc(lgamma, _unary_doc("lgamma", "torch.lgamma", inplace=False))
    lgamma_ = _set_doc(lgamma_, _unary_doc("lgamma_", "torch.lgamma", inplace=True))
    log = _set_doc(log, _unary_doc("log", "torch.log", inplace=False))
    log_ = _set_doc(log_, _unary_doc("log_", "torch.log", inplace=True))
    log10 = _set_doc(log10, _unary_doc("log10", "torch.log10", inplace=False))
    log10_ = _set_doc(log10_, _unary_doc("log10_", "torch.log10", inplace=True))
    log1p = _set_doc(log1p, _unary_doc("log1p", "torch.log1p", inplace=False))
    log1p_ = _set_doc(log1p_, _unary_doc("log1p_", "torch.log1p", inplace=True))
    log2 = _set_doc(log2, _unary_doc("log2", "torch.log2", inplace=False))
    log2_ = _set_doc(log2_, _unary_doc("log2_", "torch.log2", inplace=True))
    neg = _set_doc(neg, _unary_doc("neg", "torch.neg", inplace=False))
    neg_ = _set_doc(neg_, _unary_doc("neg_", "torch.neg", inplace=True))
    reciprocal = _set_doc(
        reciprocal,
        _unary_doc("reciprocal", "torch.reciprocal", inplace=False),
    )
    reciprocal_ = _set_doc(
        reciprocal_,
        _unary_doc("reciprocal_", "torch.reciprocal", inplace=True),
    )
    round = _set_doc(
        round,
        _unary_doc(
            "round",
            "torch.round",
            inplace=False,
            note="The ``decimals`` argument is not supported.",
        ),
    )
    round_ = _set_doc(
        round_,
        _unary_doc(
            "round_",
            "torch.round",
            inplace=True,
            note="The ``decimals`` argument is not supported.",
        ),
    )
    rsqrt = _set_doc(rsqrt, _unary_doc("rsqrt", "torch.rsqrt", inplace=False))
    rsqrt_ = _set_doc(rsqrt_, _unary_doc("rsqrt_", "torch.rsqrt", inplace=True))
    sigmoid = _set_doc(sigmoid, _unary_doc("sigmoid", "torch.sigmoid", inplace=False))
    sigmoid_ = _set_doc(sigmoid_, _unary_doc("sigmoid_", "torch.sigmoid", inplace=True))
    sign = _set_doc(sign, _unary_doc("sign", "torch.sign", inplace=False))
    sign_ = _set_doc(sign_, _unary_doc("sign_", "torch.sign", inplace=True))
    sin = _set_doc(sin, _unary_doc("sin", "torch.sin", inplace=False))
    sin_ = _set_doc(sin_, _unary_doc("sin_", "torch.sin", inplace=True))
    sinh = _set_doc(sinh, _unary_doc("sinh", "torch.sinh", inplace=False))
    sinh_ = _set_doc(sinh_, _unary_doc("sinh_", "torch.sinh", inplace=True))
    sqrt = _set_doc(sqrt, _unary_doc("sqrt", "torch.sqrt", inplace=False))
    sqrt_ = _set_doc(sqrt_, _unary_doc("sqrt_", "torch.sqrt", inplace=True))
    tan = _set_doc(tan, _unary_doc("tan", "torch.tan", inplace=False))
    tan_ = _set_doc(tan_, _unary_doc("tan_", "torch.tan", inplace=True))
    tanh = _set_doc(tanh, _unary_doc("tanh", "torch.tanh", inplace=False))
    tanh_ = _set_doc(tanh_, _unary_doc("tanh_", "torch.tanh", inplace=True))
    trunc = _set_doc(trunc, _unary_doc("trunc", "torch.trunc", inplace=False))
    trunc_ = _set_doc(trunc_, _unary_doc("trunc_", "torch.trunc", inplace=True))

    clone = _set_doc(
        clone,
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

    add = _set_doc(
        add,
        _binary_doc(
            "add",
            "torch.add",
            (
                "inputs, other: Scalar",
                "inputs, other: ScalarList",
                "inputs, other: TensorList, *, alpha=1",
                "inputs, other: Tensor, *, alpha=1",
            ),
            inplace=False,
            alpha=(
                "supported only when ``other`` is a tensor list or a shared 0-D "
                "scalar tensor. Default: ``1``."
            ),
            shared_tensor=True,
        ),
    )
    add_ = _set_doc(
        add_,
        _binary_doc(
            "add_",
            "torch.add",
            (
                "inputs, other: Scalar",
                "inputs, other: ScalarList",
                "inputs, other: TensorList, *, alpha=1",
                "inputs, other: Tensor, *, alpha=1",
            ),
            inplace=True,
            alpha=(
                "supported only when ``other`` is a tensor list or a shared 0-D "
                "scalar tensor. Default: ``1``."
            ),
            shared_tensor=True,
        ),
    )
    sub = _set_doc(
        sub,
        _binary_doc(
            "sub",
            "torch.sub",
            (
                "inputs, other: Scalar",
                "inputs, other: ScalarList",
                "inputs, other: TensorList, *, alpha=1",
            ),
            inplace=False,
            alpha="supported only when ``other`` is a tensor list. Default: ``1``.",
        ),
    )
    sub_ = _set_doc(
        sub_,
        _binary_doc(
            "sub_",
            "torch.sub",
            (
                "inputs, other: Scalar",
                "inputs, other: ScalarList",
                "inputs, other: TensorList, *, alpha=1",
            ),
            inplace=True,
            alpha="supported only when ``other`` is a tensor list. Default: ``1``.",
        ),
    )
    mul = _set_doc(
        mul,
        _binary_doc(
            "mul",
            "torch.mul",
            (
                "inputs, other: Scalar",
                "inputs, other: ScalarList",
                "inputs, other: TensorList",
                "inputs, other: Tensor",
            ),
            inplace=False,
            shared_tensor=True,
        ),
    )
    mul_ = _set_doc(
        mul_,
        _binary_doc(
            "mul_",
            "torch.mul",
            (
                "inputs, other: Scalar",
                "inputs, other: ScalarList",
                "inputs, other: TensorList",
                "inputs, other: Tensor",
            ),
            inplace=True,
            shared_tensor=True,
        ),
    )
    div = _set_doc(
        div,
        _binary_doc(
            "div",
            "torch.div",
            (
                "inputs, other: Scalar",
                "inputs, other: ScalarList",
                "inputs, other: TensorList",
                "inputs, other: Tensor",
            ),
            inplace=False,
            shared_tensor=True,
            operand_note="The ``rounding_mode`` argument is not supported.",
        ),
    )
    div_ = _set_doc(
        div_,
        _binary_doc(
            "div_",
            "torch.div",
            (
                "inputs, other: Scalar",
                "inputs, other: ScalarList",
                "inputs, other: TensorList",
                "inputs, other: Tensor",
            ),
            inplace=True,
            shared_tensor=True,
            operand_note="The ``rounding_mode`` argument is not supported.",
        ),
    )

    clamp_min = _set_doc(
        clamp_min,
        _binary_doc(
            "clamp_min",
            "torch.clamp_min",
            (
                "inputs, min: Scalar",
                "inputs, min: ScalarList",
                "inputs, min: TensorList",
            ),
            inplace=False,
            operand="min",
        ),
    )
    clamp_min_ = _set_doc(
        clamp_min_,
        _binary_doc(
            "clamp_min_",
            "torch.clamp_min",
            (
                "inputs, min: Scalar",
                "inputs, min: ScalarList",
                "inputs, min: TensorList",
            ),
            inplace=True,
            operand="min",
        ),
    )
    clamp_max = _set_doc(
        clamp_max,
        _binary_doc(
            "clamp_max",
            "torch.clamp_max",
            (
                "inputs, max: Scalar",
                "inputs, max: ScalarList",
                "inputs, max: TensorList",
            ),
            inplace=False,
            operand="max",
        ),
    )
    clamp_max_ = _set_doc(
        clamp_max_,
        _binary_doc(
            "clamp_max_",
            "torch.clamp_max",
            (
                "inputs, max: Scalar",
                "inputs, max: ScalarList",
                "inputs, max: TensorList",
            ),
            inplace=True,
            operand="max",
        ),
    )
    minimum = _set_doc(
        minimum,
        _binary_doc(
            "minimum",
            "torch.minimum",
            (
                "inputs, other: Scalar",
                "inputs, other: ScalarList",
                "inputs, other: TensorList",
            ),
            inplace=False,
            operand_note=(
                "Scalar and ``ScalarList`` operands use :func:`torch.clamp_max` "
                "semantics."
            ),
        ),
    )
    minimum_ = _set_doc(
        minimum_,
        _binary_doc(
            "minimum_",
            "torch.minimum",
            (
                "inputs, other: Scalar",
                "inputs, other: ScalarList",
                "inputs, other: TensorList",
            ),
            inplace=True,
            operand_note=(
                "Scalar and ``ScalarList`` operands use :func:`torch.clamp_max` "
                "semantics."
            ),
        ),
    )
    maximum = _set_doc(
        maximum,
        _binary_doc(
            "maximum",
            "torch.maximum",
            (
                "inputs, other: Scalar",
                "inputs, other: ScalarList",
                "inputs, other: TensorList",
            ),
            inplace=False,
            operand_note=(
                "Scalar and ``ScalarList`` operands use :func:`torch.clamp_min` "
                "semantics."
            ),
        ),
    )
    maximum_ = _set_doc(
        maximum_,
        _binary_doc(
            "maximum_",
            "torch.maximum",
            (
                "inputs, other: Scalar",
                "inputs, other: ScalarList",
                "inputs, other: TensorList",
            ),
            inplace=True,
            operand_note=(
                "Scalar and ``ScalarList`` operands use :func:`torch.clamp_min` "
                "semantics."
            ),
        ),
    )

    pow = _set_doc(
        pow,
        rf"""
pow(input, exponent) -> tuple[Tensor, ...]

Applies :func:`torch.pow` at each list position.

{_common("torch.pow", False)}

Supported signatures::

    pow(input: TensorList, exponent: Scalar)
    pow(input: TensorList, exponent: ScalarList)
    pow(input: TensorList, exponent: TensorList)
    pow(input: Scalar, exponent: TensorList)

``TensorList`` and ``ScalarList`` denote a list or tuple of tensors and
scalars, respectively.

Args:
    input (Number, list of Tensor, or tuple of Tensor): bases.
    exponent (Number, list or tuple of Number, or list or tuple of Tensor): exponents.

Returns:
    a tuple containing one result tensor per list position.
""",
    )
    pow_ = _set_doc(
        pow_,
        rf"""
pow_(inputs, exponent) -> tuple[Tensor, ...] | list[Tensor]

In-place version of :func:`torch.foreach.pow`.

{_common("torch.pow", True)}

Supported signatures::

    pow_(inputs: TensorList, exponent: Scalar)
    pow_(inputs: TensorList, exponent: ScalarList)
    pow_(inputs: TensorList, exponent: TensorList)

``TensorList`` and ``ScalarList`` denote a list or tuple of tensors and
scalars, respectively. The scalar-left form has no in-place variant.

Args:
    inputs (list or tuple of Tensor): bases to mutate.
    exponent (Number, list or tuple of Number, or list or tuple of Tensor): exponents.

Returns:
    the exact ``inputs`` list or tuple.
""",
    )

    addcmul = _set_doc(
        addcmul,
        _pointwise_doc("addcmul", "torch.addcmul", inplace=False),
    )
    addcmul_ = _set_doc(
        addcmul_,
        _pointwise_doc("addcmul_", "torch.addcmul", inplace=True),
    )
    addcdiv = _set_doc(
        addcdiv,
        _pointwise_doc("addcdiv", "torch.addcdiv", inplace=False),
    )
    addcdiv_ = _set_doc(
        addcdiv_,
        _pointwise_doc("addcdiv_", "torch.addcdiv", inplace=True),
    )

    lerp = _set_doc(
        lerp,
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
    lerp_ = _set_doc(
        lerp_,
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

    max = _set_doc(
        max,
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
    norm = _set_doc(
        norm,
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

    copy_ = _set_doc(
        copy_,
        r"""
copy_(inputs, srcs, non_blocking=False) -> tuple[Tensor, ...] | list[Tensor]

Copies each tensor in ``srcs`` into the corresponding tensor in
``inputs``, following :meth:`torch.Tensor.copy_`.

Both lists must be non-empty and have the same length. The operation mutates
the tensors in ``inputs`` and returns the exact ``inputs`` container. There is
no functional ``torch.foreach.copy`` operation.

Args:
    inputs (list or tuple of Tensor): destination tensors.
    srcs (list or tuple of Tensor): source tensors.
    non_blocking (bool, optional): allows asynchronous host/device copies when
        supported. Default: ``False``.

Returns:
    the exact ``inputs`` list or tuple.
""",
    )
    zero_ = _set_doc(
        zero_,
        r"""
zero_(inputs) -> tuple[Tensor, ...] | list[Tensor]

Fills every tensor in ``inputs`` with zero.

The operation is equivalent to calling :meth:`torch.Tensor.zero_` on every
tensor and returns the exact ``inputs`` container. There is no functional
``torch.foreach.zero`` operation.

Args:
    inputs (list or tuple of Tensor): tensors to zero.

Returns:
    the exact ``inputs`` list or tuple.
""",
    )
    mm = _set_doc(
        mm,
        r"""
mm(inputs, mat2s) -> tuple[Tensor, ...]

Multiplies corresponding matrices from ``inputs`` and ``mat2s`` using
:func:`torch.mm`.

Both lists must be non-empty and have the same length. Compatible calls may use
a grouped matrix multiplication implementation; all other supported calls use a
loop of :func:`torch.mm`. There is no in-place ``torch.foreach.mm_`` operation.

Args:
    inputs (list or tuple of Tensor): first matrices.
    mat2s (list or tuple of Tensor): second matrices.

Returns:
    a tuple containing one matrix product per list position.
""",
    )
