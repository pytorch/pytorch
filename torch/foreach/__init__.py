r"""Operations over lists of tensors.

.. warning::
    ``torch.foreach`` is a beta API. Its signatures may change based on user
    feedback. Existing private ``torch._foreach_*`` functions remain available
    for compatibility during migration.

Each function applies the corresponding ordinary PyTorch operation to every
position in one or more tensor lists.

The functions will use an accelerated multi-tensor implementation when their
inputs meet its requirements. Otherwise they use a semantically equivalent
per-tensor fallback. Calling a function in this module does not guarantee a
single or fused kernel.
"""

from __future__ import annotations

import inspect
from typing import overload, ParamSpec, TYPE_CHECKING, TypeVar

import torch
from torch import Tensor
from torch.overrides import wrap_torch_function
from torch.types import _complex, Number, PySymType


if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any


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


_P = ParamSpec("_P")
_R = TypeVar("_R")
_ScalarT = TypeVar("_ScalarT", bound=Number | _complex | PySymType)
_Scalar = Number | _complex | PySymType
_ScalarList = tuple[_ScalarT, ...] | list[_ScalarT]
_TensorList = tuple[Tensor, ...] | list[Tensor]
_TensorTuple = tuple[Tensor, ...]


class _DefaultAlpha:
    def __repr__(self) -> str:
        return "1"


_DEFAULT_ALPHA = _DefaultAlpha()


def _public(doc: str) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    def decorator(func: Callable[_P, _R]) -> Callable[_P, _R]:
        signature = inspect.signature(func)

        def dispatcher(*args: _P.args, **kwargs: _P.kwargs) -> list[object]:
            try:
                bound = signature.bind(*args, **kwargs)
            except TypeError:
                # Preserve normal Python call errors before override dispatch.
                func(*args, **kwargs)
                return []
            relevant_args = []
            for arg in bound.arguments.values():
                if isinstance(arg, (list, tuple)):
                    relevant_args.extend(arg)
                else:
                    relevant_args.append(arg)
            return relevant_args

        parameters = [
            parameter.replace(
                default=1 if parameter.default is _DEFAULT_ALPHA else parameter.default,
                annotation=inspect.Parameter.empty,
            )
            for parameter in signature.parameters.values()
        ]
        func.__doc__ = doc
        wrapped = wrap_torch_function(dispatcher)(func)
        wrapped.__annotations__ = {}
        wrapped.__signature__ = signature.replace(  # type: ignore[attr-defined]
            parameters=parameters,
            return_annotation=inspect.Signature.empty,
        )
        return wrapped

    return decorator


def _result(in_place: bool) -> str:
    if in_place:
        return "the exact input list or tuple"
    return "a tuple containing one result tensor for each input tensor"


def _common(reference: str, in_place: bool, *, has_aligned_lists: bool = False) -> str:
    mutation = (
        "Mutates every tensor in ``inputs`` and returns the exact input "
        "container object."
        if in_place
        else "Does not mutate its arguments and returns a tuple of result tensors."
    )
    length_requirement = (
        "\nCorresponding tensor or scalar lists must have the same length."
        if has_aligned_lists
        else ""
    )
    return rf"""
This is semantically equivalent to applying :func:`{reference}` independently
at every list position. {mutation}

Tensor-list arguments must be non-empty.{length_requirement}
An accelerated multi-tensor implementation is used only when supported by the
inputs; otherwise the operation falls back to per-tensor execution.
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

{_common(reference, inplace, has_aligned_lists=True)}

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
{name}(inputs, tensor1, tensor2, *, value=1) -> {suffix}

Applies :func:`{reference}` to corresponding tensors from the three input
lists.

{_common(reference, inplace, has_aligned_lists=True)}

``value`` may be one shared scalar, a scalar list or tuple, or a packed 1-D
CPU tensor containing one scalar per list position.

Args:
    inputs (list or tuple of Tensor): tensors to transform.
    tensor1 (list or tuple of Tensor): first multiplicative or divisive operands.
    tensor2 (list or tuple of Tensor): second multiplicative or divisive operands.
    value (Number, list or tuple of Number, or Tensor, optional): scale values.
        Default: ``1``.

Returns:
    {_result(inplace)}.
"""


# Unary operations


@_public(_unary_doc("abs", "torch.abs", inplace=False))
def abs(inputs: _TensorList) -> _TensorTuple:
    return torch._foreach_abs(inputs)


@_public(_unary_doc("abs_", "torch.abs", inplace=True))
def abs_(inputs: _TensorList) -> _TensorList:
    return torch._foreach_abs_(inputs)


@_public(_unary_doc("acos", "torch.acos", inplace=False))
def acos(inputs: _TensorList) -> _TensorTuple:
    return torch._foreach_acos(inputs)


@_public(_unary_doc("acos_", "torch.acos", inplace=True))
def acos_(inputs: _TensorList) -> _TensorList:
    return torch._foreach_acos_(inputs)


@_public(_unary_doc("asin", "torch.asin", inplace=False))
def asin(inputs: _TensorList) -> _TensorTuple:
    return torch._foreach_asin(inputs)


@_public(_unary_doc("asin_", "torch.asin", inplace=True))
def asin_(inputs: _TensorList) -> _TensorList:
    return torch._foreach_asin_(inputs)


@_public(_unary_doc("atan", "torch.atan", inplace=False))
def atan(inputs: _TensorList) -> _TensorTuple:
    return torch._foreach_atan(inputs)


@_public(_unary_doc("atan_", "torch.atan", inplace=True))
def atan_(inputs: _TensorList) -> _TensorList:
    return torch._foreach_atan_(inputs)


@_public(_unary_doc("ceil", "torch.ceil", inplace=False))
def ceil(inputs: _TensorList) -> _TensorTuple:
    return torch._foreach_ceil(inputs)


@_public(_unary_doc("ceil_", "torch.ceil", inplace=True))
def ceil_(inputs: _TensorList) -> _TensorList:
    return torch._foreach_ceil_(inputs)


@_public(_unary_doc("cos", "torch.cos", inplace=False))
def cos(inputs: _TensorList) -> _TensorTuple:
    return torch._foreach_cos(inputs)


@_public(_unary_doc("cos_", "torch.cos", inplace=True))
def cos_(inputs: _TensorList) -> _TensorList:
    return torch._foreach_cos_(inputs)


@_public(_unary_doc("cosh", "torch.cosh", inplace=False))
def cosh(inputs: _TensorList) -> _TensorTuple:
    return torch._foreach_cosh(inputs)


@_public(_unary_doc("cosh_", "torch.cosh", inplace=True))
def cosh_(inputs: _TensorList) -> _TensorList:
    return torch._foreach_cosh_(inputs)


@_public(_unary_doc("erf", "torch.erf", inplace=False))
def erf(inputs: _TensorList) -> _TensorTuple:
    return torch._foreach_erf(inputs)


@_public(_unary_doc("erf_", "torch.erf", inplace=True))
def erf_(inputs: _TensorList) -> _TensorList:
    return torch._foreach_erf_(inputs)


@_public(_unary_doc("erfc", "torch.erfc", inplace=False))
def erfc(inputs: _TensorList) -> _TensorTuple:
    return torch._foreach_erfc(inputs)


@_public(_unary_doc("erfc_", "torch.erfc", inplace=True))
def erfc_(inputs: _TensorList) -> _TensorList:
    return torch._foreach_erfc_(inputs)


@_public(_unary_doc("exp", "torch.exp", inplace=False))
def exp(inputs: _TensorList) -> _TensorTuple:
    return torch._foreach_exp(inputs)


@_public(_unary_doc("exp_", "torch.exp", inplace=True))
def exp_(inputs: _TensorList) -> _TensorList:
    return torch._foreach_exp_(inputs)


@_public(_unary_doc("expm1", "torch.expm1", inplace=False))
def expm1(inputs: _TensorList) -> _TensorTuple:
    return torch._foreach_expm1(inputs)


@_public(_unary_doc("expm1_", "torch.expm1", inplace=True))
def expm1_(inputs: _TensorList) -> _TensorList:
    return torch._foreach_expm1_(inputs)


@_public(_unary_doc("floor", "torch.floor", inplace=False))
def floor(inputs: _TensorList) -> _TensorTuple:
    return torch._foreach_floor(inputs)


@_public(_unary_doc("floor_", "torch.floor", inplace=True))
def floor_(inputs: _TensorList) -> _TensorList:
    return torch._foreach_floor_(inputs)


@_public(_unary_doc("frac", "torch.frac", inplace=False))
def frac(inputs: _TensorList) -> _TensorTuple:
    return torch._foreach_frac(inputs)


@_public(_unary_doc("frac_", "torch.frac", inplace=True))
def frac_(inputs: _TensorList) -> _TensorList:
    return torch._foreach_frac_(inputs)


@_public(_unary_doc("lgamma", "torch.lgamma", inplace=False))
def lgamma(inputs: _TensorList) -> _TensorTuple:
    return torch._foreach_lgamma(inputs)


@_public(_unary_doc("lgamma_", "torch.lgamma", inplace=True))
def lgamma_(inputs: _TensorList) -> _TensorList:
    return torch._foreach_lgamma_(inputs)


@_public(_unary_doc("log", "torch.log", inplace=False))
def log(inputs: _TensorList) -> _TensorTuple:
    return torch._foreach_log(inputs)


@_public(_unary_doc("log_", "torch.log", inplace=True))
def log_(inputs: _TensorList) -> _TensorList:
    return torch._foreach_log_(inputs)


@_public(_unary_doc("log10", "torch.log10", inplace=False))
def log10(inputs: _TensorList) -> _TensorTuple:
    return torch._foreach_log10(inputs)


@_public(_unary_doc("log10_", "torch.log10", inplace=True))
def log10_(inputs: _TensorList) -> _TensorList:
    return torch._foreach_log10_(inputs)


@_public(_unary_doc("log1p", "torch.log1p", inplace=False))
def log1p(inputs: _TensorList) -> _TensorTuple:
    return torch._foreach_log1p(inputs)


@_public(_unary_doc("log1p_", "torch.log1p", inplace=True))
def log1p_(inputs: _TensorList) -> _TensorList:
    return torch._foreach_log1p_(inputs)


@_public(_unary_doc("log2", "torch.log2", inplace=False))
def log2(inputs: _TensorList) -> _TensorTuple:
    return torch._foreach_log2(inputs)


@_public(_unary_doc("log2_", "torch.log2", inplace=True))
def log2_(inputs: _TensorList) -> _TensorList:
    return torch._foreach_log2_(inputs)


@_public(_unary_doc("neg", "torch.neg", inplace=False))
def neg(inputs: _TensorList) -> _TensorTuple:
    return torch._foreach_neg(inputs)


@_public(_unary_doc("neg_", "torch.neg", inplace=True))
def neg_(inputs: _TensorList) -> _TensorList:
    return torch._foreach_neg_(inputs)


@_public(_unary_doc("reciprocal", "torch.reciprocal", inplace=False))
def reciprocal(inputs: _TensorList) -> _TensorTuple:
    return torch._foreach_reciprocal(inputs)


@_public(_unary_doc("reciprocal_", "torch.reciprocal", inplace=True))
def reciprocal_(inputs: _TensorList) -> _TensorList:
    return torch._foreach_reciprocal_(inputs)


@_public(
    _unary_doc(
        "round",
        "torch.round",
        inplace=False,
        note="The ``decimals`` argument is not supported.",
    )
)
def round(inputs: _TensorList) -> _TensorTuple:
    return torch._foreach_round(inputs)


@_public(
    _unary_doc(
        "round_",
        "torch.round",
        inplace=True,
        note="The ``decimals`` argument is not supported.",
    )
)
def round_(inputs: _TensorList) -> _TensorList:
    return torch._foreach_round_(inputs)


@_public(_unary_doc("rsqrt", "torch.rsqrt", inplace=False))
def rsqrt(inputs: _TensorList) -> _TensorTuple:
    return torch._foreach_rsqrt(inputs)


@_public(_unary_doc("rsqrt_", "torch.rsqrt", inplace=True))
def rsqrt_(inputs: _TensorList) -> _TensorList:
    return torch._foreach_rsqrt_(inputs)


@_public(_unary_doc("sigmoid", "torch.sigmoid", inplace=False))
def sigmoid(inputs: _TensorList) -> _TensorTuple:
    return torch._foreach_sigmoid(inputs)


@_public(_unary_doc("sigmoid_", "torch.sigmoid", inplace=True))
def sigmoid_(inputs: _TensorList) -> _TensorList:
    return torch._foreach_sigmoid_(inputs)


@_public(_unary_doc("sign", "torch.sign", inplace=False))
def sign(inputs: _TensorList) -> _TensorTuple:
    return torch._foreach_sign(inputs)


@_public(_unary_doc("sign_", "torch.sign", inplace=True))
def sign_(inputs: _TensorList) -> _TensorList:
    return torch._foreach_sign_(inputs)


@_public(_unary_doc("sin", "torch.sin", inplace=False))
def sin(inputs: _TensorList) -> _TensorTuple:
    return torch._foreach_sin(inputs)


@_public(_unary_doc("sin_", "torch.sin", inplace=True))
def sin_(inputs: _TensorList) -> _TensorList:
    return torch._foreach_sin_(inputs)


@_public(_unary_doc("sinh", "torch.sinh", inplace=False))
def sinh(inputs: _TensorList) -> _TensorTuple:
    return torch._foreach_sinh(inputs)


@_public(_unary_doc("sinh_", "torch.sinh", inplace=True))
def sinh_(inputs: _TensorList) -> _TensorList:
    return torch._foreach_sinh_(inputs)


@_public(_unary_doc("sqrt", "torch.sqrt", inplace=False))
def sqrt(inputs: _TensorList) -> _TensorTuple:
    return torch._foreach_sqrt(inputs)


@_public(_unary_doc("sqrt_", "torch.sqrt", inplace=True))
def sqrt_(inputs: _TensorList) -> _TensorList:
    return torch._foreach_sqrt_(inputs)


@_public(_unary_doc("tan", "torch.tan", inplace=False))
def tan(inputs: _TensorList) -> _TensorTuple:
    return torch._foreach_tan(inputs)


@_public(_unary_doc("tan_", "torch.tan", inplace=True))
def tan_(inputs: _TensorList) -> _TensorList:
    return torch._foreach_tan_(inputs)


@_public(_unary_doc("tanh", "torch.tanh", inplace=False))
def tanh(inputs: _TensorList) -> _TensorTuple:
    return torch._foreach_tanh(inputs)


@_public(_unary_doc("tanh_", "torch.tanh", inplace=True))
def tanh_(inputs: _TensorList) -> _TensorList:
    return torch._foreach_tanh_(inputs)


@_public(_unary_doc("trunc", "torch.trunc", inplace=False))
def trunc(inputs: _TensorList) -> _TensorTuple:
    return torch._foreach_trunc(inputs)


@_public(_unary_doc("trunc_", "torch.trunc", inplace=True))
def trunc_(inputs: _TensorList) -> _TensorList:
    return torch._foreach_trunc_(inputs)


# Binary operations


@overload
def add(inputs: _TensorList, other: _Scalar) -> _TensorTuple: ...


@overload
def add(inputs: _TensorList, other: _ScalarList[_ScalarT]) -> _TensorTuple: ...


@overload
def add(
    inputs: _TensorList,
    other: Tensor,
    *,
    alpha: _Scalar = 1,
) -> _TensorTuple: ...


@overload
def add(
    inputs: _TensorList,
    other: _TensorList,
    *,
    alpha: _Scalar = 1,
) -> _TensorTuple: ...


@_public(
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
    )
)
def add(
    inputs: _TensorList,
    other: Any,
    *,
    alpha: Any = _DEFAULT_ALPHA,
) -> _TensorTuple:
    if alpha is _DEFAULT_ALPHA:
        return torch._foreach_add(inputs, other)
    return torch._foreach_add(inputs, other, alpha=alpha)


@overload
def add_(inputs: _TensorList, other: _Scalar) -> _TensorList: ...


@overload
def add_(inputs: _TensorList, other: _ScalarList[_ScalarT]) -> _TensorList: ...


@overload
def add_(
    inputs: _TensorList,
    other: Tensor,
    *,
    alpha: _Scalar = 1,
) -> _TensorList: ...


@overload
def add_(
    inputs: _TensorList,
    other: _TensorList,
    *,
    alpha: _Scalar = 1,
) -> _TensorList: ...


@_public(
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
    )
)
def add_(
    inputs: _TensorList,
    other: Any,
    *,
    alpha: Any = _DEFAULT_ALPHA,
) -> _TensorList:
    if alpha is _DEFAULT_ALPHA:
        return torch._foreach_add_(inputs, other)
    return torch._foreach_add_(inputs, other, alpha=alpha)


@overload
def sub(inputs: _TensorList, other: _Scalar) -> _TensorTuple: ...


@overload
def sub(inputs: _TensorList, other: _ScalarList[_ScalarT]) -> _TensorTuple: ...


@overload
def sub(
    inputs: _TensorList,
    other: _TensorList,
    *,
    alpha: _Scalar = 1,
) -> _TensorTuple: ...


@_public(
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
    )
)
def sub(
    inputs: _TensorList,
    other: Any,
    *,
    alpha: Any = _DEFAULT_ALPHA,
) -> _TensorTuple:
    if alpha is _DEFAULT_ALPHA:
        return torch._foreach_sub(inputs, other)
    return torch._foreach_sub(inputs, other, alpha=alpha)


@overload
def sub_(inputs: _TensorList, other: _Scalar) -> _TensorList: ...


@overload
def sub_(inputs: _TensorList, other: _ScalarList[_ScalarT]) -> _TensorList: ...


@overload
def sub_(
    inputs: _TensorList,
    other: _TensorList,
    *,
    alpha: _Scalar = 1,
) -> _TensorList: ...


@_public(
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
    )
)
def sub_(
    inputs: _TensorList,
    other: Any,
    *,
    alpha: Any = _DEFAULT_ALPHA,
) -> _TensorList:
    if alpha is _DEFAULT_ALPHA:
        return torch._foreach_sub_(inputs, other)
    return torch._foreach_sub_(inputs, other, alpha=alpha)


@overload
def mul(inputs: _TensorList, other: _ScalarList[_ScalarT]) -> _TensorTuple: ...


@overload
def mul(inputs: _TensorList, other: Tensor) -> _TensorTuple: ...


@overload
def mul(inputs: _TensorList, other: _TensorList) -> _TensorTuple: ...


@overload
def mul(inputs: _TensorList, other: _Scalar) -> _TensorTuple: ...


@_public(
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
    )
)
def mul(inputs: _TensorList, other: Any) -> _TensorTuple:
    return torch._foreach_mul(inputs, other)


@overload
def mul_(inputs: _TensorList, other: _ScalarList[_ScalarT]) -> _TensorList: ...


@overload
def mul_(inputs: _TensorList, other: Tensor) -> _TensorList: ...


@overload
def mul_(inputs: _TensorList, other: _TensorList) -> _TensorList: ...


@overload
def mul_(inputs: _TensorList, other: _Scalar) -> _TensorList: ...


@_public(
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
    )
)
def mul_(inputs: _TensorList, other: Any) -> _TensorList:
    return torch._foreach_mul_(inputs, other)


@overload
def div(inputs: _TensorList, other: _ScalarList[_ScalarT]) -> _TensorTuple: ...


@overload
def div(inputs: _TensorList, other: Tensor) -> _TensorTuple: ...


@overload
def div(inputs: _TensorList, other: _TensorList) -> _TensorTuple: ...


@overload
def div(inputs: _TensorList, other: _Scalar) -> _TensorTuple: ...


@_public(
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
    )
)
def div(inputs: _TensorList, other: Any) -> _TensorTuple:
    return torch._foreach_div(inputs, other)


@overload
def div_(inputs: _TensorList, other: _ScalarList[_ScalarT]) -> _TensorList: ...


@overload
def div_(inputs: _TensorList, other: Tensor) -> _TensorList: ...


@overload
def div_(inputs: _TensorList, other: _TensorList) -> _TensorList: ...


@overload
def div_(inputs: _TensorList, other: _Scalar) -> _TensorList: ...


@_public(
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
    )
)
def div_(inputs: _TensorList, other: Any) -> _TensorList:
    return torch._foreach_div_(inputs, other)


@overload
def clamp_min(inputs: _TensorList, min: _Scalar) -> _TensorTuple: ...


@overload
def clamp_min(inputs: _TensorList, min: _ScalarList[_ScalarT]) -> _TensorTuple: ...


@overload
def clamp_min(inputs: _TensorList, min: _TensorList) -> _TensorTuple: ...


@_public(
    _binary_doc(
        "clamp_min",
        "torch.clamp",
        (
            "inputs, min: Scalar",
            "inputs, min: ScalarList",
            "inputs, min: TensorList",
        ),
        inplace=False,
        operand="min",
        operand_note="Only the ``min`` bound is specified.",
    )
)
def clamp_min(inputs: _TensorList, min: Any) -> _TensorTuple:
    return torch._foreach_clamp_min(inputs, min)


@overload
def clamp_min_(inputs: _TensorList, min: _Scalar) -> _TensorList: ...


@overload
def clamp_min_(inputs: _TensorList, min: _ScalarList[_ScalarT]) -> _TensorList: ...


@overload
def clamp_min_(inputs: _TensorList, min: _TensorList) -> _TensorList: ...


@_public(
    _binary_doc(
        "clamp_min_",
        "torch.clamp",
        (
            "inputs, min: Scalar",
            "inputs, min: ScalarList",
            "inputs, min: TensorList",
        ),
        inplace=True,
        operand="min",
        operand_note="Only the ``min`` bound is specified.",
    )
)
def clamp_min_(inputs: _TensorList, min: Any) -> _TensorList:
    return torch._foreach_clamp_min_(inputs, min)


@overload
def clamp_max(inputs: _TensorList, max: _Scalar) -> _TensorTuple: ...


@overload
def clamp_max(inputs: _TensorList, max: _ScalarList[_ScalarT]) -> _TensorTuple: ...


@overload
def clamp_max(inputs: _TensorList, max: _TensorList) -> _TensorTuple: ...


@_public(
    _binary_doc(
        "clamp_max",
        "torch.clamp",
        (
            "inputs, max: Scalar",
            "inputs, max: ScalarList",
            "inputs, max: TensorList",
        ),
        inplace=False,
        operand="max",
        operand_note="Only the ``max`` bound is specified.",
    )
)
def clamp_max(inputs: _TensorList, max: Any) -> _TensorTuple:
    return torch._foreach_clamp_max(inputs, max)


@overload
def clamp_max_(inputs: _TensorList, max: _Scalar) -> _TensorList: ...


@overload
def clamp_max_(inputs: _TensorList, max: _ScalarList[_ScalarT]) -> _TensorList: ...


@overload
def clamp_max_(inputs: _TensorList, max: _TensorList) -> _TensorList: ...


@_public(
    _binary_doc(
        "clamp_max_",
        "torch.clamp",
        (
            "inputs, max: Scalar",
            "inputs, max: ScalarList",
            "inputs, max: TensorList",
        ),
        inplace=True,
        operand="max",
        operand_note="Only the ``max`` bound is specified.",
    )
)
def clamp_max_(inputs: _TensorList, max: Any) -> _TensorList:
    return torch._foreach_clamp_max_(inputs, max)


@overload
def minimum(inputs: _TensorList, other: _Scalar) -> _TensorTuple: ...


@overload
def minimum(inputs: _TensorList, other: _ScalarList[_ScalarT]) -> _TensorTuple: ...


@overload
def minimum(inputs: _TensorList, other: _TensorList) -> _TensorTuple: ...


@_public(
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
            "Scalar and ``ScalarList`` operands use :func:`torch.clamp` with "
            "only ``max`` specified."
        ),
    )
)
def minimum(inputs: _TensorList, other: Any) -> _TensorTuple:
    return torch._foreach_minimum(inputs, other)


@overload
def minimum_(inputs: _TensorList, other: _Scalar) -> _TensorList: ...


@overload
def minimum_(inputs: _TensorList, other: _ScalarList[_ScalarT]) -> _TensorList: ...


@overload
def minimum_(inputs: _TensorList, other: _TensorList) -> _TensorList: ...


@_public(
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
            "Scalar and ``ScalarList`` operands use :func:`torch.clamp` with "
            "only ``max`` specified."
        ),
    )
)
def minimum_(inputs: _TensorList, other: Any) -> _TensorList:
    return torch._foreach_minimum_(inputs, other)


@overload
def maximum(inputs: _TensorList, other: _Scalar) -> _TensorTuple: ...


@overload
def maximum(inputs: _TensorList, other: _ScalarList[_ScalarT]) -> _TensorTuple: ...


@overload
def maximum(inputs: _TensorList, other: _TensorList) -> _TensorTuple: ...


@_public(
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
            "Scalar and ``ScalarList`` operands use :func:`torch.clamp` with "
            "only ``min`` specified."
        ),
    )
)
def maximum(inputs: _TensorList, other: Any) -> _TensorTuple:
    return torch._foreach_maximum(inputs, other)


@overload
def maximum_(inputs: _TensorList, other: _Scalar) -> _TensorList: ...


@overload
def maximum_(inputs: _TensorList, other: _ScalarList[_ScalarT]) -> _TensorList: ...


@overload
def maximum_(inputs: _TensorList, other: _TensorList) -> _TensorList: ...


@_public(
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
            "Scalar and ``ScalarList`` operands use :func:`torch.clamp` with "
            "only ``min`` specified."
        ),
    )
)
def maximum_(inputs: _TensorList, other: Any) -> _TensorList:
    return torch._foreach_maximum_(inputs, other)


# Pointwise operations


@overload
def addcmul(
    inputs: _TensorList,
    tensor1: _TensorList,
    tensor2: _TensorList,
    *,
    value: _ScalarList[_ScalarT],
) -> _TensorTuple: ...


@overload
def addcmul(
    inputs: _TensorList,
    tensor1: _TensorList,
    tensor2: _TensorList,
    *,
    value: Tensor,
) -> _TensorTuple: ...


@overload
def addcmul(
    inputs: _TensorList,
    tensor1: _TensorList,
    tensor2: _TensorList,
    *,
    value: _Scalar = 1,
) -> _TensorTuple: ...


@_public(_pointwise_doc("addcmul", "torch.addcmul", inplace=False))
def addcmul(
    inputs: _TensorList,
    tensor1: _TensorList,
    tensor2: _TensorList,
    *,
    value: Any = 1,
) -> _TensorTuple:
    return torch._foreach_addcmul(inputs, tensor1, tensor2, value)


@overload
def addcmul_(
    inputs: _TensorList,
    tensor1: _TensorList,
    tensor2: _TensorList,
    *,
    value: _ScalarList[_ScalarT],
) -> _TensorList: ...


@overload
def addcmul_(
    inputs: _TensorList,
    tensor1: _TensorList,
    tensor2: _TensorList,
    *,
    value: Tensor,
) -> _TensorList: ...


@overload
def addcmul_(
    inputs: _TensorList,
    tensor1: _TensorList,
    tensor2: _TensorList,
    *,
    value: _Scalar = 1,
) -> _TensorList: ...


@_public(_pointwise_doc("addcmul_", "torch.addcmul", inplace=True))
def addcmul_(
    inputs: _TensorList,
    tensor1: _TensorList,
    tensor2: _TensorList,
    *,
    value: Any = 1,
) -> _TensorList:
    return torch._foreach_addcmul_(inputs, tensor1, tensor2, value)


@overload
def addcdiv(
    inputs: _TensorList,
    tensor1: _TensorList,
    tensor2: _TensorList,
    *,
    value: _ScalarList[_ScalarT],
) -> _TensorTuple: ...


@overload
def addcdiv(
    inputs: _TensorList,
    tensor1: _TensorList,
    tensor2: _TensorList,
    *,
    value: Tensor,
) -> _TensorTuple: ...


@overload
def addcdiv(
    inputs: _TensorList,
    tensor1: _TensorList,
    tensor2: _TensorList,
    *,
    value: _Scalar = 1,
) -> _TensorTuple: ...


@_public(_pointwise_doc("addcdiv", "torch.addcdiv", inplace=False))
def addcdiv(
    inputs: _TensorList,
    tensor1: _TensorList,
    tensor2: _TensorList,
    *,
    value: Any = 1,
) -> _TensorTuple:
    return torch._foreach_addcdiv(inputs, tensor1, tensor2, value)


@overload
def addcdiv_(
    inputs: _TensorList,
    tensor1: _TensorList,
    tensor2: _TensorList,
    *,
    value: _ScalarList[_ScalarT],
) -> _TensorList: ...


@overload
def addcdiv_(
    inputs: _TensorList,
    tensor1: _TensorList,
    tensor2: _TensorList,
    *,
    value: Tensor,
) -> _TensorList: ...


@overload
def addcdiv_(
    inputs: _TensorList,
    tensor1: _TensorList,
    tensor2: _TensorList,
    *,
    value: _Scalar = 1,
) -> _TensorList: ...


@_public(_pointwise_doc("addcdiv_", "torch.addcdiv", inplace=True))
def addcdiv_(
    inputs: _TensorList,
    tensor1: _TensorList,
    tensor2: _TensorList,
    *,
    value: Any = 1,
) -> _TensorList:
    return torch._foreach_addcdiv_(inputs, tensor1, tensor2, value)


@overload
def lerp(inputs: _TensorList, end: _TensorList, weight: _Scalar) -> _TensorTuple: ...


@overload
def lerp(
    inputs: _TensorList,
    end: _TensorList,
    weight: _ScalarList[_ScalarT],
) -> _TensorTuple: ...


@overload
def lerp(
    inputs: _TensorList,
    end: _TensorList,
    weight: _TensorList,
) -> _TensorTuple: ...


@_public(
    rf"""
lerp(inputs, end, weight) -> tuple[Tensor, ...]

Applies :func:`torch.lerp` to corresponding tensors in ``inputs`` and
``end``.

{_common("torch.lerp", False, has_aligned_lists=True)}

``weight`` may be one shared scalar, a scalar list or tuple, or a tensor list.

Args:
    inputs (list or tuple of Tensor): starting tensors.
    end (list or tuple of Tensor): ending tensors.
    weight (Number, list or tuple of Number, or list or tuple of Tensor): interpolation
        weights.

Returns:
    a tuple containing one result tensor per list position.
"""
)
def lerp(inputs: _TensorList, end: _TensorList, weight: Any) -> _TensorTuple:
    return torch._foreach_lerp(inputs, end, weight)


@overload
def lerp_(inputs: _TensorList, end: _TensorList, weight: _Scalar) -> _TensorList: ...


@overload
def lerp_(
    inputs: _TensorList,
    end: _TensorList,
    weight: _ScalarList[_ScalarT],
) -> _TensorList: ...


@overload
def lerp_(
    inputs: _TensorList,
    end: _TensorList,
    weight: _TensorList,
) -> _TensorList: ...


@_public(
    rf"""
lerp_(inputs, end, weight) -> tuple[Tensor, ...] | list[Tensor]

In-place version of :func:`torch.foreach.lerp`.

{_common("torch.lerp", True, has_aligned_lists=True)}

``weight`` may be one shared scalar, a scalar list or tuple, or a tensor list.

Args:
    inputs (list or tuple of Tensor): starting tensors to mutate.
    end (list or tuple of Tensor): ending tensors.
    weight (Number, list or tuple of Number, or list or tuple of Tensor): interpolation
        weights.

Returns:
    the exact ``inputs`` list or tuple.
"""
)
def lerp_(inputs: _TensorList, end: _TensorList, weight: Any) -> _TensorList:
    return torch._foreach_lerp_(inputs, end, weight)


@overload
def pow(input: _Scalar, exponent: _TensorList) -> _TensorTuple: ...


@overload
def pow(input: _TensorList, exponent: _Scalar) -> _TensorTuple: ...


@overload
def pow(input: _TensorList, exponent: _ScalarList[_ScalarT]) -> _TensorTuple: ...


@overload
def pow(input: _TensorList, exponent: _TensorList) -> _TensorTuple: ...


@_public(
    rf"""
pow(input, exponent) -> tuple[Tensor, ...]

Applies :func:`torch.pow` at each list position.

{_common("torch.pow", False, has_aligned_lists=True)}

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
"""
)
def pow(input: Any, exponent: Any) -> _TensorTuple:
    return torch._foreach_pow(input, exponent)


@overload
def pow_(inputs: _TensorList, exponent: _Scalar) -> _TensorList: ...


@overload
def pow_(inputs: _TensorList, exponent: _ScalarList[_ScalarT]) -> _TensorList: ...


@overload
def pow_(inputs: _TensorList, exponent: _TensorList) -> _TensorList: ...


@_public(
    rf"""
pow_(inputs, exponent) -> tuple[Tensor, ...] | list[Tensor]

In-place version of :func:`torch.foreach.pow`.

{_common("torch.pow", True, has_aligned_lists=True)}

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
"""
)
def pow_(inputs: _TensorList, exponent: Any) -> _TensorList:
    return torch._foreach_pow_(inputs, exponent)


# Reductions and special operations


@_public(
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
"""
)
def clone(
    inputs: _TensorList,
    *,
    memory_format: torch.memory_format | None = torch.preserve_format,
) -> _TensorTuple:
    return torch._foreach_clone(inputs, memory_format=memory_format)


@_public(
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
"""
)
def max(inputs: _TensorList) -> _TensorTuple:
    return torch._foreach_max(inputs)


@_public(
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
"""
)
def norm(
    inputs: _TensorList,
    ord: _Scalar = 2,
    *,
    dtype: torch.dtype | None = None,
) -> _TensorTuple:
    return torch._foreach_norm(inputs, ord, dtype=dtype)


@_public(
    r"""
copy_(inputs, src, non_blocking=False) -> tuple[Tensor, ...] | list[Tensor]

Copies each tensor in ``src`` into the corresponding tensor in
``inputs``, following :meth:`torch.Tensor.copy_`.

Both lists must be non-empty and have the same length. The operation mutates
the tensors in ``inputs`` and returns the exact ``inputs`` container. There is
no functional ``torch.foreach.copy`` operation.

Args:
    inputs (list or tuple of Tensor): destination tensors.
    src (list or tuple of Tensor): source tensors.
    non_blocking (bool, optional): allows asynchronous host/device copies when
        supported. Default: ``False``.

Returns:
    the exact ``inputs`` list or tuple.
"""
)
def copy_(
    inputs: _TensorList,
    src: _TensorList,
    non_blocking: bool = False,
) -> _TensorList:
    return torch._foreach_copy_(inputs, src, non_blocking)


@_public(
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
"""
)
def zero_(inputs: _TensorList) -> _TensorList:
    return torch._foreach_zero_(inputs)


@_public(
    r"""
mm(inputs, mat2) -> tuple[Tensor, ...]

Multiplies corresponding matrices from ``inputs`` and ``mat2`` using
:func:`torch.mm`.

Both lists must be non-empty and have the same length. Compatible calls may use
a grouped matrix multiplication implementation; all other supported calls use a
loop of :func:`torch.mm`. There is no in-place ``torch.foreach.mm_`` operation.

Args:
    inputs (list or tuple of Tensor): first matrices.
    mat2 (list or tuple of Tensor): second matrices.

Returns:
    a tuple containing one matrix product per list position.
"""
)
def mm(inputs: _TensorList, mat2: _TensorList) -> _TensorTuple:
    return torch._foreach_mm(inputs, mat2)
