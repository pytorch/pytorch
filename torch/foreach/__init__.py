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

import functools as _functools
import inspect as _inspect
from typing import overload, ParamSpec, TYPE_CHECKING, TypeVar

import torch
from torch import Tensor
from torch.overrides import (
    handle_torch_function as _handle_torch_function,
    has_torch_function as _has_torch_function,
)
from torch.types import _complex, Number, PySymType


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
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


_OMITTED_ALPHA = object()


def _flatten_foreach_args(arguments: Iterable[object]) -> list[object]:
    relevant_args = []
    for argument in arguments:
        if isinstance(argument, (list, tuple)):
            relevant_args.extend(argument)
        else:
            relevant_args.append(argument)
    return relevant_args


def _make_foreach_api(doc: str) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    r"""Build a public ``torch.foreach`` function from its Python implementation.

    The resulting function constructs the supplied documentation and supports
    ``__torch_function__`` dispatch for tensor-like objects nested inside list
    or tuple arguments. This decorator also handles implementations that take in
    ``alpha=1`` to distinguish an omitted ``alpha`` from an explicit value so
    that the proper python binding can be routed to.
    """

    def decorator(func: Callable[_P, _R]) -> Callable[_P, _R]:
        implementation_signature = _inspect.signature(func)

        # Handle alpha's default of 1
        public_parameters = [
            parameter.replace(
                default=1 if parameter.default is _OMITTED_ALPHA else parameter.default,
                annotation=_inspect.Parameter.empty,
            )
            for parameter in implementation_signature.parameters.values()
        ]
        func.__doc__ = doc

        # Handle __torch__function__
        @_functools.wraps(func)
        def wrapped(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            relevant_args = _flatten_foreach_args((*args, *kwargs.values()))
            if _has_torch_function(relevant_args):
                # Bind the call to validate it and restore signature parameter
                # order before selecting an override.
                try:
                    bound = implementation_signature.bind(*args, **kwargs)
                except TypeError as bind_error:
                    # The signature comes directly from func, so CPython should
                    # reject this call before entering the function body. Calling
                    # it gives users Python's better formatted TypeError.
                    try:
                        func(*args, **kwargs)
                    except TypeError as call_error:
                        raise call_error from None
                    raise RuntimeError(
                        f"{func.__name__} accepted arguments rejected by its "
                        "signature. The function has already run; an in-place "
                        "function may have mutated its inputs"
                    ) from bind_error

                relevant_args = _flatten_foreach_args(bound.arguments.values())
                return _handle_torch_function(wrapped, relevant_args, *args, **kwargs)
            return func(*args, **kwargs)

        # The implementation annotations contain private aliases and Any where
        # overloads define the public parameter types. Keep those details out of
        # runtime introspection, but expose the common public return type.
        return_annotation = _TensorList if func.__name__.endswith("_") else _TensorTuple
        wrapped.__annotations__ = {"return": return_annotation}
        wrapped.__signature__ = implementation_signature.replace(  # type: ignore[attr-defined]
            parameters=public_parameters,
            return_annotation=return_annotation,
        )
        return wrapped

    return decorator


def _result_doc(inplace: bool) -> str:
    if inplace:
        return "the exact input list or tuple"
    return "a tuple containing one result tensor for each input tensor"


def _common_doc(
    reference: str,
    *,
    inplace: bool,
    has_aligned_lists: bool = False,
    original_op_is_method: bool = False,
) -> str:
    reference_role = "meth" if original_op_is_method else "func"
    mutation = (
        "Mutates every tensor in ``inputs`` and returns the exact input "
        "container object."
        if inplace
        else "Does not mutate its arguments and returns a tuple of result tensors."
    )
    length_requirement = (
        "\nCorresponding tensor or scalar lists must have the same length."
        if has_aligned_lists
        else ""
    )
    return rf"""
This is semantically equivalent to applying :{reference_role}:`{reference}` independently
at every list position. {mutation}

Tensor-list arguments must be non-empty.{length_requirement}
An accelerated multi-tensor implementation is used only when supported by the
inputs; otherwise the operation falls back to per-tensor execution.
"""


def _unary_doc(
    reference: str,
    *,
    inplace: bool,
    note: str | None = None,
) -> str:
    note_text = "" if note is None else f"\n{note}\n"
    return rf"""
Applies :func:`{reference}` to each tensor in ``inputs``{" in-place" if inplace else ""}.

{_common_doc(reference, inplace=inplace)}
{note_text}

Args:
    inputs (list or tuple of Tensor): tensors to transform.

Returns:
    {_result_doc(inplace)}.
"""


def _binary_doc(
    reference: str,
    *,
    inplace: bool,
    alpha: str | None = None,
    operand: str = "other",
    shared_tensor: bool = False,
    operand_note: str | None = None,
) -> str:
    alpha_doc = "" if alpha is None else f"\n    alpha (Number, optional): {alpha}"
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
Applies :func:`{reference}` to every tensor in ``inputs``.

{_common_doc(reference, inplace=inplace, has_aligned_lists=True)}
{shared_tensor_note}{operand_note_text}

Args:
    inputs (list or tuple of Tensor): tensors to transform.
    {operand} ({operand_type}): operand shared across positions or supplied per
        position.{alpha_doc}

Returns:
    {_result_doc(inplace)}.
"""


def _pointwise_doc(reference: str, *, inplace: bool) -> str:
    return rf"""
Applies :func:`{reference}` to corresponding tensors from the three input
lists.

{_common_doc(reference, inplace=inplace, has_aligned_lists=True)}

``value`` may be one shared scalar, a scalar list or tuple, or a packed 1-D
CPU tensor containing one scalar per list position.

Args:
    inputs (list or tuple of Tensor): tensors to transform.
    tensor1 (list or tuple of Tensor): first multiplicative or divisive operands.
    tensor2 (list or tuple of Tensor): second multiplicative or divisive operands.
    value (Number, list or tuple of Number, or Tensor, optional): scale values.
        Default: ``1``.

Returns:
    {_result_doc(inplace)}.
"""


# Unary operations


@_make_foreach_api(_unary_doc("torch.abs", inplace=False))
def abs(inputs: _TensorList, /) -> _TensorTuple:
    return torch._foreach_abs(inputs)


@_make_foreach_api(_unary_doc("torch.abs", inplace=True))
def abs_(inputs: _TensorList, /) -> _TensorList:
    torch._foreach_abs_(inputs)
    # why not just return torch._foreach_abs_(inputs)?
    # cuz dynamo bypasses python bindings and we'd want the in-place
    # invariant to still hold with torch.compile
    return inputs


@_make_foreach_api(_unary_doc("torch.acos", inplace=False))
def acos(inputs: _TensorList, /) -> _TensorTuple:
    return torch._foreach_acos(inputs)


@_make_foreach_api(_unary_doc("torch.acos", inplace=True))
def acos_(inputs: _TensorList, /) -> _TensorList:
    torch._foreach_acos_(inputs)
    return inputs


@_make_foreach_api(_unary_doc("torch.asin", inplace=False))
def asin(inputs: _TensorList, /) -> _TensorTuple:
    return torch._foreach_asin(inputs)


@_make_foreach_api(_unary_doc("torch.asin", inplace=True))
def asin_(inputs: _TensorList, /) -> _TensorList:
    torch._foreach_asin_(inputs)
    return inputs


@_make_foreach_api(_unary_doc("torch.atan", inplace=False))
def atan(inputs: _TensorList, /) -> _TensorTuple:
    return torch._foreach_atan(inputs)


@_make_foreach_api(_unary_doc("torch.atan", inplace=True))
def atan_(inputs: _TensorList, /) -> _TensorList:
    torch._foreach_atan_(inputs)
    return inputs


@_make_foreach_api(_unary_doc("torch.ceil", inplace=False))
def ceil(inputs: _TensorList, /) -> _TensorTuple:
    return torch._foreach_ceil(inputs)


@_make_foreach_api(_unary_doc("torch.ceil", inplace=True))
def ceil_(inputs: _TensorList, /) -> _TensorList:
    torch._foreach_ceil_(inputs)
    return inputs


@_make_foreach_api(_unary_doc("torch.cos", inplace=False))
def cos(inputs: _TensorList, /) -> _TensorTuple:
    return torch._foreach_cos(inputs)


@_make_foreach_api(_unary_doc("torch.cos", inplace=True))
def cos_(inputs: _TensorList, /) -> _TensorList:
    torch._foreach_cos_(inputs)
    return inputs


@_make_foreach_api(_unary_doc("torch.cosh", inplace=False))
def cosh(inputs: _TensorList, /) -> _TensorTuple:
    return torch._foreach_cosh(inputs)


@_make_foreach_api(_unary_doc("torch.cosh", inplace=True))
def cosh_(inputs: _TensorList, /) -> _TensorList:
    torch._foreach_cosh_(inputs)
    return inputs


@_make_foreach_api(_unary_doc("torch.erf", inplace=False))
def erf(inputs: _TensorList, /) -> _TensorTuple:
    return torch._foreach_erf(inputs)


@_make_foreach_api(_unary_doc("torch.erf", inplace=True))
def erf_(inputs: _TensorList, /) -> _TensorList:
    torch._foreach_erf_(inputs)
    return inputs


@_make_foreach_api(_unary_doc("torch.erfc", inplace=False))
def erfc(inputs: _TensorList, /) -> _TensorTuple:
    return torch._foreach_erfc(inputs)


@_make_foreach_api(_unary_doc("torch.erfc", inplace=True))
def erfc_(inputs: _TensorList, /) -> _TensorList:
    torch._foreach_erfc_(inputs)
    return inputs


@_make_foreach_api(_unary_doc("torch.exp", inplace=False))
def exp(inputs: _TensorList, /) -> _TensorTuple:
    return torch._foreach_exp(inputs)


@_make_foreach_api(_unary_doc("torch.exp", inplace=True))
def exp_(inputs: _TensorList, /) -> _TensorList:
    torch._foreach_exp_(inputs)
    return inputs


@_make_foreach_api(_unary_doc("torch.expm1", inplace=False))
def expm1(inputs: _TensorList, /) -> _TensorTuple:
    return torch._foreach_expm1(inputs)


@_make_foreach_api(_unary_doc("torch.expm1", inplace=True))
def expm1_(inputs: _TensorList, /) -> _TensorList:
    torch._foreach_expm1_(inputs)
    return inputs


@_make_foreach_api(_unary_doc("torch.floor", inplace=False))
def floor(inputs: _TensorList, /) -> _TensorTuple:
    return torch._foreach_floor(inputs)


@_make_foreach_api(_unary_doc("torch.floor", inplace=True))
def floor_(inputs: _TensorList, /) -> _TensorList:
    torch._foreach_floor_(inputs)
    return inputs


@_make_foreach_api(_unary_doc("torch.frac", inplace=False))
def frac(inputs: _TensorList, /) -> _TensorTuple:
    return torch._foreach_frac(inputs)


@_make_foreach_api(_unary_doc("torch.frac", inplace=True))
def frac_(inputs: _TensorList, /) -> _TensorList:
    torch._foreach_frac_(inputs)
    return inputs


@_make_foreach_api(_unary_doc("torch.lgamma", inplace=False))
def lgamma(inputs: _TensorList, /) -> _TensorTuple:
    return torch._foreach_lgamma(inputs)


@_make_foreach_api(_unary_doc("torch.lgamma", inplace=True))
def lgamma_(inputs: _TensorList, /) -> _TensorList:
    torch._foreach_lgamma_(inputs)
    return inputs


@_make_foreach_api(_unary_doc("torch.log", inplace=False))
def log(inputs: _TensorList, /) -> _TensorTuple:
    return torch._foreach_log(inputs)


@_make_foreach_api(_unary_doc("torch.log", inplace=True))
def log_(inputs: _TensorList, /) -> _TensorList:
    torch._foreach_log_(inputs)
    return inputs


@_make_foreach_api(_unary_doc("torch.log10", inplace=False))
def log10(inputs: _TensorList, /) -> _TensorTuple:
    return torch._foreach_log10(inputs)


@_make_foreach_api(_unary_doc("torch.log10", inplace=True))
def log10_(inputs: _TensorList, /) -> _TensorList:
    torch._foreach_log10_(inputs)
    return inputs


@_make_foreach_api(_unary_doc("torch.log1p", inplace=False))
def log1p(inputs: _TensorList, /) -> _TensorTuple:
    return torch._foreach_log1p(inputs)


@_make_foreach_api(_unary_doc("torch.log1p", inplace=True))
def log1p_(inputs: _TensorList, /) -> _TensorList:
    torch._foreach_log1p_(inputs)
    return inputs


@_make_foreach_api(_unary_doc("torch.log2", inplace=False))
def log2(inputs: _TensorList, /) -> _TensorTuple:
    return torch._foreach_log2(inputs)


@_make_foreach_api(_unary_doc("torch.log2", inplace=True))
def log2_(inputs: _TensorList, /) -> _TensorList:
    torch._foreach_log2_(inputs)
    return inputs


@_make_foreach_api(_unary_doc("torch.neg", inplace=False))
def neg(inputs: _TensorList, /) -> _TensorTuple:
    return torch._foreach_neg(inputs)


@_make_foreach_api(_unary_doc("torch.neg", inplace=True))
def neg_(inputs: _TensorList, /) -> _TensorList:
    torch._foreach_neg_(inputs)
    return inputs


@_make_foreach_api(_unary_doc("torch.reciprocal", inplace=False))
def reciprocal(inputs: _TensorList, /) -> _TensorTuple:
    return torch._foreach_reciprocal(inputs)


@_make_foreach_api(_unary_doc("torch.reciprocal", inplace=True))
def reciprocal_(inputs: _TensorList, /) -> _TensorList:
    torch._foreach_reciprocal_(inputs)
    return inputs


@_make_foreach_api(
    _unary_doc(
        "torch.round",
        inplace=False,
        note="The ``decimals`` argument is not supported.",
    )
)
def round(inputs: _TensorList, /) -> _TensorTuple:
    return torch._foreach_round(inputs)


@_make_foreach_api(
    _unary_doc(
        "torch.round",
        inplace=True,
        note="The ``decimals`` argument is not supported.",
    )
)
def round_(inputs: _TensorList, /) -> _TensorList:
    torch._foreach_round_(inputs)
    return inputs


@_make_foreach_api(_unary_doc("torch.rsqrt", inplace=False))
def rsqrt(inputs: _TensorList, /) -> _TensorTuple:
    return torch._foreach_rsqrt(inputs)


@_make_foreach_api(_unary_doc("torch.rsqrt", inplace=True))
def rsqrt_(inputs: _TensorList, /) -> _TensorList:
    torch._foreach_rsqrt_(inputs)
    return inputs


@_make_foreach_api(_unary_doc("torch.sigmoid", inplace=False))
def sigmoid(inputs: _TensorList, /) -> _TensorTuple:
    return torch._foreach_sigmoid(inputs)


@_make_foreach_api(_unary_doc("torch.sigmoid", inplace=True))
def sigmoid_(inputs: _TensorList, /) -> _TensorList:
    torch._foreach_sigmoid_(inputs)
    return inputs


@_make_foreach_api(_unary_doc("torch.sign", inplace=False))
def sign(inputs: _TensorList, /) -> _TensorTuple:
    return torch._foreach_sign(inputs)


@_make_foreach_api(_unary_doc("torch.sign", inplace=True))
def sign_(inputs: _TensorList, /) -> _TensorList:
    torch._foreach_sign_(inputs)
    return inputs


@_make_foreach_api(_unary_doc("torch.sin", inplace=False))
def sin(inputs: _TensorList, /) -> _TensorTuple:
    return torch._foreach_sin(inputs)


@_make_foreach_api(_unary_doc("torch.sin", inplace=True))
def sin_(inputs: _TensorList, /) -> _TensorList:
    torch._foreach_sin_(inputs)
    return inputs


@_make_foreach_api(_unary_doc("torch.sinh", inplace=False))
def sinh(inputs: _TensorList, /) -> _TensorTuple:
    return torch._foreach_sinh(inputs)


@_make_foreach_api(_unary_doc("torch.sinh", inplace=True))
def sinh_(inputs: _TensorList, /) -> _TensorList:
    torch._foreach_sinh_(inputs)
    return inputs


@_make_foreach_api(_unary_doc("torch.sqrt", inplace=False))
def sqrt(inputs: _TensorList, /) -> _TensorTuple:
    return torch._foreach_sqrt(inputs)


@_make_foreach_api(_unary_doc("torch.sqrt", inplace=True))
def sqrt_(inputs: _TensorList, /) -> _TensorList:
    torch._foreach_sqrt_(inputs)
    return inputs


@_make_foreach_api(_unary_doc("torch.tan", inplace=False))
def tan(inputs: _TensorList, /) -> _TensorTuple:
    return torch._foreach_tan(inputs)


@_make_foreach_api(_unary_doc("torch.tan", inplace=True))
def tan_(inputs: _TensorList, /) -> _TensorList:
    torch._foreach_tan_(inputs)
    return inputs


@_make_foreach_api(_unary_doc("torch.tanh", inplace=False))
def tanh(inputs: _TensorList, /) -> _TensorTuple:
    return torch._foreach_tanh(inputs)


@_make_foreach_api(_unary_doc("torch.tanh", inplace=True))
def tanh_(inputs: _TensorList, /) -> _TensorList:
    torch._foreach_tanh_(inputs)
    return inputs


@_make_foreach_api(_unary_doc("torch.trunc", inplace=False))
def trunc(inputs: _TensorList, /) -> _TensorTuple:
    return torch._foreach_trunc(inputs)


@_make_foreach_api(_unary_doc("torch.trunc", inplace=True))
def trunc_(inputs: _TensorList, /) -> _TensorList:
    torch._foreach_trunc_(inputs)
    return inputs


# Binary operations


@overload
def add(inputs: _TensorList, other: _Scalar, /) -> _TensorTuple: ...


@overload
def add(inputs: _TensorList, other: _ScalarList[_ScalarT], /) -> _TensorTuple: ...


@overload
def add(
    inputs: _TensorList,
    other: Tensor,
    /,
    *,
    alpha: _Scalar,
) -> _TensorTuple: ...


@overload
def add(
    inputs: _TensorList,
    other: _TensorList,
    /,
    *,
    alpha: _Scalar = 1,
) -> _TensorTuple: ...


@_make_foreach_api(
    _binary_doc(
        "torch.add",
        inplace=False,
        alpha=(
            "supported only when ``other`` is a tensor list or a shared 0-D "
            "scalar tensor. Default: ``1``."
        ),
        shared_tensor=True,
        operand_note=(
            "For a shared 0-D tensor, pass ``alpha`` explicitly, including when "
            "its value is ``1``, to select the Tensor overload. Omitting "
            "``alpha`` may convert the tensor to a host scalar."
        ),
    )
)
def add(
    inputs: _TensorList,
    other: Any,
    /,
    *,
    alpha: Any = _OMITTED_ALPHA,
) -> _TensorTuple:
    if alpha is _OMITTED_ALPHA:
        return torch._foreach_add(inputs, other)
    return torch._foreach_add(inputs, other, alpha=alpha)


@overload
def add_(inputs: _TensorList, other: _Scalar, /) -> _TensorList: ...


@overload
def add_(inputs: _TensorList, other: _ScalarList[_ScalarT], /) -> _TensorList: ...


@overload
def add_(
    inputs: _TensorList,
    other: Tensor,
    /,
    *,
    alpha: _Scalar,
) -> _TensorList: ...


@overload
def add_(
    inputs: _TensorList,
    other: _TensorList,
    /,
    *,
    alpha: _Scalar = 1,
) -> _TensorList: ...


@_make_foreach_api(
    _binary_doc(
        "torch.add",
        inplace=True,
        alpha=(
            "supported only when ``other`` is a tensor list or a shared 0-D "
            "scalar tensor. Default: ``1``."
        ),
        shared_tensor=True,
        operand_note=(
            "For a shared 0-D tensor, pass ``alpha`` explicitly, including when "
            "its value is ``1``, to select the Tensor overload. Omitting "
            "``alpha`` may convert the tensor to a host scalar."
        ),
    )
)
def add_(
    inputs: _TensorList,
    other: Any,
    /,
    *,
    alpha: Any = _OMITTED_ALPHA,
) -> _TensorList:
    if alpha is _OMITTED_ALPHA:
        torch._foreach_add_(inputs, other)
    else:
        torch._foreach_add_(inputs, other, alpha=alpha)
    return inputs


@overload
def sub(inputs: _TensorList, other: _Scalar, /) -> _TensorTuple: ...


@overload
def sub(inputs: _TensorList, other: _ScalarList[_ScalarT], /) -> _TensorTuple: ...


@overload
def sub(
    inputs: _TensorList,
    other: _TensorList,
    /,
    *,
    alpha: _Scalar = 1,
) -> _TensorTuple: ...


@_make_foreach_api(
    _binary_doc(
        "torch.sub",
        inplace=False,
        alpha="supported only when ``other`` is a tensor list. Default: ``1``.",
    )
)
def sub(
    inputs: _TensorList,
    other: Any,
    /,
    *,
    alpha: Any = _OMITTED_ALPHA,
) -> _TensorTuple:
    if alpha is _OMITTED_ALPHA:
        return torch._foreach_sub(inputs, other)
    return torch._foreach_sub(inputs, other, alpha=alpha)


@overload
def sub_(inputs: _TensorList, other: _Scalar, /) -> _TensorList: ...


@overload
def sub_(inputs: _TensorList, other: _ScalarList[_ScalarT], /) -> _TensorList: ...


@overload
def sub_(
    inputs: _TensorList,
    other: _TensorList,
    /,
    *,
    alpha: _Scalar = 1,
) -> _TensorList: ...


@_make_foreach_api(
    _binary_doc(
        "torch.sub",
        inplace=True,
        alpha="supported only when ``other`` is a tensor list. Default: ``1``.",
    )
)
def sub_(
    inputs: _TensorList,
    other: Any,
    /,
    *,
    alpha: Any = _OMITTED_ALPHA,
) -> _TensorList:
    if alpha is _OMITTED_ALPHA:
        torch._foreach_sub_(inputs, other)
    else:
        torch._foreach_sub_(inputs, other, alpha=alpha)
    return inputs


@overload
def mul(inputs: _TensorList, other: _ScalarList[_ScalarT], /) -> _TensorTuple: ...


@overload
def mul(inputs: _TensorList, other: Tensor, /) -> _TensorTuple: ...


@overload
def mul(inputs: _TensorList, other: _TensorList, /) -> _TensorTuple: ...


@overload
def mul(inputs: _TensorList, other: _Scalar, /) -> _TensorTuple: ...


@_make_foreach_api(
    _binary_doc(
        "torch.mul",
        inplace=False,
        shared_tensor=True,
    )
)
def mul(inputs: _TensorList, other: Any, /) -> _TensorTuple:
    return torch._foreach_mul(inputs, other)


@overload
def mul_(inputs: _TensorList, other: _ScalarList[_ScalarT], /) -> _TensorList: ...


@overload
def mul_(inputs: _TensorList, other: Tensor, /) -> _TensorList: ...


@overload
def mul_(inputs: _TensorList, other: _TensorList, /) -> _TensorList: ...


@overload
def mul_(inputs: _TensorList, other: _Scalar, /) -> _TensorList: ...


@_make_foreach_api(
    _binary_doc(
        "torch.mul",
        inplace=True,
        shared_tensor=True,
    )
)
def mul_(inputs: _TensorList, other: Any, /) -> _TensorList:
    torch._foreach_mul_(inputs, other)
    return inputs


@overload
def div(inputs: _TensorList, other: _ScalarList[_ScalarT], /) -> _TensorTuple: ...


@overload
def div(inputs: _TensorList, other: Tensor, /) -> _TensorTuple: ...


@overload
def div(inputs: _TensorList, other: _TensorList, /) -> _TensorTuple: ...


@overload
def div(inputs: _TensorList, other: _Scalar, /) -> _TensorTuple: ...


@_make_foreach_api(
    _binary_doc(
        "torch.div",
        inplace=False,
        shared_tensor=True,
        operand_note="The ``rounding_mode`` argument is not supported.",
    )
)
def div(inputs: _TensorList, other: Any, /) -> _TensorTuple:
    return torch._foreach_div(inputs, other)


@overload
def div_(inputs: _TensorList, other: _ScalarList[_ScalarT], /) -> _TensorList: ...


@overload
def div_(inputs: _TensorList, other: Tensor, /) -> _TensorList: ...


@overload
def div_(inputs: _TensorList, other: _TensorList, /) -> _TensorList: ...


@overload
def div_(inputs: _TensorList, other: _Scalar, /) -> _TensorList: ...


@_make_foreach_api(
    _binary_doc(
        "torch.div",
        inplace=True,
        shared_tensor=True,
        operand_note="The ``rounding_mode`` argument is not supported.",
    )
)
def div_(inputs: _TensorList, other: Any, /) -> _TensorList:
    torch._foreach_div_(inputs, other)
    return inputs


@overload
def clamp_min(inputs: _TensorList, min: _Scalar, /) -> _TensorTuple: ...


@overload
def clamp_min(inputs: _TensorList, min: _ScalarList[_ScalarT], /) -> _TensorTuple: ...


@overload
def clamp_min(inputs: _TensorList, min: _TensorList, /) -> _TensorTuple: ...


@_make_foreach_api(
    _binary_doc(
        "torch.clamp",
        inplace=False,
        operand="min",
        operand_note="Only the ``min`` bound is specified.",
    )
)
def clamp_min(inputs: _TensorList, min: Any, /) -> _TensorTuple:
    return torch._foreach_clamp_min(inputs, min)


@overload
def clamp_min_(inputs: _TensorList, min: _Scalar, /) -> _TensorList: ...


@overload
def clamp_min_(inputs: _TensorList, min: _ScalarList[_ScalarT], /) -> _TensorList: ...


@overload
def clamp_min_(inputs: _TensorList, min: _TensorList, /) -> _TensorList: ...


@_make_foreach_api(
    _binary_doc(
        "torch.clamp",
        inplace=True,
        operand="min",
        operand_note="Only the ``min`` bound is specified.",
    )
)
def clamp_min_(inputs: _TensorList, min: Any, /) -> _TensorList:
    torch._foreach_clamp_min_(inputs, min)
    return inputs


@overload
def clamp_max(inputs: _TensorList, max: _Scalar, /) -> _TensorTuple: ...


@overload
def clamp_max(inputs: _TensorList, max: _ScalarList[_ScalarT], /) -> _TensorTuple: ...


@overload
def clamp_max(inputs: _TensorList, max: _TensorList, /) -> _TensorTuple: ...


@_make_foreach_api(
    _binary_doc(
        "torch.clamp",
        inplace=False,
        operand="max",
        operand_note="Only the ``max`` bound is specified.",
    )
)
def clamp_max(inputs: _TensorList, max: Any, /) -> _TensorTuple:
    return torch._foreach_clamp_max(inputs, max)


@overload
def clamp_max_(inputs: _TensorList, max: _Scalar, /) -> _TensorList: ...


@overload
def clamp_max_(inputs: _TensorList, max: _ScalarList[_ScalarT], /) -> _TensorList: ...


@overload
def clamp_max_(inputs: _TensorList, max: _TensorList, /) -> _TensorList: ...


@_make_foreach_api(
    _binary_doc(
        "torch.clamp",
        inplace=True,
        operand="max",
        operand_note="Only the ``max`` bound is specified.",
    )
)
def clamp_max_(inputs: _TensorList, max: Any, /) -> _TensorList:
    torch._foreach_clamp_max_(inputs, max)
    return inputs


@overload
def minimum(inputs: _TensorList, other: _Scalar, /) -> _TensorTuple: ...


@overload
def minimum(inputs: _TensorList, other: _ScalarList[_ScalarT], /) -> _TensorTuple: ...


@overload
def minimum(inputs: _TensorList, other: _TensorList, /) -> _TensorTuple: ...


@_make_foreach_api(
    _binary_doc(
        "torch.minimum",
        inplace=False,
        operand_note=(
            "Scalar and ``ScalarList`` forms are semantically equivalent to "
            ":func:`torch.clamp` with only ``max`` specified."
        ),
    )
)
def minimum(inputs: _TensorList, other: Any, /) -> _TensorTuple:
    return torch._foreach_minimum(inputs, other)


@overload
def minimum_(inputs: _TensorList, other: _Scalar, /) -> _TensorList: ...


@overload
def minimum_(inputs: _TensorList, other: _ScalarList[_ScalarT], /) -> _TensorList: ...


@overload
def minimum_(inputs: _TensorList, other: _TensorList, /) -> _TensorList: ...


@_make_foreach_api(
    _binary_doc(
        "torch.minimum",
        inplace=True,
        operand_note=(
            "Scalar and ``ScalarList`` forms are semantically equivalent to "
            ":func:`torch.clamp` with only ``max`` specified."
        ),
    )
)
def minimum_(inputs: _TensorList, other: Any, /) -> _TensorList:
    torch._foreach_minimum_(inputs, other)
    return inputs


@overload
def maximum(inputs: _TensorList, other: _Scalar, /) -> _TensorTuple: ...


@overload
def maximum(inputs: _TensorList, other: _ScalarList[_ScalarT], /) -> _TensorTuple: ...


@overload
def maximum(inputs: _TensorList, other: _TensorList, /) -> _TensorTuple: ...


@_make_foreach_api(
    _binary_doc(
        "torch.maximum",
        inplace=False,
        operand_note=(
            "Scalar and ``ScalarList`` forms are semantically equivalent to "
            ":func:`torch.clamp` with only ``min`` specified."
        ),
    )
)
def maximum(inputs: _TensorList, other: Any, /) -> _TensorTuple:
    return torch._foreach_maximum(inputs, other)


@overload
def maximum_(inputs: _TensorList, other: _Scalar, /) -> _TensorList: ...


@overload
def maximum_(inputs: _TensorList, other: _ScalarList[_ScalarT], /) -> _TensorList: ...


@overload
def maximum_(inputs: _TensorList, other: _TensorList, /) -> _TensorList: ...


@_make_foreach_api(
    _binary_doc(
        "torch.maximum",
        inplace=True,
        operand_note=(
            "Scalar and ``ScalarList`` forms are semantically equivalent to "
            ":func:`torch.clamp` with only ``min`` specified."
        ),
    )
)
def maximum_(inputs: _TensorList, other: Any, /) -> _TensorList:
    torch._foreach_maximum_(inputs, other)
    return inputs


# Pointwise operations


@overload
def addcmul(
    inputs: _TensorList,
    tensor1: _TensorList,
    tensor2: _TensorList,
    /,
    *,
    value: _ScalarList[_ScalarT],
) -> _TensorTuple: ...


@overload
def addcmul(
    inputs: _TensorList,
    tensor1: _TensorList,
    tensor2: _TensorList,
    /,
    *,
    value: Tensor,
) -> _TensorTuple: ...


@overload
def addcmul(
    inputs: _TensorList,
    tensor1: _TensorList,
    tensor2: _TensorList,
    /,
    *,
    value: _Scalar = 1,
) -> _TensorTuple: ...


@_make_foreach_api(_pointwise_doc("torch.addcmul", inplace=False))
def addcmul(
    inputs: _TensorList,
    tensor1: _TensorList,
    tensor2: _TensorList,
    /,
    *,
    value: Any = 1,
) -> _TensorTuple:
    return torch._foreach_addcmul(inputs, tensor1, tensor2, value)


@overload
def addcmul_(
    inputs: _TensorList,
    tensor1: _TensorList,
    tensor2: _TensorList,
    /,
    *,
    value: _ScalarList[_ScalarT],
) -> _TensorList: ...


@overload
def addcmul_(
    inputs: _TensorList,
    tensor1: _TensorList,
    tensor2: _TensorList,
    /,
    *,
    value: Tensor,
) -> _TensorList: ...


@overload
def addcmul_(
    inputs: _TensorList,
    tensor1: _TensorList,
    tensor2: _TensorList,
    /,
    *,
    value: _Scalar = 1,
) -> _TensorList: ...


@_make_foreach_api(_pointwise_doc("torch.addcmul", inplace=True))
def addcmul_(
    inputs: _TensorList,
    tensor1: _TensorList,
    tensor2: _TensorList,
    /,
    *,
    value: Any = 1,
) -> _TensorList:
    torch._foreach_addcmul_(inputs, tensor1, tensor2, value)
    return inputs


@overload
def addcdiv(
    inputs: _TensorList,
    tensor1: _TensorList,
    tensor2: _TensorList,
    /,
    *,
    value: _ScalarList[_ScalarT],
) -> _TensorTuple: ...


@overload
def addcdiv(
    inputs: _TensorList,
    tensor1: _TensorList,
    tensor2: _TensorList,
    /,
    *,
    value: Tensor,
) -> _TensorTuple: ...


@overload
def addcdiv(
    inputs: _TensorList,
    tensor1: _TensorList,
    tensor2: _TensorList,
    /,
    *,
    value: _Scalar = 1,
) -> _TensorTuple: ...


@_make_foreach_api(_pointwise_doc("torch.addcdiv", inplace=False))
def addcdiv(
    inputs: _TensorList,
    tensor1: _TensorList,
    tensor2: _TensorList,
    /,
    *,
    value: Any = 1,
) -> _TensorTuple:
    return torch._foreach_addcdiv(inputs, tensor1, tensor2, value)


@overload
def addcdiv_(
    inputs: _TensorList,
    tensor1: _TensorList,
    tensor2: _TensorList,
    /,
    *,
    value: _ScalarList[_ScalarT],
) -> _TensorList: ...


@overload
def addcdiv_(
    inputs: _TensorList,
    tensor1: _TensorList,
    tensor2: _TensorList,
    /,
    *,
    value: Tensor,
) -> _TensorList: ...


@overload
def addcdiv_(
    inputs: _TensorList,
    tensor1: _TensorList,
    tensor2: _TensorList,
    /,
    *,
    value: _Scalar = 1,
) -> _TensorList: ...


@_make_foreach_api(_pointwise_doc("torch.addcdiv", inplace=True))
def addcdiv_(
    inputs: _TensorList,
    tensor1: _TensorList,
    tensor2: _TensorList,
    /,
    *,
    value: Any = 1,
) -> _TensorList:
    torch._foreach_addcdiv_(inputs, tensor1, tensor2, value)
    return inputs


@overload
def lerp(inputs: _TensorList, end: _TensorList, weight: _Scalar, /) -> _TensorTuple: ...


@overload
def lerp(
    inputs: _TensorList,
    end: _TensorList,
    weight: _ScalarList[_ScalarT],
    /,
) -> _TensorTuple: ...


@overload
def lerp(
    inputs: _TensorList,
    end: _TensorList,
    weight: _TensorList,
    /,
) -> _TensorTuple: ...


@_make_foreach_api(
    rf"""
Applies :func:`torch.lerp` to corresponding tensors in ``inputs`` and
``end``.

{_common_doc("torch.lerp", inplace=False, has_aligned_lists=True)}

``weight`` may be one shared scalar, a scalar list or tuple, or a tensor list.

Args:
    inputs (list or tuple of Tensor): starting tensors.
    end (list or tuple of Tensor): ending tensors.
    weight (Number, list or tuple of Number, or list or tuple of Tensor):
        interpolation weights.

Returns:
    a tuple containing one result tensor per list position.
"""
)
def lerp(inputs: _TensorList, end: _TensorList, weight: Any, /) -> _TensorTuple:
    return torch._foreach_lerp(inputs, end, weight)


@overload
def lerp_(inputs: _TensorList, end: _TensorList, weight: _Scalar, /) -> _TensorList: ...


@overload
def lerp_(
    inputs: _TensorList,
    end: _TensorList,
    weight: _ScalarList[_ScalarT],
    /,
) -> _TensorList: ...


@overload
def lerp_(
    inputs: _TensorList,
    end: _TensorList,
    weight: _TensorList,
    /,
) -> _TensorList: ...


@_make_foreach_api(
    rf"""
In-place version of :func:`torch.foreach.lerp`.

{_common_doc("torch.lerp", inplace=True, has_aligned_lists=True)}

``weight`` may be one shared scalar, a scalar list or tuple, or a tensor list.

Args:
    inputs (list or tuple of Tensor): starting tensors to mutate.
    end (list or tuple of Tensor): ending tensors.
    weight (Number, list or tuple of Number, or list or tuple of Tensor):
        interpolation weights.

Returns:
    the exact ``inputs`` list or tuple.
"""
)
def lerp_(inputs: _TensorList, end: _TensorList, weight: Any, /) -> _TensorList:
    torch._foreach_lerp_(inputs, end, weight)
    return inputs


@overload
def pow(input: _Scalar, exponent: _TensorList, /) -> _TensorTuple: ...


@overload
def pow(input: _TensorList, exponent: _Scalar, /) -> _TensorTuple: ...


@overload
def pow(input: _TensorList, exponent: _ScalarList[_ScalarT], /) -> _TensorTuple: ...


@overload
def pow(input: _TensorList, exponent: _TensorList, /) -> _TensorTuple: ...


@_make_foreach_api(
    rf"""
Applies :func:`torch.pow` at each list position.

{_common_doc("torch.pow", inplace=False, has_aligned_lists=True)}

Args:
    input (Number, list of Tensor, or tuple of Tensor): bases.
    exponent (Number, list or tuple of Number, or list or tuple of Tensor):
        exponents.

Returns:
    a tuple containing one result tensor per list position.
"""
)
def pow(input: Any, exponent: Any, /) -> _TensorTuple:
    return torch._foreach_pow(input, exponent)


@overload
def pow_(inputs: _TensorList, exponent: _Scalar, /) -> _TensorList: ...


@overload
def pow_(inputs: _TensorList, exponent: _ScalarList[_ScalarT], /) -> _TensorList: ...


@overload
def pow_(inputs: _TensorList, exponent: _TensorList, /) -> _TensorList: ...


@_make_foreach_api(
    rf"""
In-place version of :func:`torch.foreach.pow`.

{_common_doc("torch.pow", inplace=True, has_aligned_lists=True)}

The scalar-left form has no in-place variant.

Args:
    inputs (list or tuple of Tensor): bases to mutate.
    exponent (Number, list or tuple of Number, or list or tuple of Tensor):
        exponents.

Returns:
    the exact ``inputs`` list or tuple.
"""
)
def pow_(inputs: _TensorList, exponent: Any, /) -> _TensorList:
    torch._foreach_pow_(inputs, exponent)
    return inputs


# Reductions and special operations


@_make_foreach_api(
    rf"""
Clones every tensor in ``inputs``.

{_common_doc("torch.clone", inplace=False)}

Args:
    inputs (list or tuple of Tensor): tensors to clone.
    memory_format (:class:`torch.memory_format`, optional): desired memory
        format. If ``None``, the input memory format is preserved. Default: ``None``.

Returns:
    a tuple containing the cloned tensors.
"""
)
def clone(
    inputs: _TensorList,
    /,
    *,
    memory_format: torch.memory_format | None = None,
) -> _TensorTuple:
    return torch._foreach_clone(inputs, memory_format=memory_format)


@_make_foreach_api(
    rf"""
Returns the maximum value of each tensor in ``inputs``.

{_common_doc("torch.max", inplace=False)}

This operation reduces every input tensor over all dimensions. It does not
accept a dimension or return indices.

Args:
    inputs (list or tuple of Tensor): tensors to reduce.

Returns:
    a tuple of scalar tensors.
"""
)
def max(inputs: _TensorList, /) -> _TensorTuple:
    return torch._foreach_max(inputs)


@_make_foreach_api(
    rf"""
Returns the vector norm of each tensor in ``inputs``.

{_common_doc("torch.linalg.vector_norm", inplace=False)}

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
    /,
    *,
    ord: _Scalar = 2,
    dtype: torch.dtype | None = None,
) -> _TensorTuple:
    return torch._foreach_norm(inputs, ord, dtype=dtype)


@_make_foreach_api(
    rf"""
Copies each tensor in ``src`` into the corresponding tensor in
``inputs``, following :meth:`torch.Tensor.copy_`.

{_common_doc("torch.Tensor.copy_", inplace=True, has_aligned_lists=True, original_op_is_method=True)}

There is no functional ``torch.foreach.copy`` operation.

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
    /,
    *,
    non_blocking: bool = False,
) -> _TensorList:
    torch._foreach_copy_(inputs, src, non_blocking)
    return inputs


@_make_foreach_api(
    rf"""
Fills every tensor in ``inputs`` with zero.

{_common_doc("torch.Tensor.zero_", inplace=True, original_op_is_method=True)}

There is no functional ``torch.foreach.zero`` operation.

Args:
    inputs (list or tuple of Tensor): tensors to zero.

Returns:
    the exact ``inputs`` list or tuple.
"""
)
def zero_(inputs: _TensorList, /) -> _TensorList:
    torch._foreach_zero_(inputs)
    return inputs


@_make_foreach_api(
    r"""
Multiplies corresponding matrices from ``inputs`` and ``mat2`` using
:func:`torch.mm`. This is semantically equivalent to applying
:func:`torch.mm` independently at every list position. It does not mutate its
arguments and returns a tuple of result tensors.

On supported CUDA inputs, an accelerated grouped matrix multiplication
implementation may be used. Other inputs fall back to per-position execution.

Both tensor-list arguments must be non-empty and have the same length.
There is no in-place ``torch.foreach.mm_`` operation.

Args:
    inputs (list or tuple of Tensor): first matrices.
    mat2 (list or tuple of Tensor): second matrices.

Returns:
    a tuple containing one matrix product per list position.
"""
)
def mm(inputs: _TensorList, mat2: _TensorList, /) -> _TensorTuple:
    return torch._foreach_mm(inputs, mat2)
