import typing
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Generic, overload, TypeAlias, TypeVar
from typing_extensions import ParamSpec, TypeIs, TypeVarTuple, Unpack

import torch
import torch.fx.node
import torch.utils._pytree as pytree
from torch._library.fake_class_registry import FakeScriptObject
from torch._library.opaque_object import is_opaque_type
from torch._ops import HigherOrderOperator


_R = TypeVar("_R")
_P = ParamSpec("_P")
_Ts = TypeVarTuple("_Ts")


def is_graphable(val: object) -> TypeIs[torch.fx.node.BaseArgumentTypes]:
    """Definition: a graphable type is a type that is an acceptable input/output type to a FX node."""
    return isinstance(
        val, (*torch.fx.node.base_types, FakeScriptObject)
    ) or is_opaque_type(type(val))


def is_graphable_type(typ: type[object]) -> bool:
    """Return whether the given type is graphable."""
    return (
        issubclass(typ, torch.fx.node.base_types)
        or is_opaque_type(typ)
        or issubclass(typ, FakeScriptObject)
    )


def to_graphable(stuff: pytree.PyTree) -> tuple[list[object], pytree.TreeSpec]:
    """Flattens stuff into a flat list of graphable types."""
    # We can consider preserving things like List[int] to improve
    # perf and readability (right now that is all flattened out)
    flat_args, spec = pytree.tree_flatten(stuff)
    for arg in flat_args:
        if not is_graphable(arg):
            raise RuntimeError(
                f"Expected all pytree.tree_leaves of (args, kwargs) to be graphable types, but found "
                f"non-fx-graphable type {type(arg)}. If this type is meant to be constant, mark it as "
                f"via pytree.register_constant; otherwise, register it as a pytree."
            )
    return flat_args, spec


def from_graphable(
    flat_args: tuple[Unpack[_Ts]], spec: pytree.TreeSpec
) -> pytree.PyTree:
    """The inverse of to_graphable."""
    stuff = pytree.tree_unflatten(flat_args, spec)
    return stuff


def func_to_graphable(
    func: Callable[..., object],
) -> tuple[list[object], pytree.TreeSpec]:
    """
    Pack and flatten a function type into graphable types.
    This is useful for legalizing the function argument of `flat_apply`.
    """
    return pytree.tree_flatten(_ConstantFunction(func))


@dataclass(frozen=True, slots=True)
class _ConstantFunction(Generic[_P, _R]):
    func: Callable[_P, _R]

    def __call__(self, *args: _P.args, **kwargs: _P.kwargs) -> _R:
        return self.func(*args, **kwargs)


pytree.register_constant(_ConstantFunction)


_OpTypes = (
    torch._ops.OpOverload | torch._ops.OpOverloadPacket | torch._ops.HigherOrderOperator
)
_op_types = typing.get_args(_OpTypes)


_Base: TypeAlias = torch.fx.node.BaseArgumentTypes
# pyrefly bug: pyrefly is complaining: Expected a type form, got instance of `Literal['_FXOutput']
# pyrefly: ignore[not-a-type]
_FXOutput = _Base | Sequence["_FXOutput"]


class FlatApply(HigherOrderOperator):
    def __init__(self) -> None:
        super().__init__("flat_apply")

    def __call__(
        self,
        func: _OpTypes | pytree.TreeSpec,
        in_spec: pytree.TreeSpec,
        *flat_args: Unpack[_Ts],
        # If True then the output is checked to be valid. If False then it is up
        # to the caller to ensure the output is appropriate.
        checked_output: bool = True,
        **_unused: object,
    ) -> object:
        """
        Functions that take in non-graphable types cannot directly be put into FX graph.

        Given func(*args, **kwargs), if all of the non-graphable types are pytrees,
        then we're able to store a call to flat_apply(func, in_spec, *flat_args) in the FX graph.

        The semantics of flat_apply(func, in_spec, *flat_args) are roughly equivalent to:

        >>> def flat_apply_impl(func, in_spec, *flat_args):
        >>>     args, kwargs = pytree.tree_unflatten(flat_args, in_spec)
        >>>     output = func(*args, **kwargs)
        >>>     return output

        flat_apply supports the following two cases:
        - an input type is a container type (e.g. of tensors) registered as a pytree.
        We'll tree_flatten the input type and store the spec.
        - an input type is a constant type (i.e. torch.compile will specialize on it)
        registered with pytree.register_constant. The constant type goes directly
        into the spec.
        """
        assert isinstance(func, _op_types) or pytree._is_constant_holder(func)
        assert len(_unused) == 0
        # pyrefly: ignore[bad-argument-type]  # pyrefly bug?
        return impl(func, in_spec, flat_args, checked_output)


@overload
def is_valid_output(x: tuple[object, ...]) -> TypeIs[tuple[_FXOutput, ...]]: ...


@overload
def is_valid_output(x: Sequence[object]) -> TypeIs[Sequence[_FXOutput]]: ...


def is_valid_output(x: object) -> bool:
    if isinstance(x, (tuple, list)):
        return all(map(is_valid_output, x))
    return is_graphable(x)


def impl(
    func: _OpTypes | pytree.TreeSpec,
    in_spec: pytree.TreeSpec,
    flat_args: tuple[Unpack[_Ts]],
    checked_output: bool,
) -> _FXOutput:
    if isinstance(func, pytree.TreeSpec):
        # assume _ConstantFunction
        func = pytree._retrieve_constant(func)
        assert isinstance(func, _ConstantFunction)

    # pyrefly: ignore[bad-argument-type]  # pyrefly bug?
    args, kwargs = from_graphable(flat_args, in_spec)
    out = func(*args, **kwargs)

    if checked_output:
        # For "normal" usage all outputs must either be graphable or
        # lists/tuples of graphables.
        assert is_valid_output(out)
    return out


flat_apply = FlatApply()
