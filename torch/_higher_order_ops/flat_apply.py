import typing
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing_extensions import TypeIs

import torch
import torch.fx.node
import torch.utils._pytree as pytree
from torch._ops import HigherOrderOperator


def is_graphable(val: object) -> TypeIs[torch.fx.node.BaseArgumentTypes]:
    """Definition: a graphable type is a type that that is an acceptable input/output type to a FX node."""
    return isinstance(val, torch.fx.node.base_types)


def is_graphable_type(typ: type[object]) -> bool:
    """Return whether the given type is graphable"""
    return issubclass(typ, torch.fx.node.base_types)


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


def from_graphable(flat_args: Iterable[object], spec: pytree.TreeSpec) -> pytree.PyTree:
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


@dataclass(frozen=True)
class _ConstantFunction:
    func: Callable[..., object]

    def __call__(self, *args: object, **kwargs: object):
        return self.func(*args, **kwargs)


pytree.register_constant(_ConstantFunction)


_OpTypes = (
    torch._ops.OpOverload | torch._ops.OpOverloadPacket | torch._ops.HigherOrderOperator
)
_op_types = typing.get_args(_OpTypes)


class FlatApply(HigherOrderOperator):
    def __init__(self) -> None:
        super().__init__("flat_apply")

    def __call__(
        self,
        func: _OpTypes | pytree.TreeSpec,
        in_spec: pytree.TreeSpec,
        *flat_args: object,
        **_unused: object,
    ):
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
        return impl(func, in_spec, flat_args)


def impl(
    func: _OpTypes | pytree.TreeSpec,
    in_spec: pytree.TreeSpec,
    flat_args: tuple[object, ...],
) -> object | tuple[object, pytree.TreeSpec]:
    if not isinstance(func, _op_types):
        # assume _ConstantFunction
        func = pytree._retrieve_constant(func)
        assert isinstance(func, _ConstantFunction)

    args, kwargs = from_graphable(flat_args, in_spec)
    out = func(*args, **kwargs)

    # Right now, all outputs must either be graphable or lists/tuples of graphables.
    #
    # If you need non-graphable outputs then use FlatApplyFlat and unflatten the
    # result in-graph.
    def is_valid_output(x: object) -> bool:
        if isinstance(x, (tuple, list)):
            return all(map(is_valid_output, x))
        return is_graphable(x)

    assert is_valid_output(out)
    return out


flat_apply = FlatApply()


class FlatApplyFlat(HigherOrderOperator):
    def __init__(self) -> None:
        super().__init__("flat_apply_flat")
        self.out_spec: pytree.TreeSpec | None = None

    def __call__(
        self,
        func: _OpTypes | pytree.TreeSpec,
        in_spec: pytree.TreeSpec,
        *flat_args: tuple[object],
        **_unused: object,
    ):
        """
        See flat_apply. FlatApplyFlat works like FlatApply but also flattens the function output.

        The semantics of flat_apply_flat(func, in_spec, *flat_args) are roughly equivalent to:

        >>> def flat_apply_impl(func, in_spec, *flat_args):
        >>>     args, kwargs = pytree.tree_unflatten(flat_args, in_spec)
        >>>     output = func(*args, **kwargs)
        >>>     flat_output, self.out_spec = pytree.tree_flatten(output)
        >>>     return flat_output

        Why is out_spec a constant on the HOP instead of a returned value? Like
        in_spec we want the output to be constant so we can reconstruct the
        output statically.
        """
        assert isinstance(func, _op_types) or pytree._is_constant_holder(func)
        assert len(_unused) == 0

        if not isinstance(func, _op_types):
            # assume _ConstantFunction
            func = pytree._retrieve_constant(func)
            assert isinstance(func, _ConstantFunction)

        args, kwargs = from_graphable(flat_args, in_spec)
        out = func(*args, **kwargs)

        result, out_spec = to_graphable(out)

        if self.out_spec is None:
            self.out_spec = out_spec
        elif self.out_spec != out_spec:
            raise RuntimeError(
                "Invalid function passed to FlatApplyFlat. The called function returned a value with a different pytree shape on a subsequent call."
            )

        return result
