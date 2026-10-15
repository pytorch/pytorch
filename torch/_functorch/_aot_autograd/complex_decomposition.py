"""
Graph pass to decompose complex-valued operations into real-valued operations.

This pass runs AFTER functionalization, so the only remaining mutations are `copy_`
nodes into graph inputs, and the only aliasing is outputs that view inputs. It retraces
the graph with ComplexTensor dispatch active, which decomposes each complex op into
real-valued sub-ops. The graph signature (input/output count and dtypes) is preserved:
complex inputs get unpacked via aten.real/aten.imag at the top, and complex outputs get
repacked via aten.complex at the bottom. Inputs that are mutated or aliased by an output
are wrapped in InterleavedComplexTensor instead, which views the input's storage.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import torch
import torch.fx as fx
import torch.fx.traceback as fx_traceback
import torch.utils._pytree as pytree
from torch import Tensor
from torch._subclasses.complex_tensor import (
    ComplexTensor,
    InterleavedComplexTensor,
    WrapComplexMode,
)
from torch.fx.experimental.proxy_tensor import make_fx


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from torch._ops import OpOverload


def _can_wrap(value: Any) -> bool:
    return (
        isinstance(value, Tensor)
        and value.dtype.is_complex
        and not isinstance(value, ComplexTensor | InterleavedComplexTensor)
    )


def _maybe_wrap(value: Any, interleaved: bool) -> Any:
    if not _can_wrap(value):
        return value
    if interleaved:
        return InterleavedComplexTensor(value)
    return ComplexTensor(torch.real(value), torch.imag(value))


def _maybe_unwrap(value: Any) -> Any:
    if isinstance(value, ComplexTensor | InterleavedComplexTensor):
        return value.as_interleaved()
    return value


def _view_base(node: Any) -> Any:
    while (
        isinstance(node, fx.Node)
        and isinstance(node.target, torch._ops.OpOverload)
        and node.target.is_view
    ):
        node = node.args[0]
    return node


def _needs_interleaved(gm: fx.GraphModule) -> list[bool]:
    """
    For each graph input, whether it must stay an `InterleavedComplexTensor`.

    That is required for inputs mutated in the graph (the `copy_` must reach the caller's
    storage) and for inputs aliased by an output. `ComplexTensor` copies its input,
    so it is used for all other inputs.
    """
    keep = set()
    for n in gm.graph.nodes:
        if (
            isinstance(n.target, torch._ops.OpOverload)
            and n.target.overloadpacket == torch.ops.aten.copy_
        ):
            keep.add(_view_base(n.args[0]))
    for out in pytree.tree_leaves(gm.graph.output_node().args[0]):
        keep.add(_view_base(out))
    return [p in keep for p in gm.graph.find_nodes(op="placeholder")]


def _has_complex(gm: fx.GraphModule, flat_args: list[Any]) -> bool:
    for arg in flat_args:
        if _can_wrap(arg):
            return True
    for node in gm.graph.nodes:
        if node.op == "call_function":
            val = node.meta.get("val")
            if _can_wrap(val):
                return True
    return False


class _ComplexInterpreter(fx.Interpreter):
    def __init__(self, gm: fx.GraphModule, raw_inputs: dict[int, Tensor]) -> None:
        super().__init__(gm)
        # Maps id() of each wrapped InterleavedComplexTensor input to the unwrapped input.
        self.raw_inputs = raw_inputs

    def call_function(self, target: Any, args: Any, kwargs: Any) -> Any:
        if (
            isinstance(target, torch._ops.OpOverload)
            and target.overloadpacket == torch.ops.aten.copy_
            and id(args[0]) in self.raw_inputs
        ):
            # Inductor only applies input mutations that `copy_` into the graph input
            # itself; InterleavedComplexTensor would write into a view of it instead.
            raw = self.raw_inputs[id(args[0])]
            raw.copy_(_maybe_unwrap(args[1]), *args[2:], **kwargs)
            return args[0]
        with WrapComplexMode():
            return super().call_function(target, args, kwargs)


def decompose_complex_in_graph(
    gm: fx.GraphModule,
    flat_args: list[Any],
    decompositions: Mapping[OpOverload, Callable[..., Any]] | None = None,
) -> fx.GraphModule:
    if not _has_complex(gm, flat_args):
        return gm

    interleaved = _needs_interleaved(gm)

    def wrapper(*args: Any) -> Any:
        wrapped = tuple(
            _maybe_wrap(a, i) for a, i in zip(args, interleaved, strict=True)
        )
        raw_inputs = {
            id(w): a
            for w, a in zip(wrapped, args, strict=True)
            if isinstance(w, InterleavedComplexTensor)
        }
        result = _ComplexInterpreter(gm, raw_inputs).run(*wrapped)
        return pytree.tree_map(_maybe_unwrap, result)

    with fx_traceback.preserve_node_meta():
        return make_fx(wrapper, decomposition_table=decompositions)(*flat_args)
