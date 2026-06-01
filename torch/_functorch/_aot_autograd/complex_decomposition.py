"""
Graph pass to decompose complex-valued operations into real-valued operations.

This pass runs AFTER functionalization, so the graph is purely functional (no
mutations, no aliasing). It retraces the graph with ComplexTensor dispatch active,
which decomposes each complex op into real-valued sub-ops. The graph signature
(input/output count and dtypes) is preserved: complex inputs get unpacked via
aten.real/aten.imag at the top, and complex outputs get repacked via aten.complex
at the bottom.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import torch
import torch.fx as fx
import torch.fx.traceback as fx_traceback
import torch.utils._pytree as pytree
from torch import Tensor
from torch._subclasses.complex_tensor import ComplexTensor, WrapComplexMode
from torch.fx.experimental.proxy_tensor import make_fx


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from torch._ops import OpOverload


def _can_wrap(value: Any) -> bool:
    return (
        isinstance(value, Tensor)
        and value.dtype.is_complex
        and not isinstance(value, ComplexTensor)
    )


def _maybe_wrap(value: Any) -> Any:
    if _can_wrap(value):
        return ComplexTensor(torch.real(value), torch.imag(value))
    return value


def _maybe_unwrap(value: Any) -> Any:
    if isinstance(value, ComplexTensor):
        return value.as_interleaved()
    return value


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


def _aliased_input_indices(op: torch._ops.OpOverload) -> list[int]:
    schema = op._schema
    ret_aliases = set()
    idxs = []

    for ret in schema.returns:
        ret_alias = ret.alias_info
        if ret_alias is None:
            continue
        ret_aliases.update(ret_alias.before_set)

    if len(ret_aliases) == 0:
        return idxs

    for i, arg in enumerate(schema.arguments):
        if arg.alias_info is None:
            continue
        if not arg.alias_info.before_set.isdisjoint(ret_aliases):
            idxs.append(i)

    return idxs


def _collect_storage_aliases(node: fx.Node) -> set[fx.Node]:
    seen: set[fx.Node] = set()
    stack: list[fx.Node] = [node]
    while stack:
        n = stack.pop()
        if n in seen:
            continue
        seen.add(n)
        if not isinstance(n.target, torch._ops.OpOverload):
            continue
        idxs = _aliased_input_indices(n.target)
        for idx in idxs:
            src = n.args[idx]
            if isinstance(src, fx.Node):
                stack.append(src)
    return seen


def _assert_no_incorrect_aliasing_mutation(gm: fx.GraphModule) -> None:
    """
    In this function, we check that the compiled graph respects aliasing semantics.

    There are three cases in which this check fires:
        1. A complex tensor is mutated in place via an incorrect alias.
           The ops that produce incorrect aliases are listed in `INCORRECT_ALIASING_OPS`.
        2. A complex input is mutated in place. This is not allowed because complex inputs
           decompose to real and imaginary parts, which are made contiguous for performance.
           Therefore, any complex input passed in does not alias the original tensor.
        3. A complex input aliases an output. This is not allowed as the input and output
           tensors can't alias each other because of 2.

    In all three cases, we raise an error if possible incorrectness is detected.
    """
    from torch._subclasses.complex_tensor._ops.common import INCORRECT_ALIASING_OPS

    mutated: set[fx.Node] = set()
    complex_inputs: set[fx.Node] = set()
    for n in gm.graph.nodes:
        if n.op == "placeholder" and n.meta["val"].dtype.is_complex:
            complex_inputs.add(n)
        if (
            isinstance(n.target, torch._ops.OpOverload)
            and n.target.overloadpacket == torch.ops.aten.copy_
        ):
            mutated.add(n.args[0])

    poisoned: set[fx.Node] = set()
    for n in gm.graph.nodes:
        if isinstance(n.target, torch._ops.OpOverload) and (
            n.target.overloadpacket in INCORRECT_ALIASING_OPS
            or n.target in INCORRECT_ALIASING_OPS
        ):
            poisoned.update(_collect_storage_aliases(n))
    if not mutated.isdisjoint(poisoned):
        raise RuntimeError(
            "torch.view_as_real or torch.view_as_complex was called on a complex tensor, and its storage is also mutated in this compiled region. Please clone the tensor before mutating, move the mutation outside the compiled region, or avoid these ops."
        )

    complex_input_aliases: set[fx.Node] = set()
    for n in complex_inputs:
        complex_input_aliases.update(_collect_storage_aliases(n))

    if not mutated.isdisjoint(complex_input_aliases):
        raise RuntimeError(
            "Complex input nodes are mutated in this compiled region. Please clone the tensor before mutating, move the mutation outside the compiled region, or avoid these ops."
        )


def decompose_complex_in_graph(
    gm: fx.GraphModule,
    flat_args: list[Any],
    decompositions: Mapping[OpOverload, Callable[..., Any]] | None = None,
) -> fx.GraphModule:
    if not _has_complex(gm, flat_args):
        return gm

    _assert_no_incorrect_aliasing_mutation(gm)

    def wrapper(*args: Any) -> Any:
        wrapped = tuple(_maybe_wrap(a) for a in args)
        with WrapComplexMode():
            result = fx.Interpreter(gm).run(*wrapped)
        return pytree.tree_map(_maybe_unwrap, result)

    with fx_traceback.preserve_node_meta():
        return make_fx(wrapper, decomposition_table=decompositions)(*flat_args)
