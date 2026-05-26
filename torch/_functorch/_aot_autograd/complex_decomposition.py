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
    # view_as_real decomposes to torch.stack (a copy) under complex_wrapper,
    # so the result no longer aliases its input. If any storage that aliases
    # a view_as_real input is also mutated (aten.copy_.default epilogue, the
    # only mutating op post-functionalization), the compiled graph would
    # silently diverge from eager. Fail loudly instead.
    from torch._subclasses.complex_tensor._ops.common import INCORRECT_ALIASING_OPS

    # Output nodes always have an iterable `.args[0]`
    mutated: set[fx.Node] = {*gm.graph.output_node().args[0]}  # type: ignore[not-iterable]
    for n in gm.graph.nodes:
        if n.op == "placeholder":
            mutated.add(n)
        if (
            isinstance(n.target, torch._ops.OpOverload)
            and n.target.overloadpacket == torch.ops.aten.copy_
        ):
            mutated.add(n.arg[0])

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

    return make_fx(wrapper, decomposition_table=decompositions)(*flat_args)
