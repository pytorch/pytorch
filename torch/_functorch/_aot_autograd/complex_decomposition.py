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


def decompose_complex_in_graph(
    gm: fx.GraphModule,
    flat_args: list[Any],
    decompositions: Mapping[OpOverload, Callable[..., Any]] | None = None,
) -> fx.GraphModule:
    if not _has_complex(gm, flat_args):
        return gm

    def wrapper(*args: Any) -> Any:
        wrapped = tuple(_maybe_wrap(a) for a in args)
        with WrapComplexMode():
            result = fx.Interpreter(gm).run(*wrapped)
        return pytree.tree_map(_maybe_unwrap, result)

    with fx_traceback.preserve_node_meta():
        return make_fx(wrapper, decomposition_table=decompositions)(*flat_args)
