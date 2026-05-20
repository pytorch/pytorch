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

from collections.abc import Callable, Mapping
from typing import Any

import torch
import torch.fx as fx
import torch.utils._pytree as pytree
from torch import Tensor
from torch._ops import OpOverload
from torch._subclasses.complex_tensor import ComplexTensor
from torch.fx.experimental.proxy_tensor import make_fx


class _ComplexDecompInterpreter(fx.Interpreter):
    """Interpreter that decomposes complex ops via ComplexTensor dispatch.

    When a node's result is a complex tensor (but not already a ComplexTensor),
    it gets wrapped so downstream nodes decompose correctly.
    """

    def call_function(
        self, target: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> Any:
        if (
            isinstance(target, torch._ops.OpOverload)
            and target.overloadpacket == torch.ops.aten.complex
        ):
            return ComplexTensor(args[0], args[1])

        result = super().call_function(target, args, kwargs)
        return _maybe_wrap(result)


def _maybe_wrap(value: Any) -> Any:
    if (
        isinstance(value, Tensor)
        and not isinstance(value, ComplexTensor)
        and value.dtype.is_complex
    ):
        return ComplexTensor(torch.real(value), torch.imag(value))
    return value


def _maybe_unwrap(value: Any) -> Any:
    if isinstance(value, ComplexTensor):
        return value.as_interleaved()
    return value


def _has_complex(gm: fx.GraphModule, example_inputs: list[Any]) -> bool:
    for inp in example_inputs:
        if isinstance(inp, Tensor) and inp.dtype.is_complex:
            return True
    for node in gm.graph.nodes:
        if node.op == "call_function":
            val = node.meta.get("val")
            if isinstance(val, Tensor) and val.dtype.is_complex:
                return True
    return False


def decompose_complex_in_graph(
    gm: fx.GraphModule,
    example_inputs: list[Any],
    decompositions: Mapping[OpOverload, Callable[..., Any]] | None = None,
) -> fx.GraphModule:
    if not _has_complex(gm, example_inputs):
        return gm

    def wrapper(*args: Any) -> Any:
        wrapped = tuple(_maybe_wrap(a) for a in args)
        result = _ComplexDecompInterpreter(gm).run(*wrapped)
        return pytree.tree_map(_maybe_unwrap, result)

    return make_fx(wrapper, decomposition_table=decompositions or {})(*example_inputs)
