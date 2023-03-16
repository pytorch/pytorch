from __future__ import annotations

import operator
from typing import Callable, List, Sequence, Union

import torch
import torch._ops
import torch.fx

from torch.onnx._internal.fx import errors, function_dispatcher

_TARGET_TYPE = Union[Callable, torch._ops.OpOverload]


def unsupported_call_functions(
    graph_module: torch.fx.GraphModule,
) -> Sequence[_TARGET_TYPE]:
    ops: List[_TARGET_TYPE] = []
    for node in graph_module.graph.nodes:
        if node.op == "call_function":
            if node.target == operator.getitem:
                # FIXME: This op is special handled in `_export_fx_node_to_onnxscript`.
                continue
            try:
                function_dispatcher.find_symbolic_function(node.target)
            except errors.UnsupportedCallFunctionError as e:
                ops.append(node.target)

    return ops
