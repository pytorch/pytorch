from __future__ import annotations

from typing import Callable, Dict

import torch
import torch._ops
import torch.fx
from torch.fx import traceback as fx_traceback
from torch.fx.experimental import proxy_tensor

from torch.onnx._internal import _beartype


@_beartype.beartype
def _rename_placeholder_targets(
    module: "torch.fx.GraphModule", reference_module: "torch.fx.GraphModule"
):
    """Align the argument names in module with those in reference_module.
    After calling this function, the two forward(...) in module and reference_module should have
    the same signature.
    """
    placeholders = [node for node in module.graph.nodes if node.op == "placeholder"]
    reference_placeholders = [
        node for node in reference_module.graph.nodes if node.op == "placeholder"
    ]

    for placeholder, reference_placeholder in zip(placeholders, reference_placeholders):
        placeholder.target = reference_placeholder.target
        placeholder.name = reference_placeholder.name

    module.recompile()


def decompose(
    module: torch.fx.GraphModule,
    decomposition_table: Dict[torch._ops.OpOverload, Callable],
    *args,
) -> torch.fx.GraphModule:
    # A trick adopted from `dynamo.export` in `eval_frame.py`.
    # Running graph with interpreter is needed for propagating the stack_trace.

    def graph_with_interpreter(*args):
        with fx_traceback.preserve_node_meta():
            return torch.fx.Interpreter(module).run(*args)

    # Apply decomposition table to the input graph.
    # Make sure the feed-in "module" is stateless.
    decomposed_module = proxy_tensor.make_fx(
        graph_with_interpreter,
        decomposition_table=decomposition_table,
        tracing_mode="fake",
        _allow_non_fake_inputs=True,
    )(*args)
    # Rename placeholder targets to match the original module's signature since
    # We don't want to map forward(x, y, z) to forward(arg0, arg1, arg2).
    _rename_placeholder_targets(decomposed_module, module)

    return decomposed_module
