from __future__ import annotations

from typing import Callable

import torch.fx
import torch.fx.traceback as fx_traceback
from torch.onnx._internal import _beartype


@_beartype.beartype
def wrap_graph_module_for_node_meta_preservation(
    graph_module: torch.fx.GraphModule,
) -> Callable:
    """Wrap a GraphModule with contexts to preserve node meta information, such as stacktrace info.

    This is typically useful before calling `make_fx`. Without this wrapper, the
    stacktrace information will be lost afterwards.
    """

    def wrapped(*args):
        with fx_traceback.preserve_node_meta():
            return torch.fx.Interpreter(graph_module).run(*args)

    return wrapped


@_beartype.beartype
def replace_placeholder_name_and_target(
    module: torch.fx.GraphModule, reference_module: torch.fx.GraphModule
):
    """Replace the argument names in module with those in reference_module.

    This function assumes the two modules have the same signature structure.

    Raises:
        RuntimeError: If the two modules have different number of arguments.
    """
    placeholders = [node for node in module.graph.nodes if node.op == "placeholder"]
    reference_placeholders = [
        node for node in reference_module.graph.nodes if node.op == "placeholder"
    ]

    if len(placeholders) != len(reference_placeholders):
        raise RuntimeError(
            "The two modules have different number of arguments. "
            f"module: {len(placeholders)}, reference_module: {len(reference_placeholders)}"
        )

    for placeholder, reference_placeholder in zip(placeholders, reference_placeholders):
        placeholder.target = reference_placeholder.target
        placeholder.name = reference_placeholder.name

    module.recompile()
