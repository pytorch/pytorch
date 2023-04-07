"""Common utility functions for FX passes.

These functions should NOT be directly invoked outside of `passes` package.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional

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
def set_node_name(
    node: torch.fx.Node,
    new_name: str,
    name_to_node_cache: Optional[Dict[str, torch.fx.Node]] = None,
):
    """Safely set the unique name of a node.

    If the new name is already taken by another node, the name of the other node will be
    updated to be "{new_name}.1". This function will recursively update the names until
    there is no conflict.

    To avoid recomputing the name_to_node_cache, it can be provided as an argument. If
    provided, the caller is responsible for ensuring the cache is accurate and in sync
    with the owning module of the node.

    Args:
        node: The node to update.
        new_name: The new name to use.
        name_to_node_cache: A cache of node names to nodes. If not provided, this
            function will build the cache from the owning module of the node.
    """
    module = node.graph.owning_module
    name_to_node_cache = name_to_node_cache or {
        _node.name: _node for _node in module.graph.nodes
    }

    if new_name in name_to_node_cache and name_to_node_cache[new_name] != node:
        set_node_name(name_to_node_cache[new_name], f"{new_name}.1")

    node.name = new_name
    name_to_node_cache[new_name] = node


@_beartype.beartype
def replace_placeholder_name_and_target(
    module: torch.fx.GraphModule, reference_module: torch.fx.GraphModule
):
    """Replace the argument names in module with those in reference_module.

    This function assumes the two modules have the same signature structure.
    The caller is responsible for ensuring this. Otherwise, the behavior of this
    function is undefined. This function only does minimal sanity check that the two
    modules have the same number of arguments.

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

    name_to_node: Dict[str, torch.fx.Node] = {}
    for node in module.graph.nodes:
        name_to_node[node.name] = node

    for placeholder, reference_placeholder in zip(placeholders, reference_placeholders):
        placeholder.target = reference_placeholder.target
        set_node_name(placeholder, reference_placeholder.name, name_to_node)

    module.recompile()
