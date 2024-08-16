# mypy: allow-untyped-defs
"""Common utility functions for FX passes.

These functions should NOT be directly invoked outside of `passes` package.
"""
from __future__ import annotations

import collections
import re
from typing import Callable

import torch.fx
import torch.fx.traceback as fx_traceback


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


def _get_node_base_name(node_name: str) -> tuple[str, int | None]:
    pattern = r"(.*)\.(\d+)"
    match = re.match(pattern, node_name)
    if match is not None:
        base_name, count_str = match.groups()
        return base_name, int(count_str)
    return node_name, None


def set_node_name(
    node: torch.fx.Node,
    new_name: str,
    name_to_node_cache: dict[str, torch.fx.Node],
):
    """Safely set the unique name of a node.

    If the new name is already taken by another node, the name of the other node will be
    updated. If `new_name` is a string of format f"{base_name}.{count}", where `count`
    is an integer, the other node will be renamed as f"{base_name}.{count+1}". If not,
    the other node will be renamed as "{new_name}.1". This function will iteratively
    update the names until there is no conflict.

    ``name_to_node_cache`` is required as an argument to avoid recomputation. The caller
    is responsible for ensuring the cache is accurate and in sync with the owning module
    of the node. The values in the cache will be updated accordingly.

    Args:
        node: The node to update.
        new_name: The new name to use.
        name_to_node_cache: A cache of node names to nodes.
    """
    module = node.graph.owning_module
    node_name_to_set = collections.deque([(node, new_name)])

    while node_name_to_set:
        node, new_name = node_name_to_set.pop()
        if new_name in name_to_node_cache and name_to_node_cache[new_name] != node:
            base_name, postfix_count = _get_node_base_name(new_name)
            if postfix_count is None:
                postfix_count = 0
            node_name_to_set.append(
                (name_to_node_cache[new_name], f"{base_name}.{postfix_count + 1}")
            )
        node.name = new_name
        name_to_node_cache[new_name] = node


def replace_placeholder_name_and_target(
    module: torch.fx.GraphModule, reference_module: torch.fx.GraphModule
):
    """Replace the argument names in module with those in reference_module.

    This function assumes the two modules have the same signature structure.
    The caller is responsible for ensuring this. Otherwise, the behavior of this
    function is undefined. This function only does minimal sanity check that the two
    modules have the same number of arguments.

    Name conflicts between new names and existing node names in the graph are handled.
    Check the documentation of :func:`set_node_name` for more details.

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

    name_to_node: dict[str, torch.fx.Node] = {}
    for node in module.graph.nodes:
        name_to_node[node.name] = node

    for placeholder, reference_placeholder in zip(placeholders, reference_placeholders):
        placeholder.target = reference_placeholder.target
        set_node_name(placeholder, reference_placeholder.name, name_to_node)

    module.recompile()
