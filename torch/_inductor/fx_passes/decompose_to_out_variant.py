"""
FX pass to decompose functional custom operators to their out variants for memory reuse.

This pass uses backward liveness analysis to identify when an input buffer's last use
is at a custom op call. When the input is dead after the op and compatible with the
output, the pass replaces the functional call with the out variant.
"""

from __future__ import annotations

import logging

import torch
from torch._inductor import config
from torch._library._autogen_out import to_out_variant
from torch.utils._ordered_set import OrderedSet


log = logging.getLogger(__name__)


def decompose_to_out_variant(graph: torch.fx.Graph) -> None:
    """
    FX pass that replaces functional *custom* ops with their out variants
    when an input buffer can be reused for the output.

    1. Initialize live set with nodes used by graph outputs
    2. Traverse nodes in reverse execution order
    3. Add node's inputs to live set (meaning they're needed at this point in execution)
    4. For each custom op with an out variant:
       - If an input is NOT in live set and is compatible with the output,
       replace the op with the out variant, and use that input as the out.
    """
    if not config.decompose_to_out_variant:
        return

    live: OrderedSet[torch.fx.Node] = OrderedSet()

    for node in graph.nodes:
        if node.op == "output":
            for arg in node.all_input_nodes:
                live.add(arg)

    for node in list(reversed(graph.nodes)):
        # We only care about custom ops
        if not (
            node.op == "call_function"
            and isinstance(node.target, torch._ops.OpOverload)
            and node.target.namespace != "aten"
        ):
            for arg in node.all_input_nodes:
                live.add(arg)
            continue

        out_variant = to_out_variant(node.target)
        if out_variant is None:
            for arg in node.all_input_nodes:
                live.add(arg)
            continue

        reusable_input = find_dead_reusable_input(node, live)

        for arg in node.all_input_nodes:
            live.add(arg)
        if reusable_input is not None:
            log.debug(
                "Decomposing %s to out variant, reusing buffer from %s",
                node.name,
                reusable_input.name,
            )
            replace_with_out_variant(graph, node, out_variant, reusable_input)

    graph.lint()


def find_dead_reusable_input(
    node: torch.fx.Node, live: OrderedSet[torch.fx.Node]
) -> torch.fx.Node | None:
    """
    Find an input that is dead (not in live set), not a placeholder, and has the
    same shape/dtype as the output.
    """
    output_meta = node.meta.get("tensor_meta")
    if output_meta is None:
        return None

    for input_node in node.all_input_nodes:
        if input_node in live:
            continue
        if input_node.op == "placeholder":
            continue

        input_meta = input_node.meta.get("tensor_meta")
        if input_meta is None:
            continue

        # Require exact dtype and shape match for buffer reuse
        if (
            input_meta.dtype == output_meta.dtype
            and input_meta.shape == output_meta.shape
        ):
            return input_node

    return None


def replace_with_out_variant(
    graph: torch.fx.Graph,
    node: torch.fx.Node,
    out_variant: torch._ops.OpOverload,
    reusable_input: torch.fx.Node,
) -> None:
    """Replace a functional call with its out variant call."""
    with graph.inserting_before(node):
        out_param_names = _get_out_param_names(out_variant._schema)
        if not out_param_names:
            return

        new_kwargs = dict(node.kwargs)
        if len(out_param_names) == 1:
            new_kwargs[out_param_names[0]] = reusable_input
        else:
            new_kwargs[out_param_names[0]] = reusable_input

        out_call = graph.call_function(
            out_variant,
            args=node.args,
            kwargs=new_kwargs,
        )

        # Copy metadata but exclude eager_input_vals which would cause
        # normalization issues with the new schema
        out_call.meta = {k: v for k, v in node.meta.items() if k != "eager_input_vals"}

        node.replace_all_uses_with(out_call)
        graph.erase_node(node)


def _get_out_param_names(schema: torch._C.FunctionSchema) -> list[str]:
    """Get the names of out parameters from a schema."""
    out_names = []
    for arg in schema.arguments:
        if arg.kwarg_only and arg.alias_info is not None and arg.alias_info.is_write:
            out_names.append(arg.name)
    return out_names
