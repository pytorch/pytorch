import operator
from typing import Any, TYPE_CHECKING

import torch
from torch.fx.graph_module import GraphModule


if TYPE_CHECKING:
    from torch.fx.node import Node


def inline_invoke_subgraph(gm: GraphModule) -> GraphModule:
    """Inline all invoke_subgraph HOPs, producing a flat FX graph.

    This is useful when downstream compilers (like vllm-compile) don't support
    HOPs or prefer a flat graph, but we still want the Dynamo tracing-time
    benefits of auto-caching (trace once, stamp out cached calls).
    """
    # First, recursively inline any nested invoke_subgraph calls inside
    # subgraph modules themselves.
    for name, mod in gm.named_modules():
        if name and isinstance(mod, GraphModule):
            inline_invoke_subgraph(mod)

    invoke_nodes = list(
        gm.graph.find_nodes(
            op="call_function", target=torch.ops.higher_order.invoke_subgraph
        )
    )
    for node in invoke_nodes:
        get_attr_node = node.args[0]
        # args[1] is the identifier string, args[2:] are operands
        operands = node.args[2:]

        subgraph: GraphModule = getattr(gm, get_attr_node.target)

        # Build mapping from subgraph placeholder nodes -> parent operands
        env: dict[Node, Any] = dict(
            zip(subgraph.graph.find_nodes(op="placeholder"), operands)
        )

        # Copy subgraph nodes into parent graph, inserting before the
        # invoke_subgraph node.
        with gm.graph.inserting_before(node):
            for sub_node in subgraph.graph.nodes:
                if sub_node.op in ("placeholder", "output"):
                    continue
                env[sub_node] = gm.graph.node_copy(sub_node, lambda n: env[n])

        output_values = subgraph.graph.output_node().args[0]

        # Replace getitem users of the invoke_subgraph result with the
        # corresponding inlined output.
        for user in list(node.users):
            if user.op == "call_function" and user.target is operator.getitem:
                idx = user.args[1]
                user.replace_all_uses_with(env[output_values[idx]])  # pyrefly: ignore
                gm.graph.erase_node(user)

        gm.graph.erase_node(node)

        # Remove the get_attr node if it has no other users.
        if not get_attr_node.users:
            gm.graph.erase_node(get_attr_node)

    gm.recompile()
    return gm
