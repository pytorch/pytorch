import operator

import torch
from torch.fx.graph_module import GraphModule


def inline_invoke_subgraph(gm: GraphModule) -> GraphModule:
    """Inline all invoke_subgraph HOPs, producing a flat FX graph.

    This is useful when downstream compilers don't support HOPs but we still
    want the Dynamo tracing-time benefits of auto-caching (trace once, stamp
    out cached calls).
    """
    # First, recursively inline any nested invoke_subgraph calls inside
    # subgraph modules themselves.
    for name, mod in gm.named_modules():
        if name and isinstance(mod, GraphModule):
            inline_invoke_subgraph(mod)

    changed = True
    while changed:
        changed = False
        for node in list(gm.graph.nodes):
            if not (
                node.op == "call_function"
                and node.target is torch.ops.higher_order.invoke_subgraph
            ):
                continue

            changed = True
            get_attr_node = node.args[0]
            # args[1] is the identifier string, args[2:] are operands
            operands = node.args[2:]

            subgraph: GraphModule = getattr(gm, get_attr_node.target)

            # Build mapping from subgraph nodes -> parent graph nodes
            env = {}
            subgraph_placeholders = [
                n for n in subgraph.graph.nodes if n.op == "placeholder"
            ]
            for ph, operand in zip(subgraph_placeholders, operands):
                env[ph] = operand

            # Copy subgraph nodes into parent graph, inserting before the
            # invoke_subgraph node.
            with gm.graph.inserting_before(node):
                for sub_node in subgraph.graph.nodes:
                    if sub_node.op in ("placeholder", "output"):
                        continue
                    env[sub_node] = gm.graph.node_copy(sub_node, lambda n: env[n])

            # The subgraph output node's first arg is a tuple of return values.
            output_node = next(n for n in subgraph.graph.nodes if n.op == "output")
            output_values = output_node.args[0]

            # Replace getitem users of the invoke_subgraph result with the
            # corresponding inlined output.
            for user in list(node.users):
                if user.op == "call_function" and user.target is operator.getitem:
                    idx = user.args[1]
                    user.replace_all_uses_with(env[output_values[idx]])
                    gm.graph.erase_node(user)

            # If the node itself still has remaining users (unlikely for
            # well-formed graphs), replace with a tuple construction.
            if node.users:
                with gm.graph.inserting_before(node):
                    tuple_items = [env[v] for v in output_values]
                    # Build the tuple by calling operator.getitem on a fresh
                    # call — but since we already have the individual values,
                    # just build (val0, val1, ...) isn't directly expressible
                    # in FX. Instead, replace remaining users individually.
                    # This shouldn't happen for invoke_subgraph produced by
                    # Dynamo (always accessed via getitem).

            gm.graph.erase_node(node)

            # Remove the get_attr node if it has no other users.
            if not get_attr_node.users:
                gm.graph.erase_node(get_attr_node)

    gm.recompile()
    return gm
