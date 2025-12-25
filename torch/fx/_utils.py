# mypy: allow-untyped-defs
import functools
import sys
from typing import Optional

import torch
from torch._logging import LazyString


def lazy_format_graph_code(name, gm, maybe_id=None, **kwargs):
    """
    Returns a LazyString that formats the graph code.
    """

    def format_name():
        if maybe_id is not None:
            return f"{name} {maybe_id}"
        else:
            return name

    if "print_output" not in kwargs:
        kwargs["print_output"] = False

    if "colored" in kwargs:
        try:
            if not sys.stdout.isatty():
                kwargs["colored"] = False
        except AttributeError:
            kwargs["colored"] = False

    return LazyString(
        lambda: _format_graph_code(
            f"===== {format_name()} =====\n",
            gm.forward.__code__.co_filename,
            gm.print_readable(**kwargs),
        )
    )


def _format_graph_code(name, filename, graph_str):
    """
    Returns a string that formats the graph code.
    """
    return f"TRACED GRAPH\n {name} {filename} {graph_str}\n"


def first_call_function_nn_module_stack(graph: torch.fx.Graph) -> Optional[dict]:
    """
    Returns the nn_module_stack of the first call_function node.
    """
    for node in graph.nodes:
        if node.op == "call_function" and "nn_module_stack" in node.meta:
            return node.meta["nn_module_stack"]
    return None


def get_node_context(node, num_nodes=2) -> str:
    """
    Returns a string of the last num_nodes nodes in the graph.
    """
    node_contexts = []
    cur = node
    for _ in range(num_nodes):
        node_contexts.append(cur.format_node())
        if cur.op == "root":
            break
        cur = cur.prev
    return "\n".join(node_contexts[::-1])


# standalone_inductor returns a callable class object - this does not sit well
# with Fx graph node op call_function which expects a function. So this is just
# a wrapper function to make Fx graph codegen happy.
def _dummy_wrapper(fn):
    @functools.wraps(fn)
    def inner(*args, **kwargs):
        return fn(*args, **kwargs)

    return inner
