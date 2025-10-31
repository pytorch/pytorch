# mypy: allow-untyped-defs
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


def map_recorded_events_to_aten_ops_with_stack_trace(graph_module, traced_data):
    """
    Maps recorded profiler events to their corresponding aten operations and adds stack traces.

    Args:
        graph_module: The FX GraphModule
        traced_data: Json of profiler events from Chrome trace

    Returns:
        Dict mapping recorded event names to their aten operations with added stack traces
    """
    trace_events = traced_data.get("traceEvents", [])

    # Create a mapping from node name to node for easy lookup
    node_map = {node.name: node for node in graph_module.graph.nodes}


    # Find aten operation events
    aten_events = [e for e in trace_events if e.get("cat") == "cpu_op"]

    # Map recorded events to aten ops and add stack traces
    event_mapping = {}

    for recorded_event in trace_events:
        if (recorded_event.get("cat") in ["cpu_op"] and
            recorded_event.get("name", "").startswith("## ") and
            recorded_event.get("name", "").endswith(" ##")):
            # Extract node name from "## node_name ##"
            node_name = recorded_event["name"][3:-3]  # Remove "## " and " ##"

            if node_name in node_map:
                node = node_map[node_name]

                # Find corresponding aten operations within this recorded event's time window
                recorded_start = recorded_event["ts"]
                recorded_end = recorded_start + recorded_event["dur"]

                # Find aten ops that fall within this time window
                corresponding_aten_ops = []
                for aten_event in aten_events:
                    aten_start = aten_event["ts"]
                    aten_end = aten_start + aten_event["dur"]

                    # Check if aten event overlaps with recorded event
                    if (aten_start >= recorded_start and aten_start <= recorded_end) or \
                    (aten_end >= recorded_start and aten_end <= recorded_end) or \
                    (aten_start <= recorded_start and aten_end >= recorded_end):
                        corresponding_aten_ops.append(aten_event)

                # Add stack trace to recorded event and aten ops
                stack_trace = node.meta.get("stack_trace", "No stack trace available")

                # Add stack trace to the recorded event
                if "args" not in recorded_event:
                    recorded_event["args"] = {}
                recorded_event["args"]["stack_trace"] = stack_trace

                # Add stack trace to corresponding aten ops
                for aten_op in corresponding_aten_ops:
                    if "args" not in aten_op:
                        aten_op["args"] = {}
                    aten_op["args"]["stack_trace"] = stack_trace

                event_mapping[node_name] = {
                    "recorded_event": recorded_event,
                    "aten_operations": corresponding_aten_ops,
                    "node": node,
                    "stack_trace": stack_trace
                }

    return event_mapping
