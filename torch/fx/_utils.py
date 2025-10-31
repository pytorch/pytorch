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


def map_recorded_events_to_aten_ops_with_stack_trace(traced_data, remove_fx_events=False):
    """
    Maps recorded profiler events to their corresponding fx nodes and adds stack traces.

    Builds a timeline of all events (regular ops and FX markers for filenames/nodes),
    sorts by timestamp, then processes chronologically while maintaining a context stack of active
    filename/node scopes. Regular events are augmented with stack traces and node names from the
    innermost active context. Runtime is O(n log n) for n events.

    Args:
        traced_data: Json of profiler events from Chrome trace
        remove_fx_events: If True, removes the FX marker events (## ... ##) from the trace

    Returns:
        Dict mapping recorded event names to their aten operations with added stack traces
    """
    from torch.fx.traceback import _FX_METADATA_REGISTRY

    trace_events = traced_data.get("traceEvents", [])

    # Create event timeline: (timestamp, event_type, event_data)
    # event_type: 'start' or 'end'
    event_timeline = []
    fx_marker_events = []  # Track FX marker events to optionally remove them

    for event in trace_events:
        # TODO: do not drop any event. The output should have everything in the input AND the additional information (except those removed by remove_fx_events)
        if "ts" not in event or "dur" not in event:
            continue

        start_ts = event["ts"]
        end_ts = start_ts + event["dur"]

        # Check if this is an FX marker event
        if (event.get("cat") == "cpu_op" and
            event.get("name", "").startswith("## ") and
            event.get("name", "").endswith(" ##")):
            fx_marker_events.append(event)
            content = event["name"][3:-3]

            # Parse the content
            if content.endswith(".py"):
                # This is a graph entry event
                event_timeline.append((start_ts, 'start', 'filename', content, event))
                event_timeline.append((end_ts, 'end', 'filename', content, event))
            else:
                # Try to parse as node index
                try:
                    node_index = int(content)
                    event_timeline.append((start_ts, 'start', 'node', node_index, event))
                    event_timeline.append((end_ts, 'end', 'node', node_index, event))
                except ValueError:
                    pass
        else:
            # Regular event that needs augmentation
            event_timeline.append((start_ts, 'regular', None, None, event))

    # Sort by timestamp
    event_timeline.sort(key=lambda x: x[0])

    # Process events in chronological order with a stack
    context_stack = []  # Stack of (type, identifier, metadata)
    # type can be 'filename' or 'node'
    # identifier is the filename or node_index
    # metadata is the corresponding metadata dict

    # Invariant: all start event has a corresponding end event
    for timestamp, event_type, marker_type, identifier, event in event_timeline:
        if event_type == 'start':
            if marker_type == 'filename':
                # Push filename context - query metadata registry on-demand
                metadata = _FX_METADATA_REGISTRY.get(identifier)
                context_stack.append(('filename', identifier, metadata))
            elif marker_type == 'node':
                # Find the current filename from stack
                current_file_metadata = None
                for ctx_type, ctx_id, ctx_meta in reversed(context_stack):
                    if ctx_type == 'filename':
                        current_file_metadata = ctx_meta
                        break

                if current_file_metadata:
                    node_metadata = current_file_metadata.get("node_metadata", {})
                    if identifier in node_metadata:
                        node_meta = node_metadata[identifier]
                        context_stack.append(('node', identifier, node_meta))

        elif event_type == 'end':
            # Pop from stack - search backwards to find matching context
            for i in range(len(context_stack) - 1, -1, -1):
                ctx_type, ctx_id, ctx_meta = context_stack[i]
                if marker_type == ctx_type and identifier == ctx_id:
                    context_stack.pop(i)
                    break

        elif event_type == 'regular':
            # Apply metadata from current context stack
            # Find the most specific context (node takes precedence over filename)
            current_stack_trace = None
            current_node_name = None

            for ctx_type, ctx_id, ctx_meta in reversed(context_stack):
                if ctx_type == 'node' and ctx_meta:
                    current_stack_trace = ctx_meta.get("stack_trace", "No model stack trace available")
                    current_node_name = ctx_meta.get("name", "")
                    # Do we want to only attach the stack trace of the lowest node or stack trace of all nodes
                    # if nodes are nested, e.g. in nested graph modules
                    break

            # Augment the event
            if current_stack_trace or current_node_name:
                args = event.setdefault("args", {})
                if current_stack_trace:
                    args["stack_trace"] = current_stack_trace
                if current_node_name:
                    args["node_name"] = current_node_name

    # Remove FX marker events if requested
    if remove_fx_events:
        traced_data["traceEvents"] = [
            e for e in trace_events if e not in fx_marker_events
        ]
