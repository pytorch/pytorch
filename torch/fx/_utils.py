# mypy: allow-untyped-defs
import sys
from dataclasses import dataclass
from typing import Any, Literal, Optional

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


@dataclass
class TimelineEvent:
    """Represents an event in the profiler timeline."""
    timestamp: int
    event_type: Literal["start", "end", "regular"]
    marker_type: Optional[Literal["filename", "node"]]
    identifier: Optional[str | int]
    event: dict[str, Any]


@dataclass
class ContextStackEntry:
    """Represents a context (filename or node) in the stack."""
    context_type: Literal["filename", "node"]
    identifier: str | int
    metadata: Optional[dict]


def map_recorded_events_to_aten_ops_with_stack_trace(
    traced_data, remove_fx_events=False
):
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

    # Create event timeline
    event_timeline: list[TimelineEvent] = []
    fx_marker_events = []  # Track FX marker events to optionally remove them

    def is_fx_marker_event(event):
        return (
            event.get("cat") == "cpu_op"
            and event.get("name", "").startswith("## ")
            and event.get("name", "").endswith(" ##")
        )

    def append_fx_marker_event(event_type, identifier, event):
        start_ts = event["ts"]
        end_ts = start_ts + event["dur"]
        event_timeline.append(TimelineEvent(start_ts, "start", event_type, identifier, event))
        event_timeline.append(TimelineEvent(end_ts, "end", event_type, identifier, event))

    for event in trace_events:
        if "ts" not in event or "dur" not in event:
            continue

        if is_fx_marker_event(event):
            fx_marker_events.append(event)
            content = event["name"][3:-3]

            if content.endswith(".py"):
                append_fx_marker_event("filename", content, event)
            else:
                try:
                    node_index = int(content)
                except ValueError:
                    pass
                append_fx_marker_event("node", node_index, event)

        else:
            # Regular event that needs augmentation
            start_ts = event["ts"]
            event_timeline.append(TimelineEvent(start_ts, "regular", None, None, event))

    # Sort by timestamp
    event_timeline.sort(key=lambda x: x.timestamp)

    # Process events in chronological order with a stack
    context_stack: list[ContextStackEntry] = []

    # Invariant: all start event has a corresponding end event
    for timeline_event in event_timeline:
        match timeline_event.event_type:
            case "start":
                assert timeline_event.identifier is not None

                if timeline_event.marker_type == "filename":
                    assert isinstance(timeline_event.identifier, str)
                    # Push filename context - query metadata registry on-demand
                    metadata = _FX_METADATA_REGISTRY.get(timeline_event.identifier)
                    context_stack.append(ContextStackEntry("filename", timeline_event.identifier, metadata))
                elif timeline_event.marker_type == "node":
                    # Find the current filename from stack
                    current_file_metadata = None
                    for ctx_entry in reversed(context_stack):
                        if ctx_entry.context_type == "filename":
                            current_file_metadata = ctx_entry.metadata
                            break

                    if current_file_metadata:
                        node_metadata = current_file_metadata.get("node_metadata", {})
                        if timeline_event.identifier in node_metadata:
                            node_meta: Optional[dict] = node_metadata[timeline_event.identifier]
                            context_stack.append(ContextStackEntry("node", timeline_event.identifier, node_meta))

            case "end":
                # Pop from stack - search backwards to find matching context
                for i in range(len(context_stack) - 1, -1, -1):
                    ctx_entry = context_stack[i]
                    if timeline_event.marker_type == ctx_entry.context_type and timeline_event.identifier == ctx_entry.identifier:
                        context_stack.pop(i)
                        break

            case "regular":
                # Apply metadata from current context stack
                # Find the most specific context (node takes precedence over filename)
                current_stack_trace = None
                current_node_name = None

                for ctx_entry in reversed(context_stack):
                    if ctx_entry.context_type == "node" and ctx_entry.metadata:
                        current_stack_trace = ctx_entry.metadata.get(
                            "stack_trace", "No model stack trace available"
                        )
                        current_node_name = ctx_entry.metadata.get("name", "")
                        # Do we want to only attach the stack trace of the lowest node or stack trace of all nodes
                        # if nodes are nested, e.g. in nested graph modules
                        break

                # Augment the event
                if current_stack_trace or current_node_name:
                    args = timeline_event.event.setdefault("args", {})
                    if current_stack_trace:
                        args["stack_trace"] = current_stack_trace
                    if current_node_name:
                        args["node_name"] = current_node_name

    # Remove FX marker events if requested
    if remove_fx_events:
        traced_data["traceEvents"] = [
            e for e in trace_events if e not in fx_marker_events
        ]
