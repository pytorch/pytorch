# mypy: allow-untyped-defs
import functools
from collections import deque
from dataclasses import dataclass
from typing import Any, Literal


def _traverse(tree, next_fn, children_fn=lambda x: x.children, reverse: bool = False):
    order = reversed if reverse else lambda x: x
    remaining = deque(order(tree))
    while remaining:
        curr_event = next_fn(remaining)
        yield curr_event
        for child_event in order(children_fn(curr_event)):
            remaining.append(child_event)


traverse_dfs = functools.partial(_traverse, next_fn=lambda x: x.pop(), reverse=True)
traverse_bfs = functools.partial(
    _traverse, next_fn=lambda x: x.popleft(), reverse=False
)


def index_of_first_match(seq, predicate, start=0, end=None):
    if end is None or end >= len(seq):
        end = len(seq)
    for i in range(start, end):
        if predicate(seq[i]):
            return i
    return None


# Provide an OSS workaround for cudagraphs + CUPTI issue
# https://github.com/pytorch/pytorch/issues/75504
# TODO(dberard) - deprecate / remove workaround for CUDA >= 12, when
# we stop supporting older CUDA versions.
def _init_for_cuda_graphs() -> None:
    from torch.autograd.profiler import profile

    with profile():
        pass


@dataclass
class TimelineEvent:
    """Represents an event in the profiler timeline."""

    timestamp: int
    event_type: Literal["start", "end", "regular"]
    marker_type: Literal["filename", "node"] | None
    identifier: str | int | None
    event: dict[str, Any]


@dataclass
class ContextStackEntry:
    """Represents a context (filename or node) in the stack."""

    context_type: Literal["filename", "node"]
    identifier: str | int
    metadata: dict | None
    tid: int | None = None  # Thread ID associated with this context


def map_recorded_events_to_aten_ops_with_stack_trace(traced_data):
    """
    Maps recorded profiler events to their corresponding fx nodes and adds stack traces.

    Builds a timeline of all events (regular ops and FX markers for filenames/nodes),
    sorts by timestamp, then processes chronologically while maintaining a context stack of active
    filename/node scopes. Regular events are augmented with stack traces and node names from the
    innermost active context. Runtime is O(n log n) for n events.

    Args:
        traced_data: Json of profiler events from Chrome trace

    Returns:
        Dict mapping recorded event names to their aten operations with added stack traces
    """
    from torch.fx.traceback import _FX_METADATA_REGISTRY

    trace_events = traced_data.get("traceEvents", [])

    # Create event timeline
    event_timeline: list[TimelineEvent] = []

    def is_fx_marker_event(event):
        return (
            event.get("cat") == "cpu_op"
            and event.get("name", "").startswith("## ")
            and event.get("name", "").endswith(" ##")
        )

    def append_fx_marker_event(event_type, identifier, event):
        start_ts = event["ts"]
        end_ts = start_ts + event["dur"]
        event_timeline.append(
            TimelineEvent(start_ts, "start", event_type, identifier, event)
        )
        event_timeline.append(
            TimelineEvent(end_ts, "end", event_type, identifier, event)
        )

    for event in trace_events:
        if "ts" not in event or "dur" not in event:
            continue

        if is_fx_marker_event(event):
            content = event["name"][3:-3]

            if content.endswith(".py"):
                append_fx_marker_event("filename", content, event)
            else:
                try:
                    node_index = int(content)
                except ValueError:
                    pass
                append_fx_marker_event("node", node_index, event)  # type: ignore[possibly-undefined]

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
                if timeline_event.identifier is None:
                    raise AssertionError("identifier must not be None for start event")

                if timeline_event.marker_type == "filename":
                    if not isinstance(timeline_event.identifier, str):
                        raise AssertionError(
                            f"identifier must be str for filename marker, "
                            f"got {type(timeline_event.identifier).__name__}"
                        )
                    # Push filename context - query metadata registry on-demand
                    metadata = _FX_METADATA_REGISTRY.get(timeline_event.identifier)
                    tid = timeline_event.event.get("tid")
                    context_stack.append(
                        ContextStackEntry(
                            "filename", timeline_event.identifier, metadata, tid
                        )
                    )
                elif timeline_event.marker_type == "node":
                    # Find the current filename from stack
                    current_file_metadata = None
                    tid = timeline_event.event.get("tid")
                    for ctx_entry in reversed(context_stack):
                        if (
                            ctx_entry.context_type == "filename"
                            and ctx_entry.tid == tid
                        ):
                            current_file_metadata = ctx_entry.metadata
                            break

                    if current_file_metadata:
                        node_metadata = current_file_metadata.get("node_metadata", {})
                        if timeline_event.identifier in node_metadata:
                            node_meta: dict | None = node_metadata[
                                timeline_event.identifier
                            ]
                            context_stack.append(
                                ContextStackEntry(
                                    "node", timeline_event.identifier, node_meta, tid
                                )
                            )

            case "end":
                # Pop from stack - search backwards to find matching context
                for i in range(len(context_stack) - 1, -1, -1):
                    ctx_entry = context_stack[i]
                    if (
                        timeline_event.marker_type == ctx_entry.context_type
                        and timeline_event.identifier == ctx_entry.identifier
                    ):
                        context_stack.pop(i)
                        break

            case "regular":
                # Apply metadata from current context stack
                # Find the most specific context (node takes precedence over filename)
                # Only augment events with the same tid as the file/node event matched
                current_stack_trace = None
                current_node_name = None
                event_tid = timeline_event.event.get("tid")

                for ctx_entry in reversed(context_stack):
                    # Only apply metadata from contexts with matching tid
                    if ctx_entry.tid == event_tid:
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
