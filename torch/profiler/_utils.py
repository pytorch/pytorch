# mypy: allow-untyped-defs
import functools
import operator
import re
from collections import deque
from dataclasses import dataclass
from typing import Any, Literal, Optional, TYPE_CHECKING

from torch.autograd.profiler import profile
from torch.profiler import DeviceType


if TYPE_CHECKING:
    from torch.autograd import _KinetoEvent


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


@dataclass
class EventMetrics:
    duration_time_ns: int = 0
    self_time_ns: int = 0
    idle_time_ns: int = 0
    queue_depth: int = 0

    @property
    def fraction_idle_time(self):
        if self.duration_time_ns == 0:
            return 0.0
        return self.idle_time_ns / self.duration_time_ns


@dataclass
class Interval:
    start: int
    end: int
    queue_depth: int = 0


class EventKey:
    def __init__(self, event) -> None:
        self.event = event

    def __hash__(self):
        return hash(self.event.id)

    def __eq__(self, other):
        return self.event.id == other.event.id

    def __repr__(self) -> str:
        return f"{self.event.name}"

    def intervals_overlap(self, intervals: list[Interval]):
        overlap_time = 0
        intervals = sorted(intervals, key=lambda x: x.start)

        if intervals:
            overlap_start = max(self.event.start_time_ns, intervals[0].start)
            overlap_end = min(self.event.end_time_ns, intervals[0].end)

            if overlap_start < overlap_end:
                overlap_time += overlap_end - overlap_start

        i, j = 0, 1
        while j < len(intervals):
            prev_interval = intervals[i]
            curr_interval = intervals[j]
            j += 1
            if prev_interval.end > curr_interval.start:
                # Completely subsumed by previous interval
                if prev_interval.end > curr_interval.end:
                    j += 1
                    continue
                else:
                    curr_interval.start = prev_interval.end
                    i = j

            overlap_start = max(self.event.start_time_ns, curr_interval.start)
            overlap_end = min(self.event.end_time_ns, curr_interval.end)
            if overlap_start < overlap_end:
                overlap_time += overlap_end - overlap_start

        return overlap_time


class BasicEvaluation:
    def __init__(self, prof: profile) -> None:
        self.profile = prof
        self.metrics: dict[EventKey, EventMetrics] = {}
        self.compute_self_time()
        self.event_keys = sorted(
            self.metrics.keys(), key=lambda x: x.event.start_time_ns
        )
        self.events = [e.event for e in self.event_keys]
        self.cuda_events: list[_KinetoEvent] = []
        self.queue_depth_list = self.compute_queue_depth()
        self.compute_idle_time()

    def compute_self_time(self) -> None:
        """
        Computes event's self time(total time - time in child ops).
        """
        assert self.profile.kineto_results is not None
        stack = deque(self.profile.kineto_results.experimental_event_tree())

        # standard iterating dfs
        while stack:
            curr_event = stack.pop()
            self_time = curr_event.duration_time_ns
            for child_event in curr_event.children:
                self_time -= child_event.duration_time_ns
                stack.append(child_event)
            assert EventKey(curr_event) not in self.metrics, (
                f"Duplicate id: {curr_event.id}, {curr_event.name}"
            )
            self.metrics[EventKey(curr_event)] = EventMetrics(self_time_ns=self_time)
            self.metrics[
                EventKey(curr_event)
            ].duration_time_ns = curr_event.duration_time_ns

    def compute_queue_depth(self):
        """
        Computes queue_depth at each event. This will calculate the queue depth data for
        All the events in the tree.
        This will return a list of Interval of queue depth data of cuda launch and kernels.
        """
        assert self.profile.kineto_results is not None
        cuda_event_list = self.profile.kineto_results.events()

        def is_cuda_launch_kernel(e):
            """Check if the event is a CUDA launch kernel."""
            launch_patterns = {
                "cudaLaunchKernel",  # Standard CUDA
                "cudaLaunchKernelExC",  # Extended C
                "__cudaLaunchKernel",  # Internal
                "cudaLaunchCooperativeKernel",  # Collaborative (single-device)
                "cudaLaunchCooperativeKernelMultiDevice",  # Collaborative (multi-devices)
            }
            name = str(getattr(e, "name", e))
            return any(name.startswith(pattern) for pattern in launch_patterns)

        def is_cuda_kernel(e):
            """Check if the event is a CUDA runtime kernel."""
            # Check if the kernel is CUDA
            if e.device_type() != DeviceType.CUDA:
                return False

            name = str(getattr(e, "name", e)).lower()

            # Exclude memory operations
            exclude_patterns = {"mem", "cpy", "alloc", "free"}

            return not any(pattern in name for pattern in exclude_patterns)

        cuda_launch_events = sorted(
            (e for e in cuda_event_list if is_cuda_launch_kernel(e)),
            key=lambda x: x.start_ns(),
        )
        cuda_kernel_events = sorted(
            (e for e in cuda_event_list if is_cuda_kernel(e)),
            key=lambda x: x.start_ns(),
        )

        self.cuda_events = sorted(
            cuda_launch_events + cuda_kernel_events, key=lambda x: x.start_ns()
        )

        kernel_mapping: dict[_KinetoEvent, int] = {}
        last_mapped_kernel = 0
        for cuda_launch_event in cuda_launch_events:
            index = index_of_first_match(
                cuda_kernel_events,
                lambda x: x.linked_correlation_id()
                == cuda_launch_event.linked_correlation_id(),
                start=last_mapped_kernel,
            )
            kernel_mapping[cuda_launch_event] = index
            last_mapped_kernel = index if index is not None else last_mapped_kernel

        current_kernel_index = 0
        spawned_kernel_index = -1

        all_events = cuda_launch_events + cuda_kernel_events + self.events

        def new_old_event_comparator(event):
            if hasattr(event, "start_us"):
                return event.start_us() * 1000
            if hasattr(event, "start_ns"):
                return event.start_ns()
            if hasattr(event, "start_time_ns"):
                return event.start_time_ns
            raise Exception("Unknown Event Type")  # noqa: TRY002

        queue_depth_list: list[Interval] = []
        all_events.sort(key=new_old_event_comparator)
        for event in all_events:
            # Find latest cuda kernel event
            if hasattr(event, "start_us"):
                start_time = event.start_us() * 1000
                # pyrefly: ignore [missing-attribute]
                end_time = (event.start_us() + event.duration_us()) * 1000
                # Find current spawned cuda kernel event
                if event in kernel_mapping and kernel_mapping[event] is not None:
                    spawned_kernel_index = kernel_mapping[event]
            if hasattr(event, "start_ns"):
                start_time = event.start_ns()
                end_time = event.start_ns() + event.duration_ns()
                # Find current spawned cuda kernel event
                if event in kernel_mapping and kernel_mapping[event] is not None:
                    spawned_kernel_index = kernel_mapping[event]
            elif hasattr(event, "start_time_ns"):
                start_time = event.start_time_ns  # type: ignore[attr-defined]
                end_time = event.end_time_ns  # type: ignore[attr-defined]

            while (
                current_kernel_index < len(cuda_kernel_events)
                and (cuda_kernel_events[current_kernel_index].start_ns()) <= start_time  # type: ignore[possibly-undefined]
            ):
                current_kernel_index += 1
            current_queue_depth = spawned_kernel_index - current_kernel_index + 1
            current_queue_depth = max(current_queue_depth, 0)

            if hasattr(event, "start_us") or hasattr(event, "start_ns"):
                queue_depth_list.append(
                    Interval(start_time, end_time, current_queue_depth)  # type: ignore[possibly-undefined]
                )
            elif hasattr(event, "start_time_ns"):
                self.metrics[EventKey(event)].queue_depth = current_queue_depth

        return queue_depth_list

    def compute_idle_time(self) -> None:
        """
        Computes idle time of the profile.
        """
        # Based on queue_depth_list, we can calculate idle time for all the events
        idle = False
        idle_start = 0
        idle_intervals: list[Interval] = []
        if self.queue_depth_list and self.events:
            idle_intervals += [
                Interval(self.events[0].start_time_ns, self.queue_depth_list[0].start),
                Interval(self.queue_depth_list[-1].end, self.events[-1].end_time_ns),
            ]

        for data_point in self.queue_depth_list:
            if data_point.queue_depth == 0 and not idle:
                idle_start = data_point.end
                idle = True
            if data_point.queue_depth > 0 and idle:
                idle_intervals.append(Interval(idle_start, data_point.start))
                idle = False

        event_list = [e.event for e in self.metrics]
        for event in event_list:
            self.metrics[EventKey(event)].idle_time_ns = EventKey(
                event
            ).intervals_overlap(idle_intervals)

    def rank_events(self, length):
        """
        Filter and Rank the events based on some heuristics:
        1) Events that are in the falling phase of the queue depth.
        2) Events that have a high idle_time, self_time difference.

        Parameters:
            length: The number of events to return.
        """

        # Find the interval when qd is falling to 0
        import torch

        queue_depth_list = list(reversed(self.queue_depth_list))
        qd_values = [e.queue_depth for e in queue_depth_list]

        bottom_threashold = 0
        top_threashold = 4
        decrease_interval = []
        i = 0
        while i < len(qd_values):
            if qd_values[i] > bottom_threashold:
                i += 1
                continue
            for j in range(i + 1, len(qd_values)):
                # Find next zero and if the max value between them exceeds
                # the threshold, then we have a falling interval
                next_minimum_idx = index_of_first_match(
                    qd_values, lambda x: x <= bottom_threashold, start=j
                )
                peak_idx = argmax(qd_values, start=j, end=next_minimum_idx)

                # if is a valid peak, we add to list and continue
                if peak_idx is not None and qd_values[peak_idx] >= top_threashold:
                    decrease_interval.append(
                        Interval(
                            queue_depth_list[peak_idx].start, queue_depth_list[i].start
                        )
                    )
                    i = next_minimum_idx if next_minimum_idx is not None else i
                    break
            i += 1
        # Filter out events that are not in the decrease interval
        event_list = [
            event
            for event in self.metrics
            if event.intervals_overlap(decrease_interval)
        ]
        if event_list:
            self_time = torch.tensor(
                [self.metrics[event].self_time_ns for event in event_list],
                dtype=torch.float32,
            )
            idle_time = torch.tensor(
                [self.metrics[event].fraction_idle_time for event in event_list],
                dtype=torch.float32,
            )
            normalized_gain = (idle_time - torch.mean(idle_time)) / torch.std(idle_time)
            normalized_self = (self_time - torch.mean(self_time)) / torch.std(self_time)
            heuristic_score_list = normalized_gain + 0.6 * normalized_self

            # Sort events by heuristic
            event_list = [
                event
                for _, event in sorted(
                    zip(heuristic_score_list, event_list, strict=True),
                    key=operator.itemgetter(0),
                    reverse=True,
                )
            ]
            event_list = event_list[:length]
        return event_list

    def get_optimizable_events(self, length: int = 1, print_enable: bool = True):
        event_list = self.rank_events(length)
        if not print_enable:
            return event_list
        output = "Optimizable events:\n" if event_list else "No events to optimize\n"

        output += "\n".join(
            [
                f"""{"-" * 80}
Event:                {event}
Source code location: {source_code_location(event.event)}
Percentage idle time: {self.metrics[event].fraction_idle_time * 100:.2f}%
{"-" * 80}"""
                for event in event_list
            ]
        )
        if print_enable:
            print(output)
        return event_list


def index_of_first_match(seq, predicate, start=0, end=None):
    if end is None or end >= len(seq):
        end = len(seq)
    for i in range(start, end):
        if predicate(seq[i]):
            return i
    return None


def argmax(seq, key=lambda x: x, start=0, end=None):
    seq = seq[start:end]
    if len(seq) == 0:
        return None
    return seq.index(max(seq, key=key)) + start


def source_code_location(event):
    while event is not None:
        match = re.search(r"\.py\(.*\)", event.name)
        if match is None:
            event = event.parent
            continue
        return event.name
    return "No source code location found"


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
    marker_type: Optional[Literal["filename", "node"]]
    identifier: Optional[str | int]
    event: dict[str, Any]


@dataclass
class ContextStackEntry:
    """Represents a context (filename or node) in the stack."""

    context_type: Literal["filename", "node"]
    identifier: str | int
    metadata: Optional[dict]
    tid: Optional[int] = None  # Thread ID associated with this context


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
                assert timeline_event.identifier is not None

                if timeline_event.marker_type == "filename":
                    assert isinstance(timeline_event.identifier, str)
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
                            node_meta: Optional[dict] = node_metadata[
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
