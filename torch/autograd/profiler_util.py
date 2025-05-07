# mypy: allow-untyped-defs
import bisect
import itertools
import math
from collections import defaultdict, namedtuple
from operator import attrgetter
from typing import Any, Optional
from typing_extensions import deprecated

import torch
from torch.autograd import DeviceType


__all__ = [
    "EventList",
    "FormattedTimesMixin",
    "Interval",
    "Kernel",
    "FunctionEvent",
    "FunctionEventAvg",
    "StringTable",
    "MemRecordsAcc",
]


class EventList(list):
    """A list of Events (for pretty printing)."""

    def __init__(self, *args, **kwargs):
        use_device = kwargs.pop("use_device", None)
        profile_memory = kwargs.pop("profile_memory", False)
        with_flops = kwargs.pop("with_flops", False)
        super().__init__(*args, **kwargs)
        self._use_device = use_device
        self._profile_memory = profile_memory
        self._tree_built = False
        self._with_flops = with_flops

    def _build_tree(self):
        self._populate_cpu_children()
        self._remove_dup_nodes()
        self._set_backward_stacktraces()
        self._tree_built = True

    def __str__(self):
        return self.table()

    def _remove_dup_nodes(self):
        while True:
            to_delete = set()
            for idx in range(len(self)):
                if (
                    self[idx].cpu_parent is not None
                    and self[idx].cpu_parent.name == self[idx].name
                    and len(self[idx].cpu_parent.cpu_children) == 1
                ):
                    self[idx].cpu_parent.cpu_children = self[idx].cpu_children
                    self[idx].cpu_parent.kernels = self[idx].kernels  # lift kernels up
                    for ch in self[idx].cpu_children:
                        ch.cpu_parent = self[idx].cpu_parent
                    to_delete.add(idx)
            if len(to_delete) == 0:
                break
            new_evts = [ev for ind, ev in enumerate(self) if ind not in to_delete]
            self.clear()
            self.extend(new_evts)

    def _populate_cpu_children(self):
        """Populate child events into each underlying FunctionEvent object.

        One event is a child of another if [s1, e1) is inside [s2, e2). Where
        s1 and e1 would be start and end of the child event's interval. And
        s2 and e2 start and end of the parent event's interval

        Example: In event list [[0, 10], [1, 3], [3, 4]] would have make [0, 10]
        be a parent of two other intervals.

        If for any reason two intervals intersect only partially, this function
        will not record a parent child relationship between then.
        """
        # Some events can be async (i.e. start and end on different threads),
        # since it's generally undefined how to attribute children ranges to
        # async ranges, we do not use them when calculating nested ranges and stats
        sync_events = [
            evt
            for evt in self
            if not evt.is_async and evt.device_type == DeviceType.CPU
        ]
        events = sorted(
            sync_events,
            key=attrgetter("thread"),
        )
        # Group by both thread and node_id, so that events that happen to have
        # the same thread_id but are from different nodes aren't incorrectly
        # grouped together.
        threads = itertools.groupby(
            events, key=lambda event: (event.thread, event.node_id)
        )

        # For each thread we keep a stack of current nested parents.
        # We maintain the invariant that each interval is a subset of all other
        # intervals lower in the stack.
        #
        # First we sort the intervals by their start time. Then we iterate over them.
        # Every time we see a new interval we remove several parents from
        # the top until we restore the invariant. Then parent child relationship
        # if recorded if the stack is not empty.
        # Finally we add new interval to the list
        #
        # Algorithm has O(N * log(N)) complexity where N is number of
        # intervals
        for _thread_id, thread_events in threads:
            thread_events_ = sorted(
                thread_events,
                key=lambda event: [event.time_range.start, -event.time_range.end],
            )
            current_events: list[FunctionEvent] = []
            for event in thread_events_:
                while len(current_events) > 0:
                    parent = current_events[-1]
                    if (
                        event.time_range.start >= parent.time_range.end
                        or event.time_range.end > parent.time_range.end
                    ):
                        # this can't be a parent
                        current_events.pop()
                    else:
                        parent.append_cpu_child(event)
                        assert (
                            event.cpu_parent is None
                        ), f"There is already a CPU parent event for {event.key}"
                        event.set_cpu_parent(parent)
                        break

                current_events.append(event)

    def _set_backward_stacktraces(self):
        def bw_parent(evt):
            if evt is None:
                return None
            elif evt.scope == 1:  # BACKWARD_FUNCTION
                return evt
            else:
                return bw_parent(evt.cpu_parent)

        fwd_stacks = {}
        for evt in self:
            if bw_parent(evt) is None and evt.stack is not None:
                t = (evt.sequence_nr, evt.thread)
                if t not in fwd_stacks:
                    fwd_stacks[t] = evt.stack

        for evt in self:
            p = bw_parent(evt)
            if p is not None:
                assert p.fwd_thread is not None
                t = (p.sequence_nr, p.fwd_thread)
                if t in fwd_stacks:
                    evt.stack = fwd_stacks[t]
                else:
                    evt.stack = []

    @property
    def self_cpu_time_total(self):
        return sum(event.self_cpu_time_total for event in self)

    def table(
        self,
        sort_by=None,
        row_limit=100,
        max_src_column_width=75,
        max_name_column_width=55,
        max_shapes_column_width=80,
        header=None,
        top_level_events_only=False,
    ):
        """Print an EventList as a nicely formatted table.

        Args:
            sort_by (str, optional): Attribute used to sort entries. By default
                they are printed in the same order as they were registered.
                Valid keys include: ``cpu_time``, ``cuda_time``, ``xpu_time``,
                ``cpu_time_total``, ``cuda_time_total``, ``xpu_time_total``,
                ``cpu_memory_usage``, ``cuda_memory_usage``, ``xpu_memory_usage``,
                ``self_cpu_memory_usage``, ``self_cuda_memory_usage``,
                ``self_xpu_memory_usage``, ``count``.
            top_level_events_only(bool, optional): Boolean flag to determine the
                selection of events to display. If true, the profiler will only
                display events at top level like top-level invocation of python
                `lstm`, python `add` or other functions, nested events like low-level
                cpu/cuda/xpu ops events are omitted for profiler result readability.

        Returns:
            A string containing the table.
        """
        return _build_table(
            self,
            sort_by=sort_by,
            row_limit=row_limit,
            max_src_column_width=max_src_column_width,
            max_name_column_width=max_name_column_width,
            max_shapes_column_width=max_shapes_column_width,
            header=header,
            profile_memory=self._profile_memory,
            with_flops=self._with_flops,
            top_level_events_only=top_level_events_only,
        )

    def export_chrome_trace(self, path):
        """Export an EventList as a Chrome tracing tools file.

        The checkpoint can be later loaded and inspected under ``chrome://tracing`` URL.

        Args:
            path (str): Path where the trace will be written.
        """
        import os

        device_name = "cuda" if not self._use_device else self._use_device
        with open(path, "w") as f:
            next_id = 0
            # Use file IO over using json.dump since JSON dumping is very slow and
            # this technique is proven to give a 4x speedup.
            f.write("[")
            for evt in self:
                if evt.trace_name is None:
                    continue
                f.write(
                    '{{"name": "{}", '
                    '"ph": "X", '
                    '"ts": {}, '
                    '"dur": {}, '
                    '"tid": {}, '
                    '"pid": "CPU functions", '
                    '"args": {{}}}}, '.format(
                        evt.trace_name,
                        evt.time_range.start,
                        evt.time_range.elapsed_us(),
                        evt.thread
                        if not evt.is_remote
                        else f'" node_id:{evt.node_id}, thread_id:{evt.thread} "',
                    )
                )
                for _ in evt.kernels:
                    # 's' and 'f' draw Flow arrows from
                    # the CPU launch to the GPU kernel
                    f.write(
                        f'{{"name": "{evt.trace_name}", '
                        '"ph": "s", '
                        f'"ts": {evt.time_range.start}, '
                        f'"tid": {evt.thread}, '
                        '"pid": "CPU functions", '
                        f'"id": {next_id}, '
                        f'"cat": "cpu_to_{device_name}", '
                        '"args": {}}, '
                    )
                    # Note: use torch.profiler to get device kernel trace
                    next_id += 1
            if len(self) > 0:
                # remove trailing whitespace and comma
                f.seek(f.tell() - 2, os.SEEK_SET)
                f.truncate()
            f.write("]")

    def supported_export_stacks_metrics(self):
        return [
            "self_cpu_time_total",
            "self_cuda_time_total",
            "self_xpu_time_total",
            "self_privateuse1_time_total",
        ]

    def export_stacks(self, path: str, metric: str):
        if metric not in self.supported_export_stacks_metrics():
            raise ValueError(
                "metric should be one of: "
                + str(self.supported_export_stacks_metrics())
            )
        translate_table = str.maketrans(" ;\t\n", "____")
        with open(path, "w") as f:
            for evt in self:
                if evt.stack and len(evt.stack) > 0:
                    metric_value = getattr(
                        evt,
                        metric.replace("cuda", "device")
                        .replace("xpu", "device")
                        .replace("privateuse1", "device"),
                    )
                    if int(metric_value) > 0:
                        stack_str = ""
                        for entry in reversed(evt.stack):
                            stack_str += entry.translate(translate_table)
                            stack_str += ";"
                        stack_str = stack_str[:-1] + " " + str(int(metric_value))
                        f.write(stack_str + "\n")

    def key_averages(
        self,
        group_by_input_shapes=False,
        group_by_stack_n=0,
        group_by_overload_name=False,
    ):
        """Averages all function events over their keys.

        Args:
            group_by_input_shapes: group entries by
                (event name, input shapes) rather than just event name.
                This is useful to see which input shapes contribute to the runtime
                the most and may help with size-specific optimizations or
                choosing the best candidates for quantization (aka fitting a roof line)

            group_by_stack_n: group by top n stack trace entries

            group_by_overload_name: Differentiate operators by their overload name e.g. aten::add.Tensor
            and aten::add.out will be aggregated separately

        Returns:
            An EventList containing FunctionEventAvg objects.
        """
        assert self._tree_built
        stats: dict[tuple[str, ...], FunctionEventAvg] = defaultdict(FunctionEventAvg)

        def get_key(
            event, group_by_input_shapes, group_by_stack_n, group_by_overload_name
        ) -> tuple[str, ...]:
            key = [
                str(event.key),
                str(event.node_id),
                str(event.device_type),
                str(event.is_legacy),
                str(event.is_user_annotation),
            ]
            if group_by_overload_name:
                key.append(evt.overload_name)
            if group_by_input_shapes:
                key.append(str(event.input_shapes))
            if group_by_stack_n > 0:
                key += event.stack[:group_by_stack_n]
            return tuple(key)

        for evt in self:
            stats[
                get_key(
                    evt, group_by_input_shapes, group_by_stack_n, group_by_overload_name
                )
            ].add(evt)

        avg_list = EventList(
            stats.values(),
            use_device=self._use_device,
            profile_memory=self._profile_memory,
            with_flops=self._with_flops,
        )
        for evt in avg_list:
            evt.stack = evt.stack[:group_by_stack_n]
            if not group_by_input_shapes:
                evt.input_shapes = ""
            if not group_by_overload_name:
                evt.overload_name = ""
        return avg_list

    def total_average(self):
        """Averages all events.

        Returns:
            A FunctionEventAvg object.
        """
        total_stat = FunctionEventAvg()
        for evt in self:
            total_stat += evt
            total_stat.key = None
        total_stat.key = "Total"
        return total_stat


def _format_time(time_us):
    """Define how to format time in FunctionEvent."""
    US_IN_SECOND = 1000.0 * 1000.0
    US_IN_MS = 1000.0
    if time_us >= US_IN_SECOND:
        return f"{time_us / US_IN_SECOND:.3f}s"
    if time_us >= US_IN_MS:
        return f"{time_us / US_IN_MS:.3f}ms"
    return f"{time_us:.3f}us"


def _format_time_share(time_us, total_time_us):
    """Define how to format time in FunctionEvent."""
    if total_time_us == 0:
        assert time_us == 0, f"Expected time_us == 0 but got {time_us}"
        return "NaN"
    return f"{time_us * 100.0 / total_time_us:.2f}%"


def _format_memory(nbytes):
    """Return a formatted memory size string."""
    KB = 1024
    MB = 1024 * KB
    GB = 1024 * MB
    if abs(nbytes) >= GB:
        return f"{nbytes * 1.0 / GB:.2f} Gb"
    elif abs(nbytes) >= MB:
        return f"{nbytes * 1.0 / MB:.2f} Mb"
    elif abs(nbytes) >= KB:
        return f"{nbytes * 1.0 / KB:.2f} Kb"
    else:
        return str(nbytes) + " b"


def _attr_formatter(name):
    return property(lambda self: _format_time(getattr(self, name)))


class FormattedTimesMixin:
    """Helpers for FunctionEvent and FunctionEventAvg.

    The subclass should define `*_time_total` and `count` attributes.
    """

    cpu_time_str = _attr_formatter("cpu_time")
    device_time_str = _attr_formatter("device_time")
    cpu_time_total_str = _attr_formatter("cpu_time_total")
    device_time_total_str = _attr_formatter("device_time_total")
    self_cpu_time_total_str = _attr_formatter("self_cpu_time_total")
    self_device_time_total_str = _attr_formatter("self_device_time_total")

    @property
    def cpu_time(self):
        return 0.0 if self.count == 0 else 1.0 * self.cpu_time_total / self.count  # type: ignore[attr-defined]

    @property
    def device_time(self):
        return 0.0 if self.count == 0 else 1.0 * self.device_time_total / self.count  # type: ignore[attr-defined]

    @property
    @deprecated(
        "`cuda_time` is deprecated, please use `device_time` instead.",
        category=FutureWarning,
    )
    def cuda_time(self):  # To be deprecated
        return self.device_time


class Interval:
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def elapsed_us(self):
        r"""
        Returns the length of the interval
        """
        return self.end - self.start


Kernel = namedtuple("Kernel", ["name", "device", "duration"])


class FunctionEvent(FormattedTimesMixin):
    """Profiling information about a single function."""

    def __init__(
        self,
        id,
        name,
        thread,
        start_us,
        end_us,
        overload_name=None,
        fwd_thread=None,
        input_shapes=None,
        stack=None,
        scope=0,
        use_device=None,
        cpu_memory_usage=0,
        device_memory_usage=0,
        is_async=False,
        is_remote=False,
        sequence_nr=-1,
        node_id=-1,
        device_type=DeviceType.CPU,
        device_index=0,
        device_resource_id=None,
        is_legacy=False,
        flops=None,
        trace_name=None,
        concrete_inputs=None,
        kwinputs=None,
        is_user_annotation=False,
    ):
        self.id: int = id
        self.node_id: int = node_id
        self.name: str = name
        self.overload_name: str = overload_name
        self.trace_name: str = trace_name
        self.time_range: Interval = Interval(start_us, end_us)
        self.thread: int = thread
        self.fwd_thread: Optional[int] = fwd_thread
        self.kernels: list[Kernel] = []
        self.count: int = 1
        self.cpu_children: list[FunctionEvent] = []
        self.cpu_parent: Optional[FunctionEvent] = None
        self.input_shapes: tuple[int, ...] = input_shapes
        self.concrete_inputs: list[Any] = concrete_inputs
        self.kwinputs: dict[str, Any] = kwinputs
        self.stack: list = stack
        self.scope: int = scope
        self.use_device: Optional[str] = use_device
        self.cpu_memory_usage: int = cpu_memory_usage
        self.device_memory_usage: int = device_memory_usage
        self.is_async: bool = is_async
        self.is_remote: bool = is_remote
        self.sequence_nr: int = sequence_nr
        self.device_type: DeviceType = device_type
        self.device_index: int = device_index
        self.device_resource_id: int = (
            thread if device_resource_id is None else device_resource_id
        )
        self.is_legacy: bool = is_legacy
        self.flops: Optional[int] = flops
        self.is_user_annotation: Optional[bool] = is_user_annotation
        self.self_cpu_percent = -1
        self.total_cpu_percent = -1
        self.total_device_percent = -1

    def append_kernel(self, name, device, duration):
        assert self.device_type == DeviceType.CPU
        self.kernels.append(Kernel(name, device, duration))

    def append_cpu_child(self, child):
        """Append a CPU child of type FunctionEvent.

        One is supposed to append only direct children to the event to have
        correct self cpu time being reported.
        """
        assert self.device_type == DeviceType.CPU
        assert isinstance(child, FunctionEvent)
        assert child.device_type == DeviceType.CPU
        self.cpu_children.append(child)

    def set_cpu_parent(self, parent):
        """Set the immediate CPU parent of type FunctionEvent.

        One profiling FunctionEvent should have only one CPU parent such that
        the child's range interval is completely inside the parent's. We use
        this connection to determine the event is from top-level op or not.
        """
        assert self.device_type == DeviceType.CPU
        assert isinstance(parent, FunctionEvent)
        assert parent.device_type == DeviceType.CPU
        self.cpu_parent = parent

    # Note: async events don't have children, are not used when computing 'self'
    # metrics of other events, have only total cpu time
    @property
    def self_cpu_memory_usage(self):
        if self.is_async or self.device_type != DeviceType.CPU:
            return 0
        return self.cpu_memory_usage - sum(
            child.cpu_memory_usage for child in self.cpu_children
        )

    @property
    def self_device_memory_usage(self):
        if self.is_async or self.device_type != DeviceType.CPU:
            return 0
        return self.device_memory_usage - sum(
            child.device_memory_usage for child in self.cpu_children
        )

    @property
    @deprecated(
        "`self_cuda_memory_usage` is deprecated. Use `self_device_memory_usage` instead.",
        category=FutureWarning,
    )
    def self_cuda_memory_usage(self):  # To be deprecated
        return self.self_device_memory_usage

    @property
    def cpu_time_total(self):
        if self.device_type == DeviceType.CPU:
            return self.time_range.elapsed_us()
        else:
            return 0

    @property
    def self_cpu_time_total(self):
        if self.is_async or self.device_type != DeviceType.CPU:
            return 0
        return self.cpu_time_total - sum(
            child.cpu_time_total for child in self.cpu_children
        )

    @property
    def device_time_total(self):
        if self.is_async or not self.use_device:
            return 0
        if self.device_type == DeviceType.CPU:
            if not self.is_legacy:
                # account for the kernels in the children ops
                return sum(kinfo.duration for kinfo in self.kernels) + sum(
                    ch.device_time_total for ch in self.cpu_children
                )
            else:
                # each legacy cpu events has a single (fake) kernel
                return sum(kinfo.duration for kinfo in self.kernels)
        else:
            assert self.device_type in [
                DeviceType.CUDA,
                DeviceType.PrivateUse1,
                DeviceType.MTIA,
                DeviceType.HPU,
            ]
            return self.time_range.elapsed_us()

    @property
    @deprecated(
        "`cuda_time_total` is deprecated. Use `device_time_total` instead.",
        category=FutureWarning,
    )
    def cuda_time_total(self):  # To be deprecated
        return self.device_time_total

    @property
    def self_device_time_total(self):
        if self.is_async or not self.use_device:
            return 0
        if self.device_type == DeviceType.CPU:
            return self.device_time_total - sum(
                child.device_time_total for child in self.cpu_children
            )
        else:
            assert self.device_type in [
                DeviceType.CUDA,
                DeviceType.PrivateUse1,
                DeviceType.MTIA,
                DeviceType.HPU,
            ]
            return self.device_time_total

    @property
    @deprecated(
        "`self_cuda_time_total` is deprecated. Use `self_device_time_total` instead.",
        category=FutureWarning,
    )
    def self_cuda_time_total(self):  # To be deprecated
        return self.self_device_time_total

    @property
    def key(self):
        return self.name

    def __repr__(self):
        device_name = self.use_device
        device_time = self.device_time_str
        device_memory_usage = self.device_memory_usage
        return (
            f"<FunctionEvent id={self.id} name={self.name} overload_name={self.overload_name} "
            f"device_type={self.device_type} node_id={self.node_id} cpu_time={self.cpu_time_str} "
            f"start_us={self.time_range.start} end_us={self.time_range.end} "
            f"cpu_children={str([child.id for child in self.cpu_children])} {device_name}_time={device_time} "
            f"name={self.name} thread={self.thread} input_shapes={str(self.input_shapes)} "
            f"cpu_memory_usage={self.cpu_memory_usage} {device_name}_memory_usage={device_memory_usage} "
            f"is_async={self.is_async} is_remote={self.is_remote} seq_nr={self.sequence_nr} is_legacy={self.is_legacy}>"
        )


class FunctionEventAvg(FormattedTimesMixin):
    """Used to average stats over multiple FunctionEvent objects."""

    def __init__(self) -> None:
        self.key: Optional[str] = None
        self.count: int = 0
        self.node_id: int = 0
        self.is_async: bool = False
        self.is_remote: bool = False
        self.use_device: Optional[str] = None
        self.cpu_time_total: int = 0
        self.device_time_total: int = 0
        self.self_cpu_time_total: int = 0
        self.self_device_time_total: int = 0
        self.input_shapes: Optional[list[list[int]]] = None
        self.overload_name: Optional[str] = None
        self.stack: Optional[list] = None
        self.scope: Optional[int] = None
        self.cpu_memory_usage: int = 0
        self.device_memory_usage: int = 0
        self.self_cpu_memory_usage: int = 0
        self.self_device_memory_usage: int = 0
        self.cpu_children: Optional[list[FunctionEvent]] = None
        self.cpu_parent: Optional[FunctionEvent] = None
        self.device_type: DeviceType = DeviceType.CPU
        self.is_legacy: bool = False
        self.flops: int = 0

    def add(self, other):
        if self.key is None:
            # First function being recorded as part of FunctionEventAvg, propagate
            # fields.
            self.key = other.key
            self.node_id = other.node_id
            self.is_async = other.is_async
            self.is_remote = other.is_remote
            self.cpu_parent = other.cpu_parent
            self.cpu_children = other.cpu_children

            self.overload_name = other.overload_name
            self.input_shapes = other.input_shapes
            self.stack = other.stack
            self.scope = other.scope
            self.device_type = other.device_type
            self.is_legacy = other.is_legacy
            self.use_device = other.use_device
            self.is_user_annotation = other.is_user_annotation

        assert isinstance(other, (FunctionEvent, FunctionEventAvg))
        assert other.key == self.key

        self.cpu_time_total += other.cpu_time_total
        self.device_time_total += other.device_time_total
        self.self_cpu_time_total += other.self_cpu_time_total
        self.self_device_time_total += other.self_device_time_total
        self.cpu_memory_usage += other.cpu_memory_usage
        self.device_memory_usage += other.device_memory_usage
        self.self_cpu_memory_usage += other.self_cpu_memory_usage
        self.self_device_memory_usage += other.self_device_memory_usage
        self.count += other.count
        if self.flops is None:
            self.flops = other.flops
        elif other.flops is not None:
            self.flops += other.flops
        return self

    def __iadd__(self, other):
        return self.add(other)

    def __repr__(self):
        device_name = "cuda" if not self.use_device else self.use_device
        self_device_time = self.self_device_time_total_str
        device_time = self.device_time_str
        device_memory = self.device_memory_usage
        return (
            f"<FunctionEventAvg key={self.key} self_cpu_time={self.self_cpu_time_total_str} cpu_time={self.cpu_time_str} "
            f" self_{device_name}_time={self_device_time} {device_name}_time={device_time} input_shapes={str(self.input_shapes)} "
            f"cpu_memory_usage={self.cpu_memory_usage} {device_name}_memory_usage={device_memory}>"
        )


class StringTable(defaultdict):
    def __missing__(self, key):
        # manage cases like 't' (demangled to 'unsigned short') separately,
        # for now simply check the length to avoid unexpected results for
        # the short sequences
        self[key] = torch._C._demangle(key) if len(key) > 1 else key
        return self[key]


class MemRecordsAcc:
    """Acceleration structure for accessing mem_records in interval."""

    def __init__(self, mem_records):
        self._mem_records = mem_records
        self._start_nses: list[int] = []
        self._indices: list[int] = []
        if len(mem_records) > 0:
            tmp = sorted([(r[0].start_ns(), i) for i, r in enumerate(mem_records)])
            self._start_nses, self._indices = zip(*tmp)  # type: ignore[assignment]

    def in_interval(self, start_us, end_us):
        r"""
        Return all records in the given interval
        To maintain backward compatibility, convert us to ns in function
        """
        start_idx = bisect.bisect_left(self._start_nses, start_us * 1000)
        end_idx = bisect.bisect_right(self._start_nses, end_us * 1000)
        for i in range(start_idx, end_idx):
            yield self._mem_records[self._indices[i]]


def _filter_stack_entry(entry):
    filtered_entries = [
        ("autograd/__init__", "_make_grads"),
        ("autograd/__init__", "backward"),
        ("torch/tensor", "backward"),
        ("_internal/common_utils", "prof_callable"),
        ("_internal/common_utils", "prof_func_call"),
        ("_internal/common_utils", "prof_meth_call"),
    ]
    return all(not (f[0] in entry and f[1] in entry) for f in filtered_entries)


MEMORY_EVENT_NAME = "[memory]"
OUT_OF_MEMORY_EVENT_NAME = "[OutOfMemory]"


def _filter_name(name):
    # ignoring the following utility ops
    filtered_out_names = [
        MEMORY_EVENT_NAME,  # used only for the top-level memory events
        OUT_OF_MEMORY_EVENT_NAME,
        "profiler::_record_function_enter",
        "profiler::_record_function_enter_new",
        "profiler::_record_function_exit",
        "aten::is_leaf",
        "aten::output_nr",
        "aten::_version",
    ]
    return name in filtered_out_names


# Demangles and optionally rewrites the provided event name,
# with_wildcard - whether to replace certain numbered event names
# with a wildcard name to aggregate them together in the profiler table
# output
def _rewrite_name(name, with_wildcard=False):
    string_table = StringTable()
    name = string_table[name]
    if with_wildcard:
        if name.startswith("ProfilerStep#"):
            name = "ProfilerStep*"
    return name


def _build_table(
    events,
    sort_by=None,
    header=None,
    row_limit=100,
    max_src_column_width=75,
    max_name_column_width=55,
    max_shapes_column_width=80,
    with_flops=False,
    profile_memory=False,
    top_level_events_only=False,
):
    """Print a summary of events (which can be a list of FunctionEvent or FunctionEventAvg)."""
    if len(events) == 0:
        return ""

    has_device_time = any(event.self_device_time_total > 0 for event in events)
    has_device_mem = any(event.self_device_memory_usage > 0 for event in events)
    use_device = events[0].use_device
    # Running on PrivateUse1 device with profiler but not enable
    # ProfilerActivity.PrivateUse1 can also catch privateuse1 memory usage.
    # Here only need to check has_privateuse1_time if not use_device.
    if not use_device and has_device_time:
        raise RuntimeError("use_device is None, but there is device performance data.")

    has_input_shapes = any(
        (event.input_shapes is not None and len(event.input_shapes) > 0)
        for event in events
    )

    has_overload_names = any(
        (event.overload_name is not None and len(event.overload_name) > 0)
        for event in events
    )

    if sort_by is not None:
        events = EventList(
            sorted(
                events,
                key=lambda evt: getattr(
                    evt,
                    sort_by.replace("cuda", "device")
                    .replace("xpu", "device")
                    .replace("privateuse1", "device"),
                ),
                reverse=True,
            ),
            use_device=use_device,
            profile_memory=profile_memory,
            with_flops=with_flops,
        )

    name_column_width = max(len(evt.key) for evt in events) + 4
    if max_name_column_width is not None:
        name_column_width = min(name_column_width, max_name_column_width)

    shapes_column_width = max(len(str(evt.input_shapes)) for evt in events) + 4
    if max_shapes_column_width is not None:
        shapes_column_width = min(shapes_column_width, max_shapes_column_width)

    DEFAULT_COLUMN_WIDTH = 12
    flops_column_width = DEFAULT_COLUMN_WIDTH

    src_column_width = None
    stacks = [
        evt.stack for evt in events if evt.stack is not None and len(evt.stack) > 0
    ]
    has_stack = len(stacks) > 0
    if has_stack:
        src_column_width = (
            max(max(len(entry) for entry in stack) for stack in stacks) + 4
        )
        if max_src_column_width is not None:
            src_column_width = min(src_column_width, max_src_column_width)

    headers = ["Name"]
    if has_overload_names:
        headers.append("Overload Name")
    headers += [
        "Self CPU %",
        "Self CPU",
        "CPU total %",
        "CPU total",
        "CPU time avg",
    ]

    device_name = use_device.upper() if use_device is not None else "None"
    if has_device_time:
        headers.extend(
            [
                f"Self {device_name}",
                f"Self {device_name} %",
                f"{device_name} total",
                f"{device_name} time avg",
            ]
        )
    if profile_memory:
        headers.extend(
            [
                "CPU Mem",
                "Self CPU Mem",
            ]
        )
        if use_device and has_device_mem:
            headers.extend(
                [
                    f"{device_name} Mem",
                    f"Self {device_name} Mem",
                ]
            )
    headers.append("# of Calls")
    # Only append Node ID if any event has a valid (>= 0) Node ID
    append_node_id = any(evt.node_id != -1 for evt in events)
    if append_node_id:
        headers.append("Node ID")

    # Have to use a list because nonlocal is Py3 only...
    SPACING_SIZE = 2
    row_format_lst = [""]
    header_sep_lst = [""]
    line_length_lst = [-SPACING_SIZE]

    def add_column(padding, text_dir=">"):
        row_format_lst[0] += (
            "{: " + text_dir + str(padding) + "}" + (" " * SPACING_SIZE)
        )
        header_sep_lst[0] += "-" * padding + (" " * SPACING_SIZE)
        line_length_lst[0] += padding + SPACING_SIZE

    def auto_scale_flops(flops):
        flop_headers = [
            "FLOPs",
            "KFLOPs",
            "MFLOPs",
            "GFLOPs",
            "TFLOPs",
            "PFLOPs",
        ]
        assert flops > 0
        log_flops = max(0, min(math.log10(flops) / 3, float(len(flop_headers) - 1)))
        assert log_flops >= 0 and log_flops < len(flop_headers)
        return (pow(10, (math.floor(log_flops) * -3.0)), flop_headers[int(log_flops)])

    add_column(name_column_width)
    if has_overload_names:
        add_column(name_column_width)
    for _ in headers[1 + has_overload_names :]:
        add_column(DEFAULT_COLUMN_WIDTH)

    if has_input_shapes:
        headers.append("Input Shapes")
        add_column(shapes_column_width)

    if has_stack:
        headers.append("Source Location")
        add_column(src_column_width, text_dir="<")

    if with_flops:
        # Auto-scaling of flops header
        raw_flops = [evt.flops for evt in events if evt.flops > 0]
        if len(raw_flops) != 0:
            (flops_scale, flops_header) = auto_scale_flops(min(raw_flops))
            headers.append(f"Total {flops_header}")
            add_column(flops_column_width)
        else:
            with_flops = False  # can't find any valid flops

    row_format = row_format_lst[0]
    header_sep = header_sep_lst[0]
    line_length = line_length_lst[0]
    add_column = None  # type: ignore[assignment]

    # Have to use a list because nonlocal is Py3 only...
    result = []

    def append(s):
        result.append(s)
        result.append("\n")  # Yes, newline after the end as well

    sum_self_cpu_time_total = 0
    sum_self_device_time_total = 0
    for evt in events:
        sum_self_cpu_time_total += evt.self_cpu_time_total
        if evt.device_type == DeviceType.CPU and evt.is_legacy:
            # in legacy profiler, kernel info is stored in cpu events
            sum_self_device_time_total += evt.self_device_time_total
        elif (
            evt.device_type
            in [
                DeviceType.CUDA,
                DeviceType.PrivateUse1,
                DeviceType.MTIA,
            ]
            and not evt.is_user_annotation
        ):
            # in kineto profiler, there're events with the correct device type (e.g. CUDA)
            sum_self_device_time_total += evt.self_device_time_total

    # Actual printing
    if header is not None:
        append("=" * line_length)
        append(header)
    if top_level_events_only:
        append("=" * line_length)
        append("This report only display top-level ops statistics")
    append(header_sep)
    append(row_format.format(*headers))

    append(header_sep)

    def trim_path(path, src_column_width):
        if len(path) > src_column_width:
            offset = len(path) - src_column_width
            path = path[offset:]
            if len(path) > 3:
                path = "..." + path[3:]
        return path

    event_limit = 0
    for evt in events:
        if event_limit == row_limit:
            break
        if top_level_events_only and evt.cpu_parent is not None:
            continue
        else:
            event_limit += 1
        name = evt.key
        if max_name_column_width is not None and len(name) >= max_name_column_width - 3:
            name = name[: (max_name_column_width - 3)] + "..."

        evt.self_cpu_percent = _format_time_share(
            evt.self_cpu_time_total, sum_self_cpu_time_total
        )
        evt.total_cpu_percent = (
            _format_time_share(evt.cpu_time_total, sum_self_cpu_time_total)
            if not evt.is_async
            else 0
        )

        row_values = [name]
        if has_overload_names:
            overload_name = evt.overload_name
            if (
                max_name_column_width is not None
                and len(overload_name) >= max_name_column_width - 3
            ):
                overload_name = overload_name[: (max_name_column_width - 3)] + "..."
            row_values += [overload_name]
        row_values += [
            # Self CPU total %, 0 for async events.
            evt.self_cpu_percent,
            evt.self_cpu_time_total_str,  # Self CPU total
            # CPU total %, 0 for async events.
            evt.total_cpu_percent,
            evt.cpu_time_total_str,  # CPU total
            evt.cpu_time_str,  # CPU time avg
        ]
        if has_device_time:
            evt.total_device_percent = _format_time_share(
                evt.self_device_time_total, sum_self_device_time_total
            )
            row_values.extend(
                [
                    evt.self_device_time_total_str,
                    # device time total %
                    evt.total_device_percent,
                    evt.device_time_total_str,
                    evt.device_time_str,  # device time avg
                ]
            )
        if profile_memory:
            row_values.extend(
                [
                    # CPU Mem Total
                    _format_memory(evt.cpu_memory_usage),
                    # Self CPU Mem Total
                    _format_memory(evt.self_cpu_memory_usage),
                ]
            )
            if use_device and has_device_mem:
                row_values.extend(
                    [
                        # Device Mem Total
                        _format_memory(evt.device_memory_usage),
                        # Self Device Mem Total
                        _format_memory(evt.self_device_memory_usage),
                    ]
                )
        row_values.append(
            evt.count,  # Number of calls
        )

        if append_node_id:
            row_values.append(evt.node_id)
        if has_input_shapes:
            row_values.append(str(evt.input_shapes)[:shapes_column_width])
        if with_flops:
            if evt.flops <= 0:
                row_values.append("--")
            else:
                row_values.append(f"{evt.flops * flops_scale:8.3f}")  # type: ignore[possibly-undefined]
        if has_stack:
            src_field = ""
            if len(evt.stack) > 0:
                src_field = trim_path(evt.stack[0], src_column_width)
            row_values.append(src_field)
        append(row_format.format(*row_values))

        if has_stack:
            empty_headers = [""] * (len(headers) - 1)
            for entry in evt.stack[1:]:
                append(
                    row_format.format(
                        *(empty_headers + [trim_path(entry, src_column_width)])
                    )
                )
            empty_headers.append("")
            append(row_format.format(*empty_headers))

    append(header_sep)
    append(f"Self CPU time total: {_format_time(sum_self_cpu_time_total)}")
    if has_device_time:
        append(
            f"Self {use_device.upper() if use_device is not None else 'None'} "
            f"time total: {_format_time(sum_self_device_time_total)}"
        )
    return "".join(result)
