import torch
import torch.cuda
from torch.autograd.profiler_util import (
    EventList, FunctionEvent, MEMORY_EVENT_NAME, Interval,
    _filter_name, _filter_stack_entry, _rewrite_name
)

from torch.autograd import (
    DeviceType, ProfilerConfig, ProfilerState,
    _disable_profiler_legacy, _enable_profiler_legacy,
)

import itertools
from warnings import warn


class profile(object):
    """DEPRECATED: use torch.profiler instead"""
    def __init__(
            self,
            enabled=True,
            *,
            use_cuda=False,
            use_xpu=False,
            record_shapes=False,
            with_flops=False,
            profile_memory=False,
            with_stack=False,
            with_modules=False):
        self.enabled: bool = enabled
        if not self.enabled:
            return
        self.use_cuda = use_cuda
        self.use_xpu = use_xpu
        self.function_events = None
        self.entered = False
        self.record_shapes = record_shapes
        self.with_flops = with_flops
        self.record_shapes |= self.with_flops
        self.profile_memory = profile_memory
        self.with_stack = with_stack
        self.with_modules = with_modules

        if self.use_cuda and not torch.cuda.is_available():
            warn("CUDA is not available, disabling CUDA profiling")
            self.use_cuda = False

        if self.use_xpu and not (hasattr(torch, 'xpu') and torch.xpu.is_available()):    # type: ignore[attr-defined]
            warn("XPU is not available, disabling XPU profiling")
            self.use_xpu = False

        if self.use_cuda:
            self.profiler_kind = ProfilerState.CUDA
        elif self.use_xpu:
            self.profiler_kind = ProfilerState.XPU
        else:
            self.profiler_kind = ProfilerState.CPU

    def config(self):
        return ProfilerConfig(
            self.profiler_kind,
            self.record_shapes,
            self.profile_memory,
            self.with_stack,
            self.with_flops,
            self.with_modules)

    def __enter__(self):
        if not self.enabled:
            return
        if self.entered:
            raise RuntimeError("Profiler context manager is not reentrant")
        self.entered = True
        self._start_trace()
        return self

    def _start_trace(self):
        _enable_profiler_legacy(self.config())

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.enabled:
            return
        if self.use_cuda:
            torch.cuda.synchronize()
        if self.use_xpu:
            torch.xpu.synchronize()    # type: ignore[attr-defined]

        records = _disable_profiler_legacy()
        parsed_results = _parse_legacy_records(records)
        self.function_events = EventList(
            parsed_results,
            use_cuda=self.use_cuda,
            use_xpu=self.use_xpu,
            profile_memory=self.profile_memory,
            with_flops=self.with_flops)
        self.function_events._build_tree()
        return False

    def __repr__(self):
        if self.function_events is None:
            return '<unfinished profiler_legacy.profile>'
        return repr(self.function_events)

    def __str__(self):
        if self.function_events is None:
            return '<unfinished profile.profiler_legacy.profile>'
        return str(self.function_events)

    def _check_finish(self):
        if self.function_events is None:
            raise RuntimeError("Profiler didn't finish running")

    def table(self, sort_by=None, row_limit=100, max_src_column_width=75, header=None, top_level_events_only=False):
        self._check_finish()
        assert self.function_events is not None
        return self.function_events.table(
            sort_by=sort_by, row_limit=row_limit, max_src_column_width=max_src_column_width, header=header,
            top_level_events_only=top_level_events_only
        )
    table.__doc__ = EventList.table.__doc__

    def export_chrome_trace(self, path):
        self._check_finish()
        assert self.function_events is not None
        return self.function_events.export_chrome_trace(path)
    export_chrome_trace.__doc__ = EventList.export_chrome_trace.__doc__

    def export_stacks(self, path: str, metric: str = "self_cpu_time_total"):
        self._check_finish()
        assert self.function_events is not None, "Expected profiling results"
        assert self.with_stack, "export_stacks() requires with_stack=True"
        return self.function_events.export_stacks(path, metric)

    def key_averages(self, group_by_input_shape=False, group_by_stack_n=0):
        self._check_finish()
        assert self.function_events is not None, "Expected profiling results"
        return self.function_events.key_averages(group_by_input_shape, group_by_stack_n)
    key_averages.__doc__ = EventList.key_averages.__doc__

    def total_average(self):
        self._check_finish()
        assert self.function_events is not None, "Expected profiling results"
        return self.function_events.total_average()
    total_average.__doc__ = EventList.total_average.__doc__

    @property
    def self_cpu_time_total(self):
        """ Returns total time spent on CPU obtained as a sum of
        all self times across all the events.
        """
        self._check_finish()
        assert self.function_events is not None
        return self.function_events.self_cpu_time_total


def _parse_legacy_records(thread_records):
    def _get_record_key(record):
        """
        Returns a tuple to be used by _parse_legacy_records for correlating start and
        end records.
        """
        return (record.handle(), record.node_id())

    next_id = 0
    start_record = None
    functions = []
    record_stack = []

    # '__start_profile' is not guaranteed to be first, so we must find it here
    for record in itertools.chain(*thread_records):
        name = record.name()
        if start_record is None and name == '__start_profile':
            start_record = record

    assert start_record is not None and not start_record.is_remote()

    for thread_record_list in thread_records:
        # accumulated memory allocations per handle
        cpu_memory_allocs = {}
        cuda_memory_allocs = {}
        xpu_memory_allocs = {}
        # ranges per handle
        range_starts = {}
        function_stack = []

        filtered_handles = set()
        prev_record = None
        for record in thread_record_list:
            record_key = _get_record_key(record)
            if (_filter_name(record.name()) or
                    record_key in filtered_handles):
                filtered_handles.add(record_key)
                continue

            if record.kind() == 'push':
                # workaround to reduce double logging from operator
                # wrappers and redispatch
                if prev_record is not None:
                    duplicate = (
                        prev_record.name() == record.name()
                        and prev_record.kind() == record.kind()
                        and prev_record.node_id() == record.node_id()
                    )
                    if duplicate:
                        filtered_handles.add(record_key)
                        continue

                # create the function event for appending kernel
                fe = FunctionEvent(
                    id=record.handle(),
                    name=_rewrite_name(name=record.name(), with_wildcard=True),
                    thread=record.thread_id(),
                    start_us=0,
                    end_us=0,
                    stack=[],
                    node_id=record.node_id(),
                    input_shapes=record.shapes(),
                    device_type=DeviceType.CPU,
                    is_legacy=True,
                )
                function_stack.append(fe)
                range_starts[record_key] = (record, fe)
                cpu_memory_allocs[record_key] = 0
                cuda_memory_allocs[record_key] = 0
                xpu_memory_allocs[record_key] = 0
            elif record.kind() == 'pop':
                assert (
                    record_key in range_starts
                ), """Expected record with key {} to exist in range_starts.
                    This means that the pop event did not have a corresponding push.""".format(
                    record_key
                )

                start, fe = range_starts[record_key]

                cpu_memory_usage = cpu_memory_allocs[record_key]
                cuda_memory_usage = cuda_memory_allocs[record_key]
                xpu_memory_usage = xpu_memory_allocs[record_key]
                is_async = start.is_async() or (
                    start.thread_id() != record.thread_id()
                )
                is_remote_event = record.is_remote()
                start_flops = start.flops()

                fe.time_range = Interval(start_record.cpu_elapsed_us(start), start_record.cpu_elapsed_us(record))
                fe.cpu_memory_usage = cpu_memory_usage
                fe.cuda_memory_usage = cuda_memory_usage
                fe.xpu_memory_usage = xpu_memory_usage
                fe.is_async = is_async
                fe.is_remote = is_remote_event
                fe.fwd_thread = start.fwd_thread_id()
                fe.stack = [entry for entry in start.stack() if _filter_stack_entry(entry)]
                fe.scope = start.scope()
                fe.sequence_nr = start.sequence_nr()
                fe.trace_name = _rewrite_name(name=start.name(), with_wildcard=False)
                fe.fwd_thread = start.fwd_thread_id()
                fe.flops = start_flops

                # note: async events have only cpu total time
                if not is_async and start.has_cuda():
                    duration = start.cuda_elapsed_us(record)
                    if duration > 0:
                        fe.append_kernel(
                            start.name(),
                            start.device(),
                            duration)
                functions.append(fe)
                function_stack.remove(fe)
                del range_starts[record_key]
                del cpu_memory_allocs[record_key]
                del cuda_memory_allocs[record_key]
                del xpu_memory_allocs[record_key]
            elif record.kind() == 'memory_alloc':
                num_open_handles_cpu = len(cpu_memory_allocs)
                num_open_handles_cuda = len(cuda_memory_allocs)
                assert num_open_handles_cpu == num_open_handles_cuda
                for handle in cpu_memory_allocs.keys():
                    cpu_memory_allocs[handle] += record.cpu_memory_usage()
                for handle in cuda_memory_allocs.keys():
                    cuda_memory_allocs[handle] += record.cuda_memory_usage()
                for handle in xpu_memory_allocs.keys():
                    xpu_memory_allocs[handle] += record.xpu_memory_usage()
                if num_open_handles_cpu == 0:
                    # output event as a top-level memory event
                    fe = FunctionEvent(
                        id=0,
                        name=MEMORY_EVENT_NAME,
                        trace_name=None,
                        thread=0,
                        start_us=0,
                        end_us=0,
                        stack=[],
                        cpu_memory_usage=record.cpu_memory_usage(),
                        cuda_memory_usage=record.cuda_memory_usage(),
                        xpu_memory_usage=record.xpu_memory_usage(),
                        is_legacy=True,
                    )
                    functions.append(fe)
            elif record.kind() == 'mark':
                if '__xpu_start_event' in record.name():
                    continue
                if record.has_xpu():
                    if len(function_stack) > 0:
                        fe = function_stack[-1]
                        fe.append_kernel(fe.name + "(" + record.name() + ")",
                                         record.device(),
                                         record.xpu_elapsed_us())
                    else:
                        # An xpu event is recorded but no parent function was recorded.
                        fe = FunctionEvent(
                            id=record.handle(),
                            node_id=record.node_id(),
                            name=_rewrite_name(name=record.name(), with_wildcard=True),
                            thread=record.thread_id(),
                            start_us=0,
                            end_us=0,
                            stack=[],
                            input_shapes=record.shapes(),
                            is_legacy=True)
                        fe.stack = []
                        fe.append_kernel(fe.name + "(" + record.name() + ")",
                                         record.device(),
                                         record.xpu_elapsed_us())
                        functions.append(fe)
            prev_record = record

    # Sort functions by start time then by end time ascending.
    # This ensures that--in the case of nested events which
    # have the same start time (which may happen due to the
    # granularity of the given clock tick)--we always show
    # the outermost nested call first. This adds stability
    # in how FunctionEvents appear
    functions.sort(key=lambda evt: [evt.time_range.start, -evt.time_range.end])
    return functions
