import itertools
from typing import Any
import torch
from torch.autograd import DeviceType
from torch.futures import Future

from collections import defaultdict, namedtuple
from operator import attrgetter

from typing import Dict, List, Tuple, Optional

import math

try:
    # Available in Python >= 3.2
    from contextlib import ContextDecorator
except ImportError:
    import functools

    class ContextDecorator(object):  # type: ignore[no-redef]

        def __enter__(self):
            raise NotImplementedError

        def __exit__(self, exc_type, exc_val, exc_tb):
            raise NotImplementedError

        def __call__(self, func):
            @functools.wraps(func)
            def wrapped(*args, **kwargs):
                with self:
                    return func(*args, **kwargs)

            return wrapped


class EventList(list):
    """A list of Events (for pretty printing)"""
    def __init__(self, *args, **kwargs):
        use_cuda = kwargs.pop('use_cuda', True)
        profile_memory = kwargs.pop('profile_memory', False)
        with_flops = kwargs.pop('with_flops', False)
        super(EventList, self).__init__(*args, **kwargs)
        self._use_cuda = use_cuda
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
            to_delete = []
            for idx in range(len(self)):
                if (self[idx].cpu_parent is not None and
                        self[idx].cpu_parent.name == self[idx].name and
                        len(self[idx].cpu_parent.cpu_children) == 1):
                    self[idx].cpu_parent.cpu_children = self[idx].cpu_children
                    self[idx].cpu_parent.kernels = self[idx].kernels  # lift kernels up
                    for ch in self[idx].cpu_children:
                        ch.cpu_parent = self[idx].cpu_parent
                    to_delete.append(idx)
            if len(to_delete) == 0:
                break
            new_evts = [ev for ind, ev in enumerate(self) if ind not in to_delete]
            self.clear()
            self.extend(new_evts)

    def _populate_cpu_children(self):
        """Populates child events into each underlying FunctionEvent object.
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
        sync_events = [evt for evt in self if not evt.is_async and evt.device_type == DeviceType.CPU]
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
        for thread_id, thread_events in threads:
            thread_events_ = sorted(
                thread_events,
                key=lambda event: [event.time_range.start, -event.time_range.end],
            )
            current_events: List[FunctionEvent] = []
            cur_end = 0
            for event in thread_events_:
                while len(current_events) > 0:
                    parent = current_events[-1]
                    if event.time_range.start >= parent.time_range.end or \
                            event.time_range.end > parent.time_range.end:
                        # this can't be a parent
                        current_events.pop()
                    else:
                        parent.append_cpu_child(event)
                        assert (
                            event.cpu_parent is None
                        ), "There is already a CPU parent event for {}".format(
                            event.key
                        )
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
        return sum([event.self_cpu_time_total for event in self])

    def table(self, sort_by=None, row_limit=100, max_src_column_width=75, header=None, top_level_events_only=False):
        """Prints an EventList as a nicely formatted table.

        Args:
            sort_by (str, optional): Attribute used to sort entries. By default
                they are printed in the same order as they were registered.
                Valid keys include: ``cpu_time``, ``cuda_time``, ``cpu_time_total``,
                ``cuda_time_total``, ``cpu_memory_usage``, ``cuda_memory_usage``,
                ``self_cpu_memory_usage``, ``self_cuda_memory_usage``, ``count``.
            top_level_events_only(bool, optional): Boolean flag to determine the
                selection of events to display. If true, the profiler will only
                display events at top level like top-level invocation of python
                `lstm`, python `add` or other functions, nested events like low-level
                cpu/cuda ops events are omitted for profiler result readability.

        Returns:
            A string containing the table.
        """
        return build_table(
            self,
            sort_by=sort_by,
            row_limit=row_limit,
            max_src_column_width=max_src_column_width,
            header=header,
            profile_memory=self._profile_memory,
            with_flops=self._with_flops,
            top_level_events_only=top_level_events_only)

    def export_chrome_trace(self, path):
        """Exports an EventList as a Chrome tracing tools file.

        The checkpoint can be later loaded and inspected under ``chrome://tracing`` URL.

        Args:
            path (str): Path where the trace will be written.
        """
        import os
        with open(path, 'w') as f:
            chrome_events = []
            next_id = 0
            # Use file IO over using json.dump since JSON dumping is very slow and
            # this technique is proven to give a 4x speedup.
            f.write("[")
            for evt in self:
                if evt.trace_name is None:
                    continue
                f.write(
                    '{"name": "%s", '
                    '"ph": "X", '
                    '"ts": %s, '
                    '"dur": %s, '
                    '"tid": %s, '
                    '"pid": "CPU functions", '
                    '"args": {}}, '
                    % (
                        evt.trace_name,
                        evt.time_range.start,
                        evt.time_range.elapsed_us(),
                        evt.thread
                        if not evt.is_remote
                        else f'" node_id:{evt.node_id}, thread_id:{evt.thread} "',
                    )
                )
                for k in evt.kernels:
                    # 's' and 'f' draw Flow arrows from
                    # the CPU launch to the GPU kernel
                    f.write('{"name": "%s", '
                            '"ph": "s", '
                            '"ts": %s, '
                            '"tid": %s, '
                            '"pid": "CPU functions", '
                            '"id": %s, '
                            '"cat": "cpu_to_cuda", '
                            '"args": {}}, ' % (evt.trace_name, evt.time_range.start,
                                               evt.thread, next_id))
                    # Note: use torch.profiler to get device kernel trace
                    next_id += 1

            # remove trailing whitespace and comma
            f.seek(f.tell() - 2, os.SEEK_SET)
            f.truncate()
            f.write("]")

    def supported_export_stacks_metrics(self):
        return ["self_cpu_time_total", "self_cuda_time_total"]

    def export_stacks(self, path: str, metric: str):
        if metric not in self.supported_export_stacks_metrics():
            raise ValueError("metric should be one of: " + str(self.supported_export_stacks_metrics()))
        translate_table = str.maketrans(" ;\t\n", "____")
        with open(path, 'w') as f:
            for evt in self:
                if evt.stack and len(evt.stack) > 0:
                    metric_value = getattr(evt, metric)
                    if int(metric_value) > 0:
                        stack_str = ""
                        for entry in reversed(evt.stack):
                            stack_str += entry.translate(translate_table)
                            stack_str += ";"
                        stack_str = stack_str[:-1] + " " + str(int(metric_value))
                        f.write(stack_str + "\n")

    def key_averages(self, group_by_input_shapes=False, group_by_stack_n=0):
        """Averages all function events over their keys.

        Args:
            group_by_input_shapes: group entries by
                (event name, input shapes) rather than just event name.
                This is useful to see which input shapes contribute to the runtime
                the most and may help with size-specific optimizations or
                choosing the best candidates for quantization (aka fitting a roof line)

            group_by_stack_n: group by top n stack trace entries

        Returns:
            An EventList containing FunctionEventAvg objects.
        """
        assert self._tree_built
        stats: Dict[Tuple[str, ...], FunctionEventAvg] = defaultdict(FunctionEventAvg)

        def get_key(event, group_by_input_shapes, group_by_stack_n) -> Tuple[str, ...]:
            key = [str(event.key), str(event.node_id), str(event.device_type), str(event.is_legacy)]
            if group_by_input_shapes:
                key.append(str(event.input_shapes))
            if group_by_stack_n > 0:
                key += event.stack[:group_by_stack_n]
            return tuple(key)
        for evt in self:
            stats[get_key(evt, group_by_input_shapes, group_by_stack_n)].add(evt)

        avg_list = EventList(
            stats.values(),
            use_cuda=self._use_cuda,
            profile_memory=self._profile_memory,
            with_flops=self._with_flops)
        for evt in avg_list:
            evt.stack = evt.stack[:group_by_stack_n]
            if not group_by_input_shapes:
                evt.input_shapes = ""
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
        total_stat.key = 'Total'
        return total_stat


class profile(object):
    """Context manager that manages autograd profiler state and holds a summary of results.
    Under the hood it just records events of functions being executed in C++ and
    exposes those events to Python. You can wrap any code into it and it will
    only report runtime of PyTorch functions.
    Note: profiler is thread local and is automatically propagated into the async tasks

    Args:
        enabled (bool, optional): Setting this to False makes this context manager a no-op.

        use_cuda (bool, optional): Enables timing of CUDA events as well using the cudaEvent API.
            Adds approximately 4us of overhead to each tensor operation.

        record_shapes (bool, optional): If shapes recording is set, information
            about input dimensions will be collected. This allows one to see which
            dimensions have been used under the hood and further group by them
            using prof.key_averages(group_by_input_shape=True). Please note that
            shape recording might skew your profiling data. It is recommended to
            use separate runs with and without shape recording to validate the timing.
            Most likely the skew will be negligible for bottom most events (in a case
            of nested function calls). But for higher level functions the total
            self cpu time might be artificially increased because of the shape
            collection.

        with_flops (bool, optional): If with_flops is set, the profiler will estimate
            the FLOPS (floating pointer operations per second) value using the operator's input shape
            and total time. This allows one to estimate the hardware performance. Currently,
            this option only works for the matrix multiplication and 2D convolution operators.

        profile_memory (bool, optional): track tensor memory allocation/deallocation.

        with_stack (bool, optional): record source information (file and line number) for the ops.

        use_kineto (bool, optional): experimental, enable profiling with Kineto profiler.

        use_cpu (bool, optional): profile CPU events; setting to ``False`` requires
            ``use_kineto=True`` and can be used to lower the overhead for GPU-only profiling.

    .. warning:
        Enabling memory profiling or source attribution incurs additional profiler
        overhead

    .. warning:
        This context managers should not be called recursively, i.e. no nested
        instances are allowed

    .. warning:
        Due to some CUDA multiprocessing limitations (multiprocessing-cuda-note_),
        one cannot use the profiler with ``use_cuda = True`` to benchmark
        DataLoaders with ``num_workers > 0``. If you wish to benchmark data loading,
        please use ``use_cuda = False`` or ``num_workers = 0``.

    Example:
        >>> x = torch.randn((1, 1), requires_grad=True)
        >>> with torch.autograd.profiler.profile() as prof:
        >>>     for _ in range(100):  # any normal python code, really!
        >>>         y = x ** 2
        >>          y.backward()
        >>> # NOTE: some columns were removed for brevity
        >>> print(prof.key_averages().table(sort_by="self_cpu_time_total"))
        -----------------------------------  ---------------  ---------------  ---------------
        Name                                 Self CPU total   CPU time avg     Number of Calls
        -----------------------------------  ---------------  ---------------  ---------------
        mul                                  32.048ms         32.048ms         200
        pow                                  27.041ms         27.041ms         200
        PowBackward0                         9.727ms          55.483ms         100
        torch::autograd::AccumulateGrad      9.148ms          9.148ms          100
        torch::autograd::GraphRoot           691.816us        691.816us        100
        -----------------------------------  ---------------  ---------------  ---------------

    """
    def __init__(
            self,
            enabled=True,
            *,
            use_cuda=False,
            record_shapes=False,
            with_flops=False,
            profile_memory=False,
            with_stack=False,
            use_kineto=False,
            use_cpu=True):
        self.enabled: bool = enabled
        if not self.enabled:
            return
        self.use_cuda = use_cuda
        self.function_events = None
        self.entered = False
        self.record_shapes = record_shapes
        self.with_flops = with_flops
        self.record_shapes |= self.with_flops
        self.profile_memory = profile_memory
        self.with_stack = with_stack
        self.use_cpu = use_cpu
        self.kineto_results = None
        if not self.use_cpu:
            assert use_kineto, \
                "Device-only events supported only with Kineto (use_kineto=True)"

        self.profiler_kind = None
        self.kineto_activities = set()
        if use_kineto:
            self.profiler_kind = torch.autograd.ProfilerState.KINETO
            if self.use_cpu:
                self.kineto_activities.add(torch.autograd.ProfilerActivity.CPU)
            if self.use_cuda:
                self.kineto_activities.add(
                    # uses CUPTI
                    torch.autograd.ProfilerActivity.CUDA)
            assert len(self.kineto_activities) > 0, \
                "No activities specified for Kineto profiler"
        elif self.use_cuda:
            # legacy CUDA mode
            self.profiler_kind = torch.autograd.ProfilerState.CUDA
        else:
            self.profiler_kind = torch.autograd.ProfilerState.CPU

        if self.profiler_kind == torch.autograd.ProfilerState.KINETO:
            assert (
                torch.autograd.kineto_available()
            ), """Requested Kineto profiling but Kineto is not available,
                  make sure PyTorch is built with USE_KINETO=1"""

    def config(self):
        assert self.profiler_kind is not None
        return torch.autograd.ProfilerConfig(
            self.profiler_kind,
            self.record_shapes,
            self.profile_memory,
            self.with_stack,
            self.with_flops)

    def __enter__(self):
        if not self.enabled:
            return
        if self.entered:
            raise RuntimeError("profiler context manager is not reentrant")
        self.entered = True
        if self.kineto_activities:
            torch.autograd._prepare_profiler(self.config(), self.kineto_activities)
            torch.autograd._enable_profiler(self.config(), self.kineto_activities)
        else:
            torch.autograd._enable_profiler_legacy(self.config())
        return self

    def _prepare_kineto_trace(self):
        assert self.kineto_activities
        self.entered = True
        torch.autograd._prepare_profiler(self.config(), self.kineto_activities)

    def _start_kineto_trace(self):
        assert self.kineto_activities
        torch.autograd._enable_profiler(self.config(), self.kineto_activities)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.enabled:
            return
        if self.kineto_activities:
            self.kineto_results = torch.autograd._disable_profiler()
            parsed_results = parse_kineto_results(self.kineto_results)
        else:
            records = torch.autograd._disable_profiler_legacy()
            parsed_results = parse_legacy_records(records)
        self.function_events = EventList(
            parsed_results,
            use_cuda=self.use_cuda,
            profile_memory=self.profile_memory,
            with_flops=self.with_flops)
        self.function_events._build_tree()
        return False

    def __repr__(self):
        if self.function_events is None:
            return '<unfinished torch.autograd.profile>'
        return repr(self.function_events)

    def __str__(self):
        if self.function_events is None:
            return '<unfinished torch.autograd.profile>'
        return str(self.function_events)

    def _check_finish(self):
        if self.function_events is None:
            raise RuntimeError("can't export a trace that didn't finish running")

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
        if self.kineto_results is not None:
            self.kineto_results.save(path)
        else:
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


class record_function(ContextDecorator):
    """Context manager/function decorator that adds a label to a block of
    Python code (or function) when running autograd profiler. It is
    useful when tracing the code profile.

    Args:
        name (str): Label assigned to the block of code.
        node_id (int): ID of node, for distributed profiling. Unset in
        non-distributed cases.

    Example:
        >>> x = torch.randn((1, 1), requires_grad=True)
        >>> with torch.autograd.profiler.profile() as prof:
        ...     y = x ** 2
        ...     with torch.autograd.profiler.record_function("label-z"): # label the block
        ...         z = y ** 3
        ...     y.backward()
        ...
        >>> # NOTE: some columns were removed for brevity
        >>> print(prof.key_averages().table(sort_by="self_cpu_time_total"))
        -----------------------------------  ---------------  ---------------  ---------------
        Name                                 Self CPU total %  CPU time avg     Number of Calls
        -----------------------------------  ---------------  ---------------  ---------------
        pow                                  60.77%           47.470us         3
        mul                                  21.73%           25.465us         2
        PowBackward0                         12.03%           121.891us        1
        torch::autograd::AccumulateGrad      2.70%            6.324us          1
        label-z                              2.13%            12.421us         1
        torch::autograd::GraphRoot           0.64%            1.503us          1
        -----------------------------------  ---------------  ---------------  ---------------
        Self CPU time total: 234.344us
        CUDA time total: 0.000us

    """
    def __init__(self, name: str):
        self.name: str = name
        # Whether or not we should run record function's end callbacks when exiting.
        self.run_callbacks_on_exit: bool = True
        # Stores underlying RecordFunction as a tensor. TODO: move to custom
        # class (https://github.com/pytorch/pytorch/issues/35026).
        self.handle: torch.Tensor = torch.zeros(1)

    def __enter__(self):
        self.handle = torch.ops.profiler._record_function_enter(self.name)
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any):
        if self.run_callbacks_on_exit:
            torch.ops.profiler._record_function_exit(self.handle)

    def _call_end_callbacks_on_future(self, fut: Future[Any]) -> Future[Any]:
        """
        _call_end_callbacks_on_future is meant to be used for profiling async
        calls that return a future. Calling this function will extend recording
        beyond this scope, until the future is satisfied. It is useful for profiling
        the end to end time of asynchronous calls. This function should only be called
        once to attach the callback onto the future, and will throw if called multiple
        times.

        Args:
            fut: (torch._C.Future): future for which to schedule
            callback for.

        Returns:
            A future that completes with the value of the passed in future when
            the profiling callbacks have ran.

        """
        # Throw if we have already attached a callback onto the future.
        if not self.run_callbacks_on_exit:
            raise RuntimeError("_call_end_callbacks_on_future can only be called once.")

        # We are scheduling to run this RecordFunction's end callbacks when the
        # passed in future completes, so don't run end callbacks on exit.
        self.run_callbacks_on_exit = False
        profiled_future = torch.ops.profiler._call_end_callbacks_on_jit_fut(self.handle, fut)
        return profiled_future


class emit_nvtx(object):
    """Context manager that makes every autograd operation emit an NVTX range.

    It is useful when running the program under nvprof::

        nvprof --profile-from-start off -o trace_name.prof -- <regular command here>

    Unfortunately, there's no way to force nvprof to flush the data it collected
    to disk, so for CUDA profiling one has to use this context manager to annotate
    nvprof traces and wait for the process to exit before inspecting them.
    Then, either NVIDIA Visual Profiler (nvvp) can be used to visualize the timeline, or
    :func:`torch.autograd.profiler.load_nvprof` can load the results for inspection
    e.g. in Python REPL.

    .. warning:
        This context manager should not be called recursively, i.e. at most one
        instance should be enabled at any given time.

    Args:
        enabled (bool, optional, default=True): Setting ``enabled=False`` makes this context manager a no-op.
            Default: ``True``.
        record_shapes (bool, optional, default=False): If ``record_shapes=True``, the nvtx range wrapping
            each autograd op will append information about the sizes of Tensor arguments received
            by that op, in the following format:
            ``[[arg0.size(0), arg0.size(1), ...], [arg1.size(0), arg1.size(1), ...], ...]``
            Non-tensor arguments will be represented by ``[]``.
            Arguments will be listed in the order they are received by the backend op.
            Please note that this order may not match the order in which those arguments were passed
            on the Python side.  Also note that shape recording may increase the overhead of nvtx range creation.

    Example:
        >>> with torch.cuda.profiler.profile():
        ...     model(x) # Warmup CUDA memory allocator and profiler
        ...     with torch.autograd.profiler.emit_nvtx():
        ...         model(x)

    **Forward-backward correlation**

    When viewing a profile created using :class:`emit_nvtx` in the Nvidia Visual Profiler,
    correlating each backward-pass op with the corresponding forward-pass op can be difficult.
    To ease this task, :class:`emit_nvtx` appends sequence number information to the ranges it
    generates.

    During the forward pass, each function range is decorated with ``seq=<N>``.  ``seq`` is a running
    counter, incremented each time a new backward Function object is created and stashed for backward.
    Thus, the ``seq=<N>`` annotation associated with each forward function range tells you that
    if a backward Function object is created by this forward function,
    the backward object will receive sequence number N.
    During the backward pass, the top-level range wrapping each C++ backward Function's
    ``apply()`` call is decorated with ``stashed seq=<M>``.  ``M`` is the sequence number that
    the backward object was created with.  By comparing ``stashed seq`` numbers in backward with ``seq``
    numbers in forward, you can track down which forward op created each backward Function.

    Any functions executed during the backward pass are also decorated with ``seq=<N>``.  During
    default backward (with ``create_graph=False``) this information is irrelevant, and in fact,
    ``N`` may simply be 0 for all such functions.  Only the top-level ranges associated with
    backward Function objects' ``apply()`` methods are useful, as a way to correlate these Function
    objects with the earlier forward pass.

    **Double-backward**

    If, on the other hand, a backward pass with ``create_graph=True`` is underway (in other words,
    if you are setting up for a double-backward), each function's execution during backward
    is given a nonzero, useful ``seq=<N>``.  Those functions may themselves create Function objects
    to be executed later during double-backward, just as the original functions in the forward pass did.
    The relationship between backward and double-backward is conceptually the same as the relationship
    between forward and backward: The functions still emit current-sequence-number-tagged ranges,
    the Function objects they create still stash those sequence numbers, and during the eventual
    double-backward, the Function objects' ``apply()`` ranges are still tagged with ``stashed seq``
    numbers, which can be compared to `seq` numbers from the backward pass.

    .. warning:
        The sequence number is thread-local, and some forward functions don't create an associated
        backward Function object (instead delegating that to sub-functions further down the call chain).
        For these reasons, the correspondence of stashed sequence numbers in
        backward Function ``apply()`` ranges with `seq` numbers in forward-pass ranges is
        not guaranteed to be 1 to 1.  The sequence numbers alone may not be enough to fully
        disambiguate which forward function created which
        backward Function object.  You may need to make a judgment based on analytic knowledge of what
        the expected correspondence should be.
    """
    def __init__(self, enabled=True, record_shapes=False):
        self.enabled = enabled
        self.entered = False
        self.record_shapes = record_shapes

    def __enter__(self):
        if not self.enabled:
            return
        if self.entered:
            raise RuntimeError("NVTX annotation context manager is not reentrant")
        self.entered = True
        torch.cuda.synchronize()
        torch.autograd._enable_profiler_legacy(
            torch.autograd.ProfilerConfig(
                torch.autograd.ProfilerState.NVTX,
                self.record_shapes,
                False,
                False,
                False)
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.enabled:
            return
        torch.cuda.synchronize()
        torch.autograd._disable_profiler_legacy()
        return False


def load_nvprof(path):
    """Opens an nvprof trace file and parses autograd annotations.

    Args:
        path (str): path to nvprof trace
    """
    return EventList(parse_nvprof_trace(path))


################################################################################
# FunctionEvent

def format_time(time_us):
    """Defines how to format time in FunctionEvent"""
    US_IN_SECOND = 1000.0 * 1000.0
    US_IN_MS = 1000.0
    if time_us >= US_IN_SECOND:
        return '{:.3f}s'.format(time_us / US_IN_SECOND)
    if time_us >= US_IN_MS:
        return '{:.3f}ms'.format(time_us / US_IN_MS)
    return '{:.3f}us'.format(time_us)


def format_time_share(time_us, total_time_us):
    """Defines how to format time in FunctionEvent"""
    if total_time_us == 0:
        assert time_us == 0, "Expected time_us == 0 but got {}".format(time_us)
        return "NaN"
    return '{:.2f}%'.format(time_us * 100.0 / total_time_us)

def format_memory(nbytes):
    """Returns a formatted memory size string"""
    KB = 1024
    MB = 1024 * KB
    GB = 1024 * MB
    if (abs(nbytes) >= GB):
        return '{:.2f} Gb'.format(nbytes * 1.0 / GB)
    elif (abs(nbytes) >= MB):
        return '{:.2f} Mb'.format(nbytes * 1.0 / MB)
    elif (abs(nbytes) >= KB):
        return '{:.2f} Kb'.format(nbytes * 1.0 / KB)
    else:
        return str(nbytes) + ' b'

def attr_formatter(name):
    return property(lambda self: format_time(getattr(self, name)))


class FormattedTimesMixin(object):
    """Helpers for FunctionEvent and FunctionEventAvg.

    The subclass should define `*_time_total` and `count` attributes.
    """
    cpu_time_str = attr_formatter('cpu_time')
    cuda_time_str = attr_formatter('cuda_time')
    cpu_time_total_str = attr_formatter('cpu_time_total')
    cuda_time_total_str = attr_formatter('cuda_time_total')
    self_cpu_time_total_str = attr_formatter('self_cpu_time_total')
    self_cuda_time_total_str = attr_formatter('self_cuda_time_total')

    @property
    def cpu_time(self):
        return 0.0 if self.count == 0 else 1.0 * self.cpu_time_total / self.count  # type: ignore

    @property
    def cuda_time(self):
        return 0.0 if self.count == 0 else 1.0 * self.cuda_time_total / self.count  # type: ignore


class Interval(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def elapsed_us(self):
        return self.end - self.start


Kernel = namedtuple('Kernel', ['name', 'device', 'duration'])


class FunctionEvent(FormattedTimesMixin):
    """Profiling information about a single function."""
    def __init__(
            self, id, name, thread, start_us, end_us, fwd_thread=None, input_shapes=None,
            stack=None, scope=0, cpu_memory_usage=0, cuda_memory_usage=0, is_async=False,
            is_remote=False, sequence_nr=-1, node_id=-1, device_type=DeviceType.CPU, device_index=0,
            is_legacy=False, flops=None, trace_name=None):
        self.id: int = id
        self.node_id: int = node_id
        self.name: str = name
        self.trace_name: str = trace_name
        self.time_range: Interval = Interval(start_us, end_us)
        self.thread: int = thread
        self.fwd_thread: Optional[int] = fwd_thread
        self.kernels: List[Kernel] = []
        self.count: int = 1
        self.cpu_children: List[FunctionEvent] = []
        self.cpu_parent: Optional[FunctionEvent] = None
        self.input_shapes: Tuple[int, ...] = input_shapes
        self.stack: List = stack
        self.scope: int = scope
        self.cpu_memory_usage: int = cpu_memory_usage
        self.cuda_memory_usage: int = cuda_memory_usage
        self.is_async: bool = is_async
        self.is_remote: bool = is_remote
        self.sequence_nr: int = sequence_nr
        self.device_type: DeviceType = device_type
        self.device_index: int = device_index
        self.is_legacy: bool = is_legacy
        self.flops: Optional[float] = flops

    def append_kernel(self, name, device, duration):
        assert self.device_type == DeviceType.CPU
        self.kernels.append(Kernel(name, device, duration))

    def append_cpu_child(self, child):
        """Append a CPU child of type FunctionEvent.

        One is supposed to append only direct children to the event to have
        correct self cpu time being reported.
        """
        assert(self.device_type == DeviceType.CPU)
        assert(isinstance(child, FunctionEvent))
        assert(child.device_type == DeviceType.CPU)
        self.cpu_children.append(child)

    def set_cpu_parent(self, parent):
        """Set the immediate CPU parent of type FunctionEvent

        One profiling FunctionEvent should have only one CPU parent such that
        the child's range interval is completely inside the parent's. We use
        this connection to determine the event is from top-level op or not.
        """
        assert(self.device_type == DeviceType.CPU)
        assert(isinstance(parent, FunctionEvent))
        assert(parent.device_type == DeviceType.CPU)
        self.cpu_parent = parent

    # Note: async events don't have children, are not used when computing 'self'
    # metrics of other events, have only total cpu time
    @property
    def self_cpu_memory_usage(self):
        if self.is_async or self.device_type != DeviceType.CPU:
            return 0
        return self.cpu_memory_usage - sum(
            [child.cpu_memory_usage for child in self.cpu_children]
        )

    @property
    def self_cuda_memory_usage(self):
        if self.is_async or self.device_type != DeviceType.CPU:
            return 0
        return self.cuda_memory_usage - sum(
            [child.cuda_memory_usage for child in self.cpu_children]
        )

    @property
    def self_cpu_time_total(self):
        if self.is_async or self.device_type != DeviceType.CPU:
            return 0
        return self.cpu_time_total - sum(
            [child.cpu_time_total for child in self.cpu_children]
        )

    @property
    def cuda_time_total(self):
        if self.is_async:
            return 0
        if self.device_type == DeviceType.CPU:
            if not self.is_legacy:
                # account for the kernels in the children ops
                return (sum(kinfo.duration for kinfo in self.kernels) +
                        sum(ch.cuda_time_total for ch in self.cpu_children))
            else:
                # each legacy cpu events has a single (fake) kernel
                return sum(kinfo.duration for kinfo in self.kernels)
        else:
            assert self.device_type == DeviceType.CUDA
            return self.time_range.elapsed_us()

    @property
    def self_cuda_time_total(self):
        if self.is_async:
            return 0
        if self.device_type == DeviceType.CPU:
            return self.cuda_time_total - \
                sum([child.cuda_time_total for child in self.cpu_children])
        else:
            assert(self.device_type == DeviceType.CUDA)
            return self.cuda_time_total

    @property
    def cpu_time_total(self):
        if self.device_type == DeviceType.CPU:
            return self.time_range.elapsed_us()
        else:
            return 0

    @property
    def key(self):
        return self.name

    def __repr__(self):
        return (
            '<FunctionEvent id={} name={} device_type={} node_id={} cpu_time={} start_us={} end_us={} '
            'cpu_children={} cuda_time={} name={} thread={} input_shapes={} '
            'cpu_memory_usage={} cuda_memory_usage={} is_async={} is_remote={} seq_nr={} is_legacy={}>'.format(
                self.id,
                self.name,
                self.device_type,
                self.node_id,
                self.cpu_time_str,
                self.time_range.start,
                self.time_range.end,
                str([child.id for child in self.cpu_children]),
                self.cuda_time_str,
                self.name,
                self.thread,
                str(self.input_shapes),
                self.cpu_memory_usage,
                self.cuda_memory_usage,
                self.is_async,
                self.is_remote,
                self.sequence_nr,
                self.is_legacy,
            )
        )


class FunctionEventAvg(FormattedTimesMixin):
    """Used to average stats over multiple FunctionEvent objects."""
    def __init__(self):
        self.key: Optional[str] = None
        self.count: int = 0
        self.node_id: int = 0
        self.is_async: bool = False
        self.is_remote: bool = False
        self.cpu_time_total: int = 0
        self.cuda_time_total: int = 0
        self.self_cpu_time_total: int = 0
        self.self_cuda_time_total: int = 0
        self.input_shapes: Optional[List[List[int]]] = None
        self.stack: Optional[List] = None
        self.scope: Optional[int] = None
        self.cpu_memory_usage: int = 0
        self.cuda_memory_usage: int = 0
        self.self_cpu_memory_usage: int = 0
        self.self_cuda_memory_usage: int = 0
        self.cpu_children: Optional[List[FunctionEvent]] = None
        self.cpu_parent: Optional[FunctionEvent] = None
        self.device_type: DeviceType = DeviceType.CPU
        self.is_legacy: bool = False
        self.flops: float = 0.0

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

            self.input_shapes = other.input_shapes
            self.stack = other.stack
            self.scope = other.scope
            self.device_type = other.device_type
            self.is_legacy = other.is_legacy

        assert isinstance(other, (FunctionEvent, FunctionEventAvg))
        assert other.key == self.key
        self.cpu_time_total += other.cpu_time_total
        self.cuda_time_total += other.cuda_time_total
        self.self_cpu_time_total += other.self_cpu_time_total
        self.self_cuda_time_total += other.self_cuda_time_total
        self.cpu_memory_usage += other.cpu_memory_usage
        self.cuda_memory_usage += other.cuda_memory_usage
        self.self_cpu_memory_usage += other.self_cpu_memory_usage
        self.self_cuda_memory_usage += other.self_cuda_memory_usage
        self.count += other.count
        if self.flops is None:
            self.flops = other.flops
        elif other.flops is not None:
            self.flops += other.flops
        return self

    def __iadd__(self, other):
        return self.add(other)

    def __repr__(self):
        return (
            '<FunctionEventAvg key={} self_cpu_time={} cpu_time={} '
            ' self_cuda_time={} cuda_time={} input_shapes={} '
            'cpu_memory_usage={} cuda_memory_usage={}>'.format(
                self.key,
                self.self_cpu_time_total_str,
                self.cpu_time_str,
                self.self_cuda_time_total_str,
                self.cuda_time_str,
                str(self.input_shapes),
                self.cpu_memory_usage,
                self.cuda_memory_usage,
            )
        )


################################################################################
# Utilities

class StringTable(defaultdict):
    def __missing__(self, key):
        # manage cases like 't' (demangled to 'unsigned short') separately,
        # for now simply check the length to avoid unexpected results for
        # the short sequences
        self[key] = torch._C._demangle(key) if len(key) > 1 else key
        return self[key]

def filter_stack_entry(entry):
    filtered_entries = [
        ("autograd/__init__", "_make_grads"),
        ("autograd/__init__", "backward"),
        ("torch/tensor", "backward"),
        ("_internal/common_utils", "prof_callable"),
        ("_internal/common_utils", "prof_func_call"),
        ("_internal/common_utils", "prof_meth_call"),
    ]
    return all([not (f[0] in entry and f[1] in entry) for f in filtered_entries])

def filter_name(name):
    # ignoring the following utility ops
    filtered_out_names = [
        "profiler::_record_function_enter",
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
def rewrite_name(name, with_wildcard=False):
    string_table = StringTable()
    name = string_table[name]
    if with_wildcard:
        if name.startswith("ProfilerStep#"):
            name = "ProfilerStep*"
    return name

# Parsing of kineto profiler events
def parse_kineto_results(result):
    # result.events() has most of the events - PyTorch op-level and device-level events
    # result.legacy_events() has events not yet ported to kineto
    # (e.g. start/stop marks, tensor memory allocator events)

    # First, find __start_profile mark to get the absolute time of the start of the trace;
    # save memory allocation records
    start_record = None
    mem_records = []
    for record in itertools.chain(*result.legacy_events()):
        if record.kind() == 'mark' and record.name() == '__start_profile':
            assert start_record is None
            start_record = record
        if record.kind() == 'memory_alloc':
            mem_records.append([record, False])
    assert start_record is not None, "Invalid profiler output, __start_profile is missing"

    # Create and return FunctionEvent list
    function_events = []
    cuda_corr_map: Dict[int, List[FunctionEvent]] = {}
    for kineto_event in result.events():
        if filter_name(kineto_event.name()):
            continue
        rel_start_us = kineto_event.start_us() - start_record.start_us()
        rel_end_us = rel_start_us + kineto_event.duration_us()
        abs_end_us = kineto_event.start_us() + kineto_event.duration_us()

        cpu_memory_usage = 0
        cuda_memory_usage = 0
        if kineto_event.device_type() == DeviceType.CPU:
            # find the corresponding memory allocation events
            for mem_record in mem_records:
                if (mem_record[0].start_us() >= kineto_event.start_us() and
                        mem_record[0].start_us() <= abs_end_us):
                    cpu_memory_usage += mem_record[0].cpu_memory_usage()
                    cuda_memory_usage += mem_record[0].cuda_memory_usage()
                    mem_record[1] = True

        is_async = kineto_event.start_thread_id() != kineto_event.end_thread_id()
        fe = FunctionEvent(
            id=kineto_event.correlation_id(),
            name=rewrite_name(name=kineto_event.name(), with_wildcard=True),
            trace_name=rewrite_name(name=kineto_event.name(), with_wildcard=False),
            thread=kineto_event.start_thread_id(),
            start_us=rel_start_us,
            end_us=rel_end_us,
            fwd_thread=kineto_event.fwd_thread_id(),
            input_shapes=kineto_event.shapes(),
            stack=[entry for entry in kineto_event.stack() if filter_stack_entry(entry)],
            scope=kineto_event.scope(),
            cpu_memory_usage=cpu_memory_usage,
            cuda_memory_usage=cuda_memory_usage,
            is_async=is_async,
            sequence_nr=kineto_event.sequence_nr(),
            device_type=kineto_event.device_type(),
            device_index=kineto_event.device_index(),
            flops=kineto_event.flops(),
        )
        function_events.append(fe)
        corr_id = kineto_event.linked_correlation_id()
        if corr_id > 0:
            if corr_id not in cuda_corr_map:
                cuda_corr_map[corr_id] = []
            cuda_corr_map[corr_id].append(fe)

    # associate CUDA kernels and CUDA runtime (CPU) with CPU events
    for fe in function_events:
        if (fe.device_type == DeviceType.CPU and not fe.is_async and
                fe.id in cuda_corr_map):
            for f_evt in cuda_corr_map[fe.id]:
                if f_evt.device_type == DeviceType.CUDA:
                    fe.append_kernel(
                        f_evt.name,
                        f_evt.device_index,
                        f_evt.time_range.end - f_evt.time_range.start)
                elif f_evt.device_type == DeviceType.CPU:
                    # make sure that 'thread' of a CPU Kineto (e.g. CUDA Runtime) event is associated
                    # with the 'thread' of the corresponding linked PyTorch event to properly track
                    # parents and children
                    f_evt.thread = fe.thread


    # output top-level memory events
    for mem_record in mem_records:
        if not mem_record[1]:
            fe = FunctionEvent(
                id=mem_record[0].handle(),
                name="[memory]",
                trace_name=None,  # not outputting in the trace
                thread=mem_record[0].thread_id(),
                start_us=mem_record[0].start_us(),
                end_us=mem_record[0].start_us(),  # no duration
                fwd_thread=mem_record[0].fwd_thread_id(),
                input_shapes=[],
                stack=[],
                scope=mem_record[0].scope(),
                cpu_memory_usage=mem_record[0].cpu_memory_usage(),
                cuda_memory_usage=mem_record[0].cuda_memory_usage(),
                is_async=False,
                sequence_nr=-1,
                device_type=DeviceType.CPU,
                device_index=0,
            )
            function_events.append(fe)

    function_events.sort(key=lambda evt: [evt.time_range.start, -evt.time_range.end])
    return function_events

# Parsing of legacy profiler events
def parse_legacy_records(thread_records):
    def get_record_key(record):
        """
        Returns a tuple to be used by parse_legacy_records for correlating start and
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
        # ranges per handle
        range_starts = {}

        filtered_handles = set()
        prev_record = None
        for record in thread_record_list:
            record_key = get_record_key(record)
            if (filter_name(record.name()) or
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

                range_starts[record_key] = record
                cpu_memory_allocs[record_key] = 0
                cuda_memory_allocs[record_key] = 0
            elif record.kind() == 'pop':
                assert (
                    record_key in range_starts
                ), """Expected record with key {} to exist in range_starts.
                    This means that the pop event did not have a corresponding push.""".format(
                    record_key
                )

                start = range_starts[record_key]

                cpu_memory_usage = cpu_memory_allocs[record_key]
                cuda_memory_usage = cuda_memory_allocs[record_key]
                is_async = start.thread_id() != record.thread_id()
                is_remote_event = record.is_remote()
                start_flops = start.flops()

                fe = FunctionEvent(
                    id=record.handle(),
                    node_id=record.node_id(),
                    name=rewrite_name(name=start.name(), with_wildcard=True),
                    trace_name=rewrite_name(name=start.name(), with_wildcard=False),
                    thread=start.thread_id(),
                    start_us=start_record.cpu_elapsed_us(start),
                    end_us=start_record.cpu_elapsed_us(record),
                    fwd_thread=start.fwd_thread_id(),
                    input_shapes=start.shapes(),
                    stack=[entry for entry in start.stack() if filter_stack_entry(entry)],
                    scope=start.scope(),
                    cpu_memory_usage=cpu_memory_usage,
                    cuda_memory_usage=cuda_memory_usage,
                    is_async=is_async,
                    is_remote=is_remote_event,
                    sequence_nr=start.sequence_nr(),
                    device_type=DeviceType.CPU,
                    is_legacy=True,
                    flops=start_flops,
                )
                # note: async events have only cpu total time
                if not is_async and start.has_cuda():
                    duration = start.cuda_elapsed_us(record)
                    if duration > 0:
                        fe.append_kernel(
                            start.name(),
                            start.device(),
                            duration)
                functions.append(fe)
                del range_starts[record_key]
                del cpu_memory_allocs[record_key]
                del cuda_memory_allocs[record_key]
            elif record.kind() == 'memory_alloc':
                num_open_handles_cpu = len(cpu_memory_allocs)
                num_open_handles_cuda = len(cuda_memory_allocs)
                assert num_open_handles_cpu == num_open_handles_cuda
                for handle in cpu_memory_allocs.keys():
                    cpu_memory_allocs[handle] += record.cpu_memory_usage()
                for handle in cuda_memory_allocs.keys():
                    cuda_memory_allocs[handle] += record.cuda_memory_usage()
                if num_open_handles_cpu == 0:
                    # output event as a top-level memory event
                    fe = FunctionEvent(
                        id=0,
                        name="[memory]",
                        trace_name=None,
                        thread=0,
                        start_us=0,
                        end_us=0,
                        stack=[],
                        cpu_memory_usage=record.cpu_memory_usage(),
                        cuda_memory_usage=record.cuda_memory_usage(),
                        is_legacy=True,
                    )
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


################################################################################
# CUDA checkpoints

class EnforceUnique(object):
    """Raises an error if a key is seen more than once."""
    def __init__(self):
        self.seen = set()

    def see(self, *key):
        if key in self.seen:
            raise RuntimeError('duplicate key: ' + str(key))
        self.seen.add(key)


def parse_nvprof_trace(path):
    import sqlite3
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row

    # Parse strings table
    strings = {}
    for r in conn.execute("SELECT _id_ as id, value FROM StringTable"):
        strings[r["id"]] = torch._C._demangle(r["value"])

    # First, find all functions and create FunctionEvents for them
    marker_query = """
    SELECT
        start.id AS marker_id, start.name, start.timestamp AS start_time, end.timestamp AS end_time
    FROM
        CUPTI_ACTIVITY_KIND_MARKER AS start INNER JOIN CUPTI_ACTIVITY_KIND_MARKER AS end
        ON start.id = end.id
    WHERE
        start.name != 0 AND end.name = 0
    """
    functions = []
    functions_map = {}
    unique = EnforceUnique()
    for row in conn.execute(marker_query):
        unique.see(row['marker_id'])
        evt = FunctionEvent(id=row['marker_id'],
                            node_id=0,  # missing a node_id when calling FunctionEvent. This is just to ensure
                                        # that pytorch doesn't crash when creating a FunctionEvent() object
                            name=strings[row['name']],
                            start_us=row['start_time'],
                            end_us=row['end_time'],
                            thread=0)  # TODO: find in sqlite database
        functions.append(evt)
        functions_map[evt.id] = evt

    # Now, correlate all kernels with FunctionEvents
    kernel_query = """
    SELECT
        start.id AS marker_id, start.name, start.timestamp, end.timestamp,
        runtime._id_ AS runtime_id, runtime.cbid, runtime.start AS runtime_start, runtime.end AS runtime_end,
        kernel.start AS kernel_start, kernel.end AS kernel_end, kernel.name AS kernel_name
    FROM
        CUPTI_ACTIVITY_KIND_MARKER AS start
        INNER JOIN CUPTI_ACTIVITY_KIND_MARKER AS end
            ON start.id = end.id
        INNER JOIN CUPTI_ACTIVITY_KIND_RUNTIME as runtime
            ON (start.timestamp < runtime.start AND runtime.end < end.timestamp)
        INNER JOIN CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL AS kernel
            ON kernel.correlationId = runtime.correlationId
    """
    unique = EnforceUnique()
    for row in conn.execute(kernel_query):
        unique.see(row['marker_id'], row['runtime_id'])
        # 211 is cudaKernelLaunch for cuda >= 9.2; 13 is for older cuda versions
        assert (row['cbid'] == 211) or (row['cbid'] == 13)
        evt = functions_map[row['marker_id']]
        evt.append_kernel(row['kernel_name'],
                          0,
                          row['kernel_end'] - row['kernel_start'])

    functions.sort(key=lambda evt: evt.time_range.start)
    return functions


################################################################################
# Pretty printer


def build_table(
        events,
        sort_by=None,
        header=None,
        row_limit=100,
        max_src_column_width=75,
        with_flops=False,
        profile_memory=False,
        top_level_events_only=False):
    """Prints a summary of events (which can be a list of FunctionEvent or FunctionEventAvg)."""
    if len(events) == 0:
        return ""

    has_cuda_time = any([event.self_cuda_time_total > 0 for event in events])
    has_cuda_mem = any([event.self_cuda_memory_usage > 0 for event in events])
    has_input_shapes = any(
        [(event.input_shapes is not None and len(event.input_shapes) > 0) for event in events])

    if sort_by is not None:
        events = EventList(sorted(
            events, key=lambda evt: getattr(evt, sort_by), reverse=True
        ), use_cuda=has_cuda_time, profile_memory=profile_memory, with_flops=with_flops)

    MAX_NAME_COLUMN_WIDTH = 55
    name_column_width = max([len(evt.key) for evt in events]) + 4
    name_column_width = min(name_column_width, MAX_NAME_COLUMN_WIDTH)

    DEFAULT_COLUMN_WIDTH = 12
    shapes_column_width = max([len(str(evt.input_shapes)) for evt in events]) + 4
    shapes_column_width = min(shapes_column_width, 45)

    flops_column_width = DEFAULT_COLUMN_WIDTH

    src_column_width = None
    stacks = []
    for evt in events:
        if evt.stack is not None and len(evt.stack) > 0:
            stacks.append(evt.stack)
    has_stack = len(stacks) > 0
    if has_stack:
        src_column_width = max([max([len(entry) for entry in stack]) for stack in stacks]) + 4
        src_column_width = min(src_column_width, max_src_column_width)

    headers = [
        'Name',
        'Self CPU %',
        'Self CPU',
        'CPU total %',
        'CPU total',
        'CPU time avg',
    ]
    if has_cuda_time:
        headers.extend([
            'Self CUDA',
            'Self CUDA %',
            'CUDA total',
            'CUDA time avg',
        ])
    if profile_memory:
        headers.extend([
            'CPU Mem',
            'Self CPU Mem',
        ])
        if has_cuda_mem:
            headers.extend([
                'CUDA Mem',
                'Self CUDA Mem',
            ])
    headers.append(
        '# of Calls'
    )
    # Only append Node ID if any event has a valid (>= 0) Node ID
    append_node_id = any([evt.node_id != -1 for evt in events])
    if append_node_id:
        headers.append('Node ID')

    # Have to use a list because nonlocal is Py3 only...
    SPACING_SIZE = 2
    row_format_lst = [""]
    header_sep_lst = [""]
    line_length_lst = [-SPACING_SIZE]
    MAX_STACK_ENTRY = 5

    def add_column(padding, text_dir='>'):
        row_format_lst[0] += '{: ' + text_dir + str(padding) + '}' + (' ' * SPACING_SIZE)
        header_sep_lst[0] += '-' * padding + (' ' * SPACING_SIZE)
        line_length_lst[0] += padding + SPACING_SIZE

    def auto_scale_flops(flops):
        flop_headers = [
            'FLOPS',
            'KFLOPS',
            'MFLOPS',
            'GFLOPS',
            'TFLOPS',
            'PFLOPS',
        ]
        assert flops > 0
        log_flops = max(0, min(math.log10(flops) / 3, float(len(flop_headers) - 1)))
        assert log_flops >= 0 and log_flops < len(flop_headers)
        return (pow(10, (math.floor(log_flops) * -3.0)), flop_headers[int(log_flops)])

    add_column(name_column_width)
    for _ in headers[1:]:
        add_column(DEFAULT_COLUMN_WIDTH)

    if has_input_shapes:
        headers.append('Input Shapes')
        add_column(shapes_column_width)

    if has_stack:
        headers.append('Source Location')
        add_column(src_column_width, text_dir='<')

    if with_flops:
        # Auto-scaling of flops header
        US_IN_SECOND = 1000.0 * 1000.0  # cpu_time_total is in us
        raw_flops = []
        for evt in events:
            if evt.flops > 0:
                if evt.cuda_time_total != 0:
                    evt.flops = float(evt.flops) / evt.cuda_time_total * US_IN_SECOND
                else:
                    evt.flops = float(evt.flops) / evt.cpu_time_total * US_IN_SECOND
                raw_flops.append(evt.flops)
        if len(raw_flops) != 0:
            (flops_scale, flops_header) = auto_scale_flops(min(raw_flops))
            headers.append(flops_header)
            add_column(flops_column_width)
        else:
            with_flops = False  # can't find any valid flops

    row_format = row_format_lst[0]
    header_sep = header_sep_lst[0]
    line_length = line_length_lst[0]
    add_column = None  # type: ignore

    # Have to use a list because nonlocal is Py3 only...
    result = []

    def append(s):
        result.append(s)
        result.append('\n')  # Yes, newline after the end as well

    sum_self_cpu_time_total = sum([event.self_cpu_time_total for event in events])
    sum_self_cuda_time_total = 0
    for evt in events:
        if evt.device_type == DeviceType.CPU:
            # in legacy profiler, kernel info is stored in cpu events
            if evt.is_legacy:
                sum_self_cuda_time_total += evt.self_cuda_time_total
        elif evt.device_type == DeviceType.CUDA:
            # in kineto profiler, there're events with the correct device type (e.g. CUDA)
            sum_self_cuda_time_total += evt.self_cuda_time_total

    # Actual printing
    if header is not None:
        append('=' * line_length)
        append(header)
    if top_level_events_only:
        append('=' * line_length)
        append('This report only display top-level ops statistics')
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
        if len(name) >= MAX_NAME_COLUMN_WIDTH - 3:
            name = name[:(MAX_NAME_COLUMN_WIDTH - 3)] + "..."
        row_values = [
            name,
            # Self CPU total %, 0 for async events.
            format_time_share(evt.self_cpu_time_total,
                              sum_self_cpu_time_total),
            evt.self_cpu_time_total_str,  # Self CPU total
            # CPU total %, 0 for async events.
            format_time_share(evt.cpu_time_total, sum_self_cpu_time_total) if not evt.is_async else 0,
            evt.cpu_time_total_str,  # CPU total
            evt.cpu_time_str,  # CPU time avg
        ]
        if has_cuda_time:
            row_values.extend([
                evt.self_cuda_time_total_str,
                # CUDA time total %
                format_time_share(evt.self_cuda_time_total, sum_self_cuda_time_total),
                evt.cuda_time_total_str,
                evt.cuda_time_str,  # Cuda time avg
            ])
        if profile_memory:
            row_values.extend([
                # CPU Mem Total
                format_memory(evt.cpu_memory_usage),
                # Self CPU Mem Total
                format_memory(evt.self_cpu_memory_usage),
            ])
            if has_cuda_mem:
                row_values.extend([
                    # CUDA Mem Total
                    format_memory(evt.cuda_memory_usage),
                    # Self CUDA Mem Total
                    format_memory(evt.self_cuda_memory_usage),
                ])
        row_values.append(
            evt.count,  # Number of calls
        )

        if append_node_id:
            row_values.append(evt.node_id)
        if has_input_shapes:
            row_values.append(str(evt.input_shapes)[:shapes_column_width])
        if with_flops:
            if evt.flops <= 0.0:
                row_values.append("--")
            else:
                row_values.append('{0:8.3f}'.format(evt.flops * flops_scale))
        if has_stack:
            src_field = ""
            if len(evt.stack) > 0:
                src_field = trim_path(evt.stack[0], src_column_width)
            row_values.append(src_field)
        append(row_format.format(*row_values))

        if has_stack:
            empty_headers = [""] * (len(headers) - 1)
            for entry in evt.stack[1:MAX_STACK_ENTRY]:
                append(row_format.format(*(empty_headers + [trim_path(entry, src_column_width)])))
            empty_headers.append("")
            append(row_format.format(*empty_headers))

    append(header_sep)
    append("Self CPU time total: {}".format(format_time(sum_self_cpu_time_total)))
    if has_cuda_time:
        append("Self CUDA time total: {}".format(format_time(sum_self_cuda_time_total)))
    return ''.join(result)
