import itertools
from typing import Any
import torch
from torch.futures import Future

from collections import defaultdict, namedtuple
from operator import attrgetter

from typing import List, Dict, Tuple, Optional

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
        super(EventList, self).__init__(*args, **kwargs)
        self._cpu_children_populated = False
        self._use_cuda = use_cuda
        self._profile_memory = profile_memory

    def __str__(self):
        return self.table()

    def populate_cpu_children(self):
        """Populates child events into each underlying FunctionEvent object.
        One event is a child of another if [s1, e1) is inside [s2, e2). Where
        s1 and e1 would be start and end of the child event's interval. And
        s2 and e2 start and end of the parent event's interval

        Example: In event list [[0, 10], [1, 3], [3, 4]] would have make [0, 10]
        be a parent of two other intervals.

        If for any reason two intervals intersect only partially, this function
        will not record a parent child relationship between then.
        """
        if self.cpu_children_populated:
            return

        # Some events can be async (i.e. start and end on different threads),
        # since it's generally undefined how to attribute children ranges to
        # async ranges, we do not use them when calculating nested ranges and stats
        sync_events = [evt for evt in self if not evt.is_async]
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
                key=lambda event: [event.cpu_interval.start, -event.cpu_interval.end],
            )
            current_events: List[FunctionEvent] = []
            cur_end = 0
            for event in thread_events_:
                while len(current_events) > 0:
                    parent = current_events[-1]
                    if event.cpu_interval.start >= parent.cpu_interval.end or \
                            event.cpu_interval.end > parent.cpu_interval.end:
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

        self._cpu_children_populated = True

    def set_backward_stacktraces(self):
        self.populate_cpu_children()

        def bw_parent(evt):
            if evt is None:
                return None
            elif evt.scope == 1:
                return evt
            else:
                return bw_parent(evt.cpu_parent)

        fwd_stacks = {}
        for evt in self:
            if bw_parent(evt) is None:
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

    @property
    def cpu_children_populated(self):
        return self._cpu_children_populated

    def table(self, sort_by=None, row_limit=100, header=None, top_level_events_only=False):
        """Prints an EventList as a nicely formatted table.

        Arguments:
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
            header=header,
            use_cuda=self._use_cuda,
            profile_memory=self._profile_memory,
            top_level_events_only=top_level_events_only)

    def export_chrome_trace(self, path):
        """Exports an EventList as a Chrome tracing tools file.

        The checkpoint can be later loaded and inspected under ``chrome://tracing`` URL.

        Arguments:
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
                f.write(
                    '{"name": "%s", '
                    '"ph": "X", '
                    '"ts": %s, '
                    '"dur": %s, '
                    '"tid": %s, '
                    '"pid": "CPU functions", '
                    '"args": {}}, '
                    % (
                        evt.name,
                        evt.cpu_interval.start,
                        evt.cpu_interval.elapsed_us(),
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
                            '"args": {}}, ' % (evt.name, evt.cpu_interval.start,
                                               evt.thread, next_id))
                    f.write('{"name": "%s", '
                            '"ph": "f", '
                            '"ts": %s, '
                            '"tid": %s, '
                            '"pid": "CUDA functions", '
                            '"id": %s, '
                            '"cat": "cpu_to_cuda", '
                            '"args": {}}, ' % (k.name, k.interval.start, k.device, next_id))
                    f.write('{"name": "%s", '
                            '"ph": "X", '
                            '"ts": %s, '
                            '"dur": %s, '
                            '"tid": %s, '
                            '"pid": "CUDA functions", '
                            '"args": {}}, ' % (k.name, k.interval.start,
                                               k.interval.elapsed_us(), k.device))
                    next_id += 1

            # remove trailing whitespace and comma
            f.seek(f.tell() - 2, os.SEEK_SET)
            f.truncate()
            f.write("]")

    def key_averages(self, group_by_input_shapes=False, group_by_stack_n=0):
        """Averages all function events over their keys.

        Arguments:
            group_by_input_shapes: group entries by
            (event name, input shapes) rather than just event name.
            This is useful to see which input shapes contribute to the runtime
            the most and may help with size-specific optimizations or
            choosing the best candidates for quantization (aka fitting a roof line)

            group_by_stack_n: group by top n stack trace entries

        Returns:
            An EventList containing FunctionEventAvg objects.
        """
        self.populate_cpu_children()
        stats: Dict[Tuple[int, Tuple[int, int]], FunctionEventAvg] = defaultdict(FunctionEventAvg)

        def get_key(event, group_by_input_shapes, group_by_stack_n):
            key = [str(event.key), str(event.node_id)]
            if group_by_input_shapes:
                key.append(str(event.input_shapes))
            if group_by_stack_n > 0:
                key += event.stack[:group_by_stack_n]
            return tuple(key)
        for evt in self:
            stats[get_key(evt, group_by_input_shapes, group_by_stack_n)].add(evt)

        avg_list = EventList(stats.values(), use_cuda=self._use_cuda, profile_memory=self._profile_memory)
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

    Arguments:
        enabled (bool, optional): Setting this to False makes this context manager a no-op.
            Default: ``True``.

        use_cuda (bool, optional): Enables timing of CUDA events as well using the cudaEvent API.
            Adds approximately 4us of overhead to each tensor operation.
            Default: ``False``

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

        profile_memory (bool, optional): Whether to report memory usage, default: ``False``

        with_stack (bool, optional): record source information (file and line number) for the ops

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
            use_cuda=False,
            record_shapes=False,
            profile_memory=False,
            with_stack=False):
        self.enabled = enabled
        self.use_cuda = use_cuda
        self.function_events = None
        if not self.enabled:
            return
        self.entered = False
        self.record_shapes = record_shapes
        self.profile_memory = profile_memory
        self.with_stack = with_stack

    def __enter__(self):
        if not self.enabled:
            return
        if self.entered:
            raise RuntimeError("autograd profiler traces are not reentrant")
        self.entered = True
        profiler_kind = torch.autograd.ProfilerState.CUDA if self.use_cuda \
            else torch.autograd.ProfilerState.CPU

        config = torch.autograd.ProfilerConfig(
            profiler_kind,
            self.record_shapes,
            self.profile_memory,
            self.with_stack)
        torch.autograd._enable_profiler(config)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.enabled:
            return
        records = torch.autograd._disable_profiler()
        self.function_events = EventList(
            parse_event_records(records),
            use_cuda=self.use_cuda,
            profile_memory=self.profile_memory)
        if self.with_stack:
            self.function_events.set_backward_stacktraces()
        return False

    def __repr__(self):
        if self.function_events is None:
            return '<unfinished torch.autograd.profile>'
        return repr(self.function_events)

    def __str__(self):
        if self.function_events is None:
            return '<unfinished torch.autograd.profile>'
        self.function_events.populate_cpu_children()
        return str(self.function_events)

    def _check_finish(self):
        if self.function_events is None:
            raise RuntimeError("can't export a trace that didn't finish running")
        self.function_events.populate_cpu_children()

    def table(self, sort_by=None, row_limit=100, header=None, top_level_events_only=False):
        self._check_finish()
        assert self.function_events is not None
        return self.function_events.table(
            sort_by=sort_by, row_limit=row_limit, header=header,
            top_level_events_only=top_level_events_only
        )
    table.__doc__ = EventList.table.__doc__

    def export_chrome_trace(self, path):
        self._check_finish()
        assert self.function_events is not None
        return self.function_events.export_chrome_trace(path)
    export_chrome_trace.__doc__ = EventList.export_chrome_trace.__doc__

    def key_averages(self, group_by_input_shape=False, group_by_stack_n=0):
        self._check_finish()
        assert self.function_events is not None
        return self.function_events.key_averages(group_by_input_shape, group_by_stack_n)
    key_averages.__doc__ = EventList.key_averages.__doc__

    def total_average(self):
        self._check_finish()
        assert self.function_events is not None
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

    Arguments:
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

        Arguments:
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

    Arguments:
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
        torch.autograd._enable_profiler(
            torch.autograd.ProfilerConfig(
                torch.autograd.ProfilerState.NVTX,
                self.record_shapes,
                False,
                False)
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.enabled:
            return
        torch.cuda.synchronize()
        torch.autograd._disable_profiler()
        return False


def load_nvprof(path):
    """Opens an nvprof trace file and parses autograd annotations.

    Arguments:
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


Kernel = namedtuple('Kernel', ['name', 'device', 'interval'])


class FunctionEvent(FormattedTimesMixin):
    """Profiling information about a single function."""
    def __init__(
            self, id, node_id, name, thread, cpu_start, cpu_end, fwd_thread=None, input_shapes=None,
            stack=None, scope=0, cpu_memory_usage=0, cuda_memory_usage=0, is_async=False,
            is_remote=True, sequence_nr=-1):
        self.id: int = id
        self.node_id: int = node_id
        self.name: str = name
        self.cpu_interval: Interval = Interval(cpu_start, cpu_end)
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

    def append_kernel(self, name, device, start, end):
        self.kernels.append(Kernel(name, device, Interval(start, end)))

    def append_cpu_child(self, child):
        """Append a CPU child of type FunctionEvent.

        One is supposed to append only direct children to the event to have
        correct self cpu time being reported.
        """
        assert(isinstance(child, FunctionEvent))
        self.cpu_children.append(child)

    def set_cpu_parent(self, parent):
        """Set the immediate CPU parent of type FunctionEvent

        One profiling FunctionEvent should have only one CPU parent such that
        the child's range interval is completely inside the parent's. We use
        this connection to determine the event is from top-level op or not.
        """
        assert(isinstance(parent, FunctionEvent))
        self.cpu_parent = parent

    # Note: async events don't have children, are not used when computing 'self'
    # metrics of other events, have only total cpu time
    @property
    def self_cpu_memory_usage(self):
        if self.is_async:
            return 0
        return self.cpu_memory_usage - sum(
            [child.cpu_memory_usage for child in self.cpu_children]
        )

    @property
    def self_cuda_memory_usage(self):
        if self.is_async:
            return 0
        return self.cuda_memory_usage - sum(
            [child.cuda_memory_usage for child in self.cpu_children]
        )

    @property
    def self_cpu_time_total(self):
        if self.is_async:
            return 0
        return self.cpu_time_total - sum(
            [child.cpu_time_total for child in self.cpu_children]
        )

    @property
    def cuda_time_total(self):
        return sum(kinfo.interval.elapsed_us() for kinfo in self.kernels)

    @property
    def self_cuda_time_total(self):
        return sum(kinfo.interval.elapsed_us() for kinfo in self.kernels) - \
            sum([child.cuda_time_total for child in self.cpu_children])

    @property
    def cpu_time_total(self):
        return self.cpu_interval.elapsed_us()

    @property
    def key(self):
        return self.name

    def __repr__(self):
        return (
            '<FunctionEvent id={} node_id={} cpu_time={} cpu_start={} cpu_end={} '
            'cpu_children={} cuda_time={} name={} thread={} input_shapes={} '
            'cpu_memory_usage={} cuda_memory_usage={} is_async={} is_remote={} seq_nr={}>'.format(
                self.id,
                self.node_id,
                self.cpu_time_str,
                self.cpu_interval.start,
                self.cpu_interval.end,
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

def parse_event_records(thread_records):
    def get_record_key(record):
        """
        Returns a tuple to be used by parse_event_records for correlating start and
        end records.
        """
        return (record.handle(), record.node_id())

    next_id = 0
    start_record = None
    cuda_records = {}
    functions = []
    record_stack = []
    string_table = StringTable()

    # ignoring the following utility ops
    filtered_out_names = [
        "profiler::_record_function_enter",
        "profiler::_record_function_exit",
        "aten::is_leaf",
        "aten::output_nr",
        "aten::_version",
    ]

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

    # cuda start events and the overall profiler start event don't happen
    # at exactly the same time because we need to record an event on each device
    # and each record takes ~4us. So we adjust here by the difference
    # adding the difference in CPU time between the profiler start event
    # and the CPU time of the cuda start event for the device
    def adjusted_time(cuda_record, cuda_records_map):
        assert cuda_record.device() != -1
        assert start_record is not None
        cuda_time_0 = cuda_records_map[(cuda_record.node_id(), cuda_record.device())]
        return cuda_time_0.cuda_elapsed_us(cuda_record) + start_record.cpu_elapsed_us(cuda_time_0)

    # '__start_profile' is not guaranteed to be first, so we must find it here
    for record in itertools.chain(*thread_records):
        name = record.name()
        if start_record is None and name == '__start_profile':
            start_record = record
        elif '__cuda_start_event' in name:
            # N.B.: Each CUDA device has its own __cuda_start_event.
            assert record.device() != -1
            # key for cuda_records is (node_id, device) in case of multiple nodes
            # having the same device
            cuda_records[(record.node_id(), record.device())] = record

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
            if (record.name() in filtered_out_names or
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

                fe = FunctionEvent(
                    id=record.handle(),
                    node_id=record.node_id(),
                    name=string_table[start.name()],
                    thread=start.thread_id(),
                    cpu_start=start_record.cpu_elapsed_us(start),
                    cpu_end=start_record.cpu_elapsed_us(record),
                    fwd_thread=start.fwd_thread_id(),
                    input_shapes=start.shapes(),
                    stack=[entry for entry in start.stack() if filter_stack_entry(entry)],
                    scope=start.scope(),
                    cpu_memory_usage=cpu_memory_usage,
                    cuda_memory_usage=cuda_memory_usage,
                    is_async=is_async,
                    is_remote=is_remote_event,
                    sequence_nr=start.sequence_nr(),
                )
                # note: async events have only cpu total time
                if not is_async and start.has_cuda():
                    cuda_start = adjusted_time(start, cuda_records)
                    cuda_end = adjusted_time(record, cuda_records)
                    if (cuda_end - cuda_start) > 0:
                        fe.append_kernel(
                            start.name(),
                            start.device(),
                            cuda_start,
                            cuda_end)
                functions.append(fe)
                del range_starts[record_key]
                del cpu_memory_allocs[record_key]
                del cuda_memory_allocs[record_key]
            elif record.kind() == 'memory_alloc':
                for handle in cpu_memory_allocs.keys():
                    cpu_memory_allocs[handle] += record.cpu_memory_usage()
                for handle in cuda_memory_allocs.keys():
                    cuda_memory_allocs[handle] += record.cuda_memory_usage()
            prev_record = record

    # Sort functions by start time then by end time ascending.
    # This ensures that--in the case of nested events which
    # have the same start time (which may happen due to the
    # granularity of the given clock tick)--we always show
    # the outermost nested call first. This adds stability
    # in how FunctionEvents appear
    functions.sort(key=lambda evt: [evt.cpu_interval.start, -evt.cpu_interval.end])
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
                            cpu_start=row['start_time'],
                            cpu_end=row['end_time'],
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
                          row['kernel_start'],
                          row['kernel_end'])

    functions.sort(key=lambda evt: evt.cpu_interval.start)
    return functions


################################################################################
# Pretty printer


def build_table(
        events,
        sort_by=None,
        header=None,
        row_limit=100,
        use_cuda=True,
        profile_memory=False,
        top_level_events_only=False):
    """Prints a summary of events (which can be a list of FunctionEvent or FunctionEventAvg)."""
    if len(events) == 0:
        return ""

    if sort_by is not None:
        events = EventList(sorted(
            events, key=lambda evt: getattr(evt, sort_by), reverse=True
        ), use_cuda=use_cuda, profile_memory=profile_memory)

    has_input_shapes = any(
        [(event.input_shapes is not None and len(event.input_shapes) > 0) for event in events])

    name_column_width = max([len(evt.key) for evt in events]) + 4

    DEFAULT_COLUMN_WIDTH = 12

    shapes_column_width = max([len(str(evt.input_shapes)) for evt in events]) + 4
    shapes_column_width = min(shapes_column_width, 45)

    src_column_width = None
    stacks = []
    for evt in events:
        if evt.stack is not None and len(evt.stack) > 0:
            stacks.append(evt.stack)
    has_stack = len(stacks) > 0
    if has_stack:
        src_column_width = max([max([len(entry) for entry in stack]) for stack in stacks]) + 4
        src_column_width = min(src_column_width, 75)

    headers = [
        'Name',
        'Self CPU %',
        'Self CPU',
        'CPU total %',
        'CPU total',
        'CPU time avg',
    ]
    if use_cuda:
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
        if torch.cuda.is_available():
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

    add_column(name_column_width)
    for _ in headers[1:]:
        add_column(DEFAULT_COLUMN_WIDTH)

    if has_input_shapes:
        headers.append('Input Shapes')
        add_column(shapes_column_width)

    if has_stack:
        headers.append('Source Location')
        add_column(src_column_width, text_dir='<')

    row_format = row_format_lst[0]
    header_sep = header_sep_lst[0]
    line_length = line_length_lst[0]
    add_column = None  # type: ignore

    # Have to use a list because nonlocal is Py3 only...
    result = []

    def append(s):
        result.append(s)
        result.append('\n')  # Yes, newline after the end as well

    self_cpu_time_total = sum([event.self_cpu_time_total for event in events])
    cuda_time_total = sum([evt.self_cuda_time_total for evt in events])
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

    event_limit = 0
    for evt in events:
        if event_limit == row_limit:
            break
        if top_level_events_only and evt.cpu_parent is not None:
            continue
        else:
            event_limit += 1
        row_values = [
            evt.key,  # Name
            # Self CPU total, 0 for async events. %
            format_time_share(evt.self_cpu_time_total,
                              self_cpu_time_total),
            evt.self_cpu_time_total_str,  # Self CPU total
            # CPU total %, 0 for async events.
            format_time_share(evt.cpu_time_total, self_cpu_time_total) if not evt.is_async else 0,
            evt.cpu_time_total_str,  # CPU total
            evt.cpu_time_str,  # CPU time avg
        ]
        if use_cuda:
            row_values.extend([
                evt.self_cuda_time_total_str,
                # CUDA time total %
                format_time_share(evt.self_cuda_time_total, cuda_time_total),
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
            if torch.cuda.is_available():
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
        if has_stack:
            src_field = ""
            if len(evt.stack) > 0:
                src_field = evt.stack[0][:src_column_width]
            row_values.append(src_field)
        append(row_format.format(*row_values))

        if has_stack:
            empty_headers = [""] * (len(headers) - 1)
            for entry in evt.stack[1:MAX_STACK_ENTRY]:
                append(row_format.format(*(empty_headers + [entry[:src_column_width]])))
            empty_headers.append("")
            append(row_format.format(*empty_headers))

    append(header_sep)
    append("Self CPU time total: {}".format(format_time(self_cpu_time_total)))
    if use_cuda:
        append("CUDA time total: {}".format(format_time(cuda_time_total)))
    return ''.join(result)
