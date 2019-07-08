import itertools
import torch

from collections import defaultdict, namedtuple
from operator import attrgetter


class EventList(list):
    """A list of Events (for pretty printing)"""
    def __init__(self, *args, **kwargs):
        super(EventList, self).__init__(*args, **kwargs)
        self._cpu_children_populated = False

    def __str__(self):
        return self.table()

    def populate_cpu_children(self):
        """Populates child events into each underlying FunctionEvent object.
        One event is a child of another if [s1, e1) is inside [s2, e2). Where
        s1 and e1 would be start and end of the child event's interval. And
        s2 and e2 start and end of the parent event's interval

        Example: In event list [[0, 10], [1, 3], [3, 4]] would have make [0, 10]
        be a parent of two other intervals.

        If for any reason two intervals intersect only partialy, this function
        will not record a parent child relationship between then.
        """
        if self.cpu_children_populated:
            return
        events = sorted(
            self,
            key=attrgetter("thread"),
        )
        threads = itertools.groupby(events, key=attrgetter("thread"))

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
            thread_events = sorted(
                thread_events,
                key=lambda event: [event.cpu_interval.start, -event.cpu_interval.end],
            )
            current_events = []
            cur_end = 0
            for event in thread_events:
                while len(current_events) > 0:
                    parent = current_events[-1]
                    if event.cpu_interval.start >= parent.cpu_interval.end or \
                            event.cpu_interval.end > parent.cpu_interval.end:
                        # this can't be a parent
                        current_events.pop()
                    else:
                        parent.append_cpu_child(event)
                        break

                current_events.append(event)

        self._cpu_children_populated = True

    @property
    def self_cpu_time_total(self):
        return sum([event.self_cpu_time_total for event in self])

    @property
    def cpu_children_populated(self):
        return self._cpu_children_populated

    def table(self, sort_by=None, row_limit=100, header=None):
        """Prints an EventList as a nicely formatted table.

        Arguments:
            sort_by (str, optional): Attribute used to sort entries. By default
                they are printed in the same order as they were registered.
                Valid keys include: ``cpu_time``, ``cuda_time``, ``cpu_time_total``,
                ``cuda_time_total``, ``count``.

        Returns:
            A string containing the table.
        """
        return build_table(
            self, sort_by=sort_by, row_limit=row_limit, header=header)

    def export_chrome_trace(self, path):
        """Exports an EventList as a Chrome tracing tools file.

        The checkpoint can be later loaded and inspected under ``chrome://tracing`` URL.

        Arguments:
            path (str): Path where the trace will be written.
        """
        import json
        with open(path, 'w') as f:
            chrome_events = []
            next_id = 0
            for evt in self:
                chrome_events.append(dict(
                    name=evt.name,
                    ph='X',
                    ts=evt.cpu_interval.start,
                    dur=evt.cpu_interval.elapsed_us(),
                    tid=evt.thread,
                    pid='CPU functions',
                    args={},
                ))
                for k in evt.kernels:
                    # 's' and 'f' draw Flow arrows from
                    # the CPU launch to the GPU kernel
                    chrome_events.append(dict(
                        name=evt.name,
                        ph='s',
                        ts=evt.cpu_interval.start,
                        tid=evt.thread,
                        pid='CPU functions',
                        id=next_id,
                        cat='cpu_to_cuda',
                        args={},
                    ))
                    chrome_events.append(dict(
                        name=k.name,
                        ph='f',
                        ts=k.interval.start,
                        tid=k.device,
                        pid='CUDA functions',
                        id=next_id,
                        cat='cpu_to_cuda',
                        args={},
                    ))
                    chrome_events.append(dict(
                        name=k.name,
                        ph='X',
                        ts=k.interval.start,
                        dur=k.interval.elapsed_us(),
                        tid=k.device,
                        pid='CUDA functions',
                        args={},
                    ))
                    next_id += 1

            json.dump(chrome_events, f)

    def key_averages(self, group_by_input_shapes=False):
        """Averages all function events over their keys.

        @param group_by_input_shapes The key would become
        (event name, input dimensions) rather than just event name.
        This is useful to see which dimensionality contributes to the runtime
        the most and may help with dimension specific optimizations or
        choosing best candidates for quantization (aka fitting a roof line)

        Returns:
            An EventList containing FunctionEventAvg objects.
        """
        self.populate_cpu_children()
        stats = defaultdict(FunctionEventAvg)

        def get_key(event, group_by_input_shapes):
            if not group_by_input_shapes:
                return event.key
            return (event.key, str(event.input_shapes))
        for evt in self:
            stats[get_key(evt, group_by_input_shapes)].add(
                evt, group_by_input_shapes)
        return EventList(stats.values())

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

    .. warning:
        This context managers should not be called recursively, i.e. at most one
        instance should be enabled at any given time.

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
    def __init__(self, enabled=True, use_cuda=False, record_shapes=False):
        self.enabled = enabled
        self.use_cuda = use_cuda
        self.function_events = None
        if not self.enabled:
            return
        self.entered = False
        self.record_shapes = record_shapes

    def __enter__(self):
        if not self.enabled:
            return
        if self.entered:
            raise RuntimeError("autograd profiler traces are not reentrant")
        self.entered = True
        profiler_kind = torch.autograd.ProfilerState.CUDA if self.use_cuda \
            else torch.autograd.ProfilerState.CPU
        torch.autograd._enable_profiler(
            torch.autograd.ProfilerConfig(profiler_kind, self.record_shapes))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.enabled:
            return
        records = torch.autograd._disable_profiler()
        self.function_events = EventList(parse_cpu_trace(records))
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
        self.function_events.populate_cpu_children()

    def table(self, sort_by=None, row_limit=100, header=None):
        self._check_finish()
        return self.function_events.table(
            sort_by=sort_by, row_limit=row_limit, header=header)
    table.__doc__ = EventList.table.__doc__

    def export_chrome_trace(self, path):
        self._check_finish()
        return self.function_events.export_chrome_trace(path)
    export_chrome_trace.__doc__ = EventList.export_chrome_trace.__doc__

    def key_averages(self, group_by_input_shape=False):
        self._check_finish()
        return self.function_events.key_averages(group_by_input_shape)
    key_averages.__doc__ = EventList.key_averages.__doc__

    def total_average(self):
        self._check_finish()
        return self.function_events.total_average()
    total_average.__doc__ = EventList.total_average.__doc__

    @property
    def self_cpu_time_total(self):
        """ Returns total time spent on CPU obtained as a sum of
        all self times across all the events.
        """
        self._check_finish()
        return self.function_events.self_cpu_time_total


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
                self.record_shapes
            )
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
        assert(time_us == 0)
        return "NaN"
    return '{:.2f}%'.format(time_us * 100.0 / total_time_us)


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

    @property
    def cpu_time(self):
        return 0.0 if self.count == 0 else 1.0 * self.cpu_time_total / self.count

    @property
    def cuda_time(self):
        return 0.0 if self.count == 0 else 1.0 * self.cuda_time_total / self.count


class Interval(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def elapsed_us(self):
        return self.end - self.start


Kernel = namedtuple('Kernel', ['name', 'device', 'interval'])


# TODO: record TID too
class FunctionEvent(FormattedTimesMixin):
    """Profiling information about a single function."""
    def __init__(self, id, name, thread, cpu_start, cpu_end, input_shapes=None):
        self.id = id
        self.name = name
        self.cpu_interval = Interval(cpu_start, cpu_end)
        self.thread = thread
        self.kernels = []
        self.count = 1
        self.cpu_children = []
        self.input_shapes = input_shapes

    def append_kernel(self, name, device, start, end):
        self.kernels.append(Kernel(name, device, Interval(start, end)))

    def append_cpu_child(self, child):
        """Append a CPU child of type FunctionEvent.

        One is supposed to append only dirrect children to the event to have
        correct self cpu time being reported.
        """
        assert(isinstance(child, FunctionEvent))
        self.cpu_children.append(child)

    @property
    def self_cpu_time_total(self):
        return self.cpu_time_total - sum(
            [child.cpu_time_total for child in self.cpu_children]
        )

    @property
    def cuda_time_total(self):
        return sum(kinfo.interval.elapsed_us() for kinfo in self.kernels)

    @property
    def cpu_time_total(self):
        return self.cpu_interval.elapsed_us()

    @property
    def key(self):
        return self.name

    def __repr__(self):
        return (
            '<FunctionEvent id={} cpu_time={} cpu_start={} cpu_end={} '
            'cpu_children={} cuda_time={} name={} thread={} input_shapes={}>'.format(
                self.id,
                self.cpu_time_str,
                self.cpu_interval.start,
                self.cpu_interval.end,
                str([child.id for child in self.cpu_children]),
                self.cuda_time_str,
                self.name,
                self.thread,
                str(self.input_shapes),
            )
        )


class FunctionEventAvg(FormattedTimesMixin):
    """Used to average stats over multiple FunctionEvent objects."""
    def __init__(self):
        self.key = None
        self.count = 0
        self.cpu_time_total = 0
        self.cuda_time_total = 0
        self.self_cpu_time_total = 0
        self.input_shapes = None

    def add(self, other, group_by_input_shapes=False):
        if self.key is None:
            self.key = other.key
            if group_by_input_shapes:
                self.input_shapes = other.input_shapes

        assert (
            not group_by_input_shapes or
            other.input_shapes == self.input_shapes
        )
        assert isinstance(other, FunctionEvent)
        assert other.key == self.key
        self.cpu_time_total += other.cpu_time
        self.cuda_time_total += other.cuda_time
        self.self_cpu_time_total += other.self_cpu_time_total
        self.count += 1
        return self

    def __repr__(self):
        return (
            '<FunctionEventAvg key={} self_cpu_time={} cpu_time={} '
            'cuda_time={} input_shapes={}>'.format(
                self.key,
                self.self_cpu_time_total_str,
                self.cpu_time_str,
                self.cuda_time_str,
                str(self.input_shapes),
            )
        )


################################################################################
# Utilities

class StringTable(defaultdict):
    def __missing__(self, key):
        self[key] = torch._C._demangle(key)
        return self[key]


################################################################################
# CPU checkpoints

def parse_cpu_trace(thread_records):
    next_id = 0
    start_record = None
    cuda_records = {}
    functions = []
    record_stack = []
    string_table = StringTable()

    # cuda start events and the overall profiler start event don't happen
    # at exactly the same time because we need to record an event on each device
    # and each record takes ~4us. So we adjust here by the difference
    # adding the difference in CPU time between the profiler start event
    # and the CPU time of the cuda start event for the device
    def adjusted_time(cuda_record):
        assert cuda_record.device() != -1
        cuda_time_0 = cuda_records[cuda_record.device()]
        return cuda_time_0.cuda_elapsed_us(cuda_record) + start_record.cpu_elapsed_us(cuda_time_0)

    # '__start_profile' is not guarenteed to be first, so we must find it here
    for record in itertools.chain(*thread_records):
        if record.name() == '__start_profile':
            start_record = record
        elif record.name() == '__cuda_start_event':
            assert record.device() != -1
            cuda_records[record.device()] = record
    assert start_record is not None

    for record in itertools.chain(*thread_records):
        if record.kind() == 'mark':
            continue
        elif record.kind() == 'push':
            record_stack.append((next_id, record))
            next_id += 1
        elif record.kind() == 'pop':
            function_id, start = record_stack.pop()
            fe = FunctionEvent(
                id=function_id,
                name=string_table[start.name()],
                thread=start.thread_id(),
                cpu_start=start_record.cpu_elapsed_us(start),
                cpu_end=start_record.cpu_elapsed_us(record),
                input_shapes=start.shapes())
            if start.has_cuda():
                cuda_start = adjusted_time(start)
                cuda_end = adjusted_time(record)
                fe.append_kernel(start.name(),
                                 start.device(),
                                 cuda_start,
                                 cuda_end)
            functions.append(fe)

    functions.sort(key=lambda evt: evt.cpu_interval.start)
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
        assert row['cbid'] == 13  # 13 == Launch
        evt = functions_map[row['marker_id']]
        evt.append_kernel(row['kernel_name'],
                          0,
                          row['kernel_start'],
                          row['kernel_end'])

    functions.sort(key=lambda evt: evt.cpu_interval.start)
    return functions


################################################################################
# Pretty printer


def build_table(events, sort_by=None, header=None, row_limit=100):
    """Prints a summary of events (which can be a list of FunctionEvent or FunctionEventAvg)."""
    if len(events) == 0:
        return ""

    if sort_by is not None:
        events = EventList(sorted(
            events, key=lambda evt: getattr(evt, sort_by), reverse=True
        ))

    has_input_shapes = any(
        [event.input_shapes is not None for event in events])
    name_column_width = max([len(evt.key) for evt in events]) + 4
    DEFAULT_COLUMN_WIDTH = 15
    SHAPES_COLUMN_WIDTH = 35

    headers = [
        'Name',
        'Self CPU total %',
        'Self CPU total',
        'CPU total %',
        'CPU total',
        'CPU time avg',
        'CUDA total %',
        'CUDA total',
        'CUDA time avg',
        'Number of Calls',
    ]

    # Have to use a list because nonlocal is Py3 only...
    SPACING_SIZE = 2
    row_format = [""]
    header_sep = [""]
    line_length = [-SPACING_SIZE]

    def add_column(padding):
        row_format[0] += '{: <' + str(padding) + '}  '
        header_sep[0] += '-' * padding + '  '
        line_length[0] += padding + SPACING_SIZE

    add_column(name_column_width)
    for _ in headers[1:]:
        add_column(DEFAULT_COLUMN_WIDTH)

    if has_input_shapes:
        headers.append('Input Shapes')
        add_column(SHAPES_COLUMN_WIDTH)

    row_format = row_format[0]
    header_sep = header_sep[0]
    line_length = line_length[0]
    add_column = None

    # Have to use a list because nonlocal is Py3 only...
    result = []

    def append(s):
        result.append(s)
        result.append('\n')  # Yes, newline after the end as well

    self_cpu_time_total = sum([event.self_cpu_time_total for event in events])
    cuda_time_total = sum([evt.cuda_time_total for evt in events])
    # Actual printing
    if header is not None:
        append('=' * line_length)
        append(header)
    append(header_sep)
    append(row_format.format(*headers))

    append(header_sep)
    for evt in events[:row_limit]:
        row_values = [
            evt.key,  # Name
            # Self CPU total %
            format_time_share(evt.self_cpu_time_total,
                              self_cpu_time_total),
            evt.self_cpu_time_total_str,  # Self CPU total
            # CPU total %
            format_time_share(evt.cpu_time_total, self_cpu_time_total),
            evt.cpu_time_total_str,  # CPU total
            evt.cpu_time_str,  # CPU time avg
            # CUDA time total %
            format_time_share(evt.cuda_time_total, cuda_time_total),
            evt.cuda_time_total_str,
            evt.cuda_time_str,  # Cuda time avg
            evt.count,  # Number of calls
        ]
        if has_input_shapes:
            row_values.append(str(evt.input_shapes)[:SHAPES_COLUMN_WIDTH])
        append(row_format.format(*row_values))

    append(header_sep)
    append("Self CPU time total: {}".format(format_time(self_cpu_time_total)))
    append("CUDA time total: {}".format(format_time(cuda_time_total)))
    return ''.join(result)
