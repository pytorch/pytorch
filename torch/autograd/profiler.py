import subprocess
import re
import os
import sys
import itertools
from collections import defaultdict

import torch
from torch._six import FileNotFoundError


class range(object):
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        torch.autograd._push_range(self.name)

    def __exit__(self, *args):
        torch.autograd._pop_range()
        return False


class EventList(list):
    """A list of Events (for pretty printing)"""
    def __init__(self, *args, **kwargs):
        super(EventList, self).__init__(*args, **kwargs)

    def __str__(self):
        return self.table()

    def table(self, sort_by=None):
        """Prints an EventList as a nicely formatted table.

        Arguments:
            sort_by (str, optional): Attribute used to sort entries. By default
                they are printed in the same order as they were registered.
                Valid keys include: ``cpu_time``, ``cuda_time``, ``cpu_time_total``,
                ``cuda_time_total``, ``count``.

        Returns:
            A string containing the table.
        """
        return build_table(self, sort_by)

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

    def key_averages(self):
        """Averages all function events over their keys.

        Returns:
            An EventList containing FunctionEventAvg objects.
        """
        stats = defaultdict(FunctionEventAvg)
        for evt in self:
            stats[evt.key] += evt
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

    Arguments:
        enabled (bool, optional): Setting this to False makes this context manager a no-op.
            Default: ``True``.

        use_cuda (bool, optional): Enables timing of CUDA events as well using the cudaEvent API.
            Adds approximately 4us of overhead to each tensor operation.
            Default: ``False``

    .. warning:
        This context managers should not be called recursively, i.e. at most one
        instance should be enabled at any given time.

    Example:
        >>> x = torch.randn((1, 1), requires_grad=True)
        >>> with torch.autograd.profiler.profile() as prof:
        ...     y = x ** 2
        ...     y.backward()
        >>> # NOTE: some columns were removed for brevity
        ... print(prof)
        -------------------------------------  ---------------  ---------------
        Name                                          CPU time        CUDA time
        -------------------------------------  ---------------  ---------------
        PowConstant                                  142.036us          0.000us
        N5torch8autograd9GraphRootE                   63.524us          0.000us
        PowConstantBackward                          184.228us          0.000us
        MulConstant                                   50.288us          0.000us
        PowConstant                                   28.439us          0.000us
        Mul                                           20.154us          0.000us
        N5torch8autograd14AccumulateGradE             13.790us          0.000us
        N5torch8autograd5CloneE                        4.088us          0.000us
    """

    def __init__(self, enabled=True, use_cuda=False):
        self.enabled = enabled
        self.use_cuda = use_cuda
        self.function_events = None
        if not self.enabled:
            return
        self.entered = False

    def __enter__(self):
        if not self.enabled:
            return
        if self.entered:
            raise RuntimeError("autograd profiler traces are not reentrant")
        self.entered = True
        profiler_kind = torch.autograd.ProfilerState.CUDA if self.use_cuda \
            else torch.autograd.ProfilerState.CPU
        torch.autograd._enable_profiler(profiler_kind)
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

    def table(self, sort_by=None):
        self._check_finish()
        return self.function_events.table(sort_by)
    table.__doc__ = EventList.table.__doc__

    def export_chrome_trace(self, path):
        self._check_finish()
        return self.function_events.export_chrome_trace(path)
    export_chrome_trace.__doc__ = EventList.export_chrome_trace.__doc__

    def key_averages(self):
        self._check_finish()
        return self.function_events.key_averages()
    key_averages.__doc__ = EventList.key_averages.__doc__

    def total_average(self):
        self._check_finish()
        return self.function_events.total_average()
    total_average.__doc__ = EventList.total_average.__doc__


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
        enabled (bool, optional): Setting this to False makes this context manager a no-op.
            Default: ``True``.

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
    Thus, the `seq=<N>` annotation associated with each forward function range tells you that
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
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.entered = False

    def __enter__(self):
        if not self.enabled:
            return
        if self.entered:
            raise RuntimeError("NVTX annotation context manager is not reentrant")
        self.entered = True
        torch.cuda.synchronize()
        torch.autograd._enable_profiler(torch.autograd.ProfilerState.NVTX)
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
    return '{:.3f}us'.format(time_us)


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


class Kernel(object):
    def __init__(self, name, device, interval):
        self.name = name
        self.device = device
        self.interval = interval


# TODO: record TID too
class FunctionEvent(FormattedTimesMixin):
    """Profiling information about a single function."""
    def __init__(self, id, name, thread, cpu_start, cpu_end):
        self.id = id
        self.name = name
        self.cpu_interval = Interval(cpu_start, cpu_end)
        self.thread = thread
        self.kernels = []
        self.count = 1

    def append_kernel(self, name, device, start, end):
        self.kernels.append(Kernel(name, device, Interval(start, end)))

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
        return '<FunctionEvent id={} cpu_time={} cuda_time={} name={} thread={}>'.format(
            self.id, self.cpu_time_str, self.cuda_time_str, self.name, self.thread)


class FunctionEventAvg(FormattedTimesMixin):
    """Used to average stats over multiple FunctionEvent objects."""
    def __init__(self):
        self.key = None
        self.count = self.cpu_time_total = self.cuda_time_total = 0

    def __iadd__(self, other):
        if self.key is None:
            self.key = other.key
        assert isinstance(other, FunctionEvent)
        assert other.key == self.key
        self.cpu_time_total += other.cpu_time
        self.cuda_time_total += other.cuda_time
        self.count += 1
        return self

    def __repr__(self):
        return '<FunctionEventAvg cpu_time={} cuda_time={} key={}>'.format(
            self.cpu_time_str, self.cuda_time_str, self.key)


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
                cpu_end=start_record.cpu_elapsed_us(record))
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

def build_table(events, sort_by=None, header=None):
    """Prints a summary of events (which can be a list of FunctionEvent or FunctionEventAvg)."""
    if sort_by is not None:
        events = sorted(events, key=lambda evt: getattr(evt, sort_by))

    name_lengths = [len(evt.key) for evt in events]
    if len(name_lengths) == 0:
        return ""
    max_name_length = max(name_lengths)
    max_name_length += 4  # Add some nice padding
    col_width = 15
    col_format = '  {: >' + str(col_width) + '}'
    row_format = '{: <' + str(max_name_length) + '}' + col_format * 5
    header_sep = '-' * max_name_length + ('  ' + '-' * col_width) * 5

    # Have to use a list because nonlocal is Py3 only...
    result = []

    def append(s):
        result.append(s)
        result.append('\n')  # Yes, newline after the end as well

    # Actual printing
    if header is not None:
        line_length = max_name_length + (col_width + 2) * 5
        append('=' * line_length)
        append(header)
    append(header_sep)
    append(row_format.format('Name', 'CPU time', 'CUDA time', 'Calls', 'CPU total', 'CUDA total'))
    append(header_sep)
    for evt in events:
        append(row_format.format(evt.key, evt.cpu_time_str, evt.cuda_time_str,
                                 evt.count, evt.cpu_time_total_str, evt.cuda_time_total_str))

    return ''.join(result)
