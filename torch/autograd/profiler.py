import torch
import subprocess
import os
import sys
import copy
import tempfile
import itertools
from collections import defaultdict, namedtuple


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
            for evt in self:
                chrome_events.append(dict(
                    name=evt.name,
                    ph='X',
                    ts=evt.start / 1000,
                    dur=evt.cpu_time_total / 1000,
                    tid='Autograd functions',
                    pid='Autograd functions',
                    args={},
                ))
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
            Default: True.

    .. warning:
        This context managers should not be called recursively, i.e. at most one
        instance should be enabled at any given time.

    Example:
        >>> x = Variable(torch.randn(1, 1), requires_grad=True)
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

    def __init__(self, enabled=True):
        self.enabled = enabled
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
        torch.autograd._enable_profiler(False)
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

    def export_chrome_trace(self, path):
        if self.function_events is None:
            raise RuntimeError("can't export a trace that didn't finish running")
        return self.function_events.export_chrome_trace(path)
    export_chrome_trace.__doc__ = EventList.export_chrome_trace.__doc__

    def key_averages(self):
        if self.function_events is None:
            raise RuntimeError("can't average a trace that didn't finish running")
        return self.function_events.key_averages()
    key_averages.__doc__ = EventList.key_averages.__doc__

    def total_average(self):
        if self.function_events is None:
            raise RuntimeError("can't average a trace that didn't finish running")
        return self.function_events.total_average()
    total_average.__doc__ = EventList.total_average.__doc__


class emit_nvtx(object):
    """Context manager that makes every autograd operation emit an NVTX range.

    It is useful when running the program under nvprof. Unfortunately, there's no
    way to force nvprof to flush the data it collected to disk, so for CUDA profiling
    one has to use this context manager to annotate nvprof traces, and then use
    :func:`torch.autograd.profiler.open_nvtx` to analyze the checkpoint.

    .. warning:
        This context managers should not be called recursively, i.e. at most one
        instance should be enabled at any given time.

    Arguments:
        enabled (bool, optional): Setting this to False makes this context manager a no-op.
            Default: True.

    Example:
        >>> with torch.cuda.profiler.profile():
        ...     model(x) # Warmup CUDA memory allocator and profiler
        ...     with torch.autograd.profiler.emit_nvtx():
        ...         model(x)
    """
    def __init__(self, enabled=True):
        self.enabled = True
        self.entered = False

    def __enter__(self):
        if not self.enabled:
            return
        if self.entered:
            raise RuntimeError("NVTX annotation context manager is not reentrant")
        self.entered = True
        torch.cuda.synchronize()
        torch.autograd._enable_profiler(True)
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

def format_time(time_ns):
    """Defines how to format time in FunctionEvent"""
    return '{:.3f}us'.format(time_ns / 1000)


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


# TODO: record TID too
class FunctionEvent(FormattedTimesMixin):
    """Profiling information about a single function."""
    def __init__(self, id, name, start, end):
        self.id = id
        self.name = name
        self.start = start
        self.end = end
        self.kernels = []
        self.count = 1

    @property
    def cuda_time_total(self):
        return sum(kinfo[1] for kinfo in self.kernels)

    @property
    def cpu_time_total(self):
        return self.end - self.start

    @property
    def key(self):
        return self.name

    def __repr__(self):
        return '<FunctionEvent id={} cpu_time={} cuda_time={} name={}>'.format(
            self.id, self.cpu_time_str, self.cuda_time_str, self.name)


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

def demangle(name):
    """Demangle a C++ identifier using c++filt"""
    try:
        with open(os.devnull, 'w') as devnull:
            return subprocess.check_output(['c++filt', '-n', name], stderr=devnull).rstrip().decode("ascii")
    except subprocess.CalledProcessError:
        return name


class StringTable(defaultdict):
    def __missing__(self, key):
        self[key] = demangle(key)
        return self[key]


################################################################################
# CPU checkpoints

Record = namedtuple('Record', ['name', 'timestamp', 'kind'])


def parse_cpu_trace(thread_records):
    next_id = 0
    start_time = None
    functions = []
    function_stack = []
    string_table = StringTable()
    for r in itertools.chain(*thread_records):
        record = Record(*r)
        if record.name == '__start_profile':
            start_time = record.timestamp
        if record.kind == 'mark':
            continue
        elif record.kind == 'push':
            function_stack.append(FunctionEvent(
                id=next_id, name=string_table[record.name], start=record.timestamp, end=record.timestamp))
            next_id += 1
        elif record.kind == 'pop':
            function_stack[-1].end = record.timestamp
            functions.append(function_stack.pop())

    # Normalize times
    if start_time is None:
        raise RuntimeError('Malformed profile: no start marker')
    for event in functions:
        event.start -= start_time
        event.end -= start_time

    functions.sort(key=lambda evt: evt.start)
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
        strings[r["id"]] = demangle(r["value"])

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
                            start=row['start_time'],
                            end=row['end_time'])
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
        evt.kernels.append((row['kernel_name'], row['kernel_end'] - row['kernel_start']))

    functions.sort(key=lambda evt: evt.start)
    return functions


################################################################################
# Pretty printer

def build_table(events, sort_by=None, header=None):
    """Prints a summary of events (which can be a list of FunctionEvent or FunctionEventAvg)."""
    if sort_by is not None:
        events = sorted(events, key=lambda evt: getattr(evt, sort_by))

    max_name_length = max(len(evt.key) for evt in events)
    max_name_length += 4  # Add some nice padding
    col_width = 15
    col_format = '  {: >' + str(col_width) + '}'
    row_format = '{: <' + str(max_name_length) + '}' + col_format * 5
    header_sep = '-' * max_name_length + ('  ' + '-' * col_width) * 5

    # Have to use a list because nonlocal is Py3 only...
    result = ['']

    def append(s):
        result[0] += s
        result[0] += '\n'

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

    return result[0]
