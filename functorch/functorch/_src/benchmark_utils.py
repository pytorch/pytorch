import time

import torch
from torch.profiler import profile


def synchronize():
    pass


def dump_chrome_trace(f, input, trace_filename, optimize_ctx, activities, num_runs=1,
                      devices=None, kwargs_for_f=None, kwargs_for_profiler=None):
    """
    Output the chrome trace of running f(input, **kwargs_for_f) with [optimize_ctx]
    [num_runs] times to [trace_filename].

    [activities] are the activities that the profiler will record, e.g. ProfilerActivity.CUDA.
    Return total runtime without the profiler

    Outputs to trace_filename
    """

    if devices is None:
        devices = ["cuda"]

    global synchronize
    if devices != ["cpu"] and torch.cuda.is_available():
        synchronize = torch.cuda.synchronize

    if kwargs_for_f is None:
        kwargs_for_f = {}
    if kwargs_for_profiler is None:
        kwargs_for_profiler = {}

    with optimize_ctx:
        torch.manual_seed(1337)
        for _ in range(5):  # warmup runs
            f(input, **kwargs_for_f)
            synchronize()
        torch.manual_seed(1337)
        t0 = time.perf_counter()
        for _ in range(num_runs):
            f(input, **kwargs_for_f)
            synchronize()
        t1 = time.perf_counter()
    timing = t1 - t0

    with profile(activities=activities, **kwargs_for_profiler) as prof:
        with optimize_ctx:
            synchronize()
            torch.manual_seed(1337)
            for _ in range(num_runs):
                f(input, **kwargs_for_f)
                synchronize()
    prof.export_chrome_trace(trace_filename)

    return timing
