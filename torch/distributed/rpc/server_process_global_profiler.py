#!/usr/bin/python3

import torch
from torch.autograd import ProfilerConfig

from . import (
    _disable_server_process_global_profiler,
    _enable_server_process_global_profiler,
)


def enable_server_process_global_profiler(config: ProfilerConfig):
    r"""
    Turn on the state that indicates the server-side process-global profiling is
    on. This enables all RPC threads running server-side request callbacks.

    Arguments:
        config (torch.autograd.profiler.ProfilerConfig): This config specifies
            1) what mode. CPU only or CPU + CUDA.
            2) Whether to record input shapes.

    Returns:
        None

    """
    _enable_server_process_global_profiler(config)


def disable_server_process_global_profiler():
    """
    Turn off the state that indicates the server-side process-global profiling is
    on. Aggregrate all profiling events recorded by RPC threads.

    Returns:
        event_list (torch.autograd.profiler.EventList). A list that have helper
        methods like show record items in a pretty-print table,
        do averaging by grouping on keys and more.
    """
    profile_ranges = _disable_server_process_global_profiler()

    process_global_function_events = []
    for profile_range in profile_ranges:
        thread_local_function_events = torch.autograd.profiler.parse_cpu_trace(
            profile_range
        )
        process_global_function_events.extend(thread_local_function_events)

    process_global_function_events.sort(
        key=lambda function_event: [
            function_event.cpu_interval.start,
            -(function_event.cpu_interval.end),
        ]
    )

    return torch.autograd.profiler.EventList(
        process_global_function_events, use_cuda=False, profile_memory=False
    )
