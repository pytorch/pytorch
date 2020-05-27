#!/usr/bin/python3

import torch
from torch.autograd import ProfilerConfig

from . import (
    __disable_server_process_global_profiler,
    __enable_server_process_global_profiler,
)


def _enable_server_process_global_profiler(config: ProfilerConfig):
    r"""
    Turn on server-side process-global profiling.
    This enables thread-local profiler on all RPC threads running server-side request callbacks.

    Arguments:
        config (torch.autograd.profiler.ProfilerConfig): This config specifies
            1) what mode. CPU only or CPU + CUDA.
            2) Whether to record input shapes.

    Returns:
        None

    Example::
        >>> import torch
        >>> profiler_config = torch.autograd.ProfilerConfig(
        >>>     /* profiler_state */ torch.autograd.ProfilerState.CPU,
        >>>     /* record_input_shapes */ False,
        >>>     /* profile_memory */ False,
        >>> )
        >>> rpc.enable_server_process_global_profiler(profiler_config)

    """
    __enable_server_process_global_profiler(config)


def _disable_server_process_global_profiler():
    """
    Turn off server-side process-global profiling.
    Aggregrate all profiling events recorded by RPC threads.

    Returns:
        event_list (torch.autograd.profiler.EventList). A list that have helper
        methods like show record items in a pretty-print table,
        do averaging by grouping on keys and more.
    """
    profile_ranges = __disable_server_process_global_profiler()

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
