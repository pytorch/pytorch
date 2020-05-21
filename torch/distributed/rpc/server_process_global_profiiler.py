#!/usr/bin/python3

import torch

from . import (
    _disable_server_process_global_profiler,
    _enable_server_process_global_profiler,
)


def enable_server_process_global_profiler(config):
    _enable_server_process_global_profiler(config)


def disable_server_process_global_profiler():
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
