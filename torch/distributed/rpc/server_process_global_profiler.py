#!/usr/bin/python3

import torch
from torch.autograd import ProfilerConfig
from torch.autograd.profiler import profile

from . import (
    __disable_server_process_global_profiler,
    __enable_server_process_global_profiler,
)


# ==================== Functional API ====================


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
        >>>     torch.autograd.ProfilerState.CPU,  # profiler_kind
        >>>     False,  # record_input_shapes
        >>>     False,  # profile_memory
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


# ==================== Context API ====================


class _server_process_global_profile(profile):
    """
    It has the same API as ``torch.autpgrad.profiler.profile`` class,
    except that it enables profiling on all threads running RPC server request callbacks.
    """
    def __enter__(self):
        if not self.enabled:
            return

        if self.entered:
            raise RuntimeError("autograd profiler traces are not reentrant")
        self.entered = True

        profiler_kind = (
            torch.autograd.ProfilerState.CUDA
            if self.use_cuda
            else torch.autograd.ProfilerState.CPU
        )
        profiler_config = torch.autograd.ProfilerConfig(
            profiler_kind, self.record_shapes, self.profile_memory
        )
        _enable_server_process_global_profiler(profiler_config)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.enabled:
            return

        self.function_events = _disable_server_process_global_profiler()
        return False
