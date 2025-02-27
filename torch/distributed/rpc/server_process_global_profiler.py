#!/usr/bin/python3
# mypy: allow-untyped-defs

import itertools

import torch
from torch.autograd.profiler_legacy import profile

from . import (
    _disable_server_process_global_profiler,
    _enable_server_process_global_profiler,
)


__all__: list[str] = []


class _server_process_global_profile(profile):
    """
    It has the same API as ``torch.autograd.profiler.profile`` class,
    except that it enables profiling on all threads running RPC server request callbacks.

    Context manager that manages autograd profiler state and holds a summary of results.
    Under the hood it just records events of functions being executed in C++ and
    exposes those events to Python. You can wrap any code into it and it will
    only report runtime of PyTorch functions.
    Note: profiler is thread local and is automatically propagated into the async tasks

    Args:
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

    .. warning:
        Enabling memory profiling incurs additional profiler overhead

    .. warning:
        Due to some CUDA multiprocessing limitations (multiprocessing-cuda-note_),
        one cannot use the profiler with ``use_cuda = True`` to benchmark
        DataLoaders with ``num_workers > 0``. If you wish to benchmark data loading,
        please use ``use_cuda = False`` or ``num_workers = 0``.

    Example:
        >>> # xdoctest: +SKIP
        >>> # On worker 0:
        >>> import torch
        >>> import torch.distributed.rpc as rpc
        >>> rpc.init_rpc("worker0", rank=0, world_size=2)
        >>> x, y = torch.tensor(1), torch.tensor(2)
        >>> outer_profile_rref = rpc.remote(
        ...     dst_worker_name, rpc._server_process_global_profile
        ... )
        >>> outer_profile_rref.rpc_sync().__enter__()
        >>> rpc.rpc_sync(dst_worker_name, torch.add, (x, y))
        >>> inner_profile_rref = rpc.remote(
        ...     dst_worker_name, rpc._server_process_global_profile
        ... )
        >>> inner_profile_rref.rpc_sync().__enter__()
        >>> rpc.rpc_sync(dst_worker_name, torch.sub, (x, y))
        >>> inner_profile_rref.rpc_sync().__exit__(None, None, None)
        >>> outer_profile_rref.rpc_sync().__exit__(None, None, None)
        >>> print(inner_profile_rref.rpc_sync().key_averages())
        ---------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------
        Name       Self CPU total %  Self CPU total   CPU total %      CPU total        CPU time avg     Number of Calls
        ---------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------
        sub        85.06%           76.275us         100.00%          89.667us         89.667us         1
        empty      14.94%           13.392us         14.94%           13.392us         13.392us         1
        ---------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------
        Self CPU time total: 89.667us
        >>> print(outer_profile_rref.rpc_sync().key_averages())
        ---------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------
        Name       Self CPU total %  Self CPU total   CPU total %      CPU total        CPU time avg     Number of Calls
        ---------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------
        sub        35.65%           76.275us         41.91%           89.667us         89.667us         1
        empty      12.67%           27.101us         12.67%           27.101us         13.551us         2
        add        51.68%           110.550us        58.09%           124.259us        124.259us        1
        ---------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------
        Self CPU time total: 213.926us
        >>> rpc.shutdown()

        >>> # On worker 1:
        >>> import torch.distributed.rpc as rpc
        >>> rpc.init_rpc("worker1", rank=1, world_size=2)
        >>> # wait for worker 0 to finish work, and then shutdown.
        >>> rpc.shutdown()
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __enter__(self):
        """
        Turn on server-side process-global profiling.
        This enables thread-local profiler on all RPC threads running server-side request callbacks.
        """
        if not self.enabled:
            return

        if self.entered:  # type: ignore[has-type]
            raise RuntimeError("autograd profiler traces are not reentrant")
        self.entered = True

        profiler_kind = (
            torch.autograd.ProfilerState.CUDA
            if self.use_cuda
            else torch.autograd.ProfilerState.CPU
        )
        profiler_config = torch.autograd.ProfilerConfig(
            profiler_kind,
            self.record_shapes,
            self.profile_memory,
            False,
            False,
            False,
            torch.profiler._ExperimentalConfig(),
        )
        _enable_server_process_global_profiler(profiler_config)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Turn off server-side process-global profiling.
        Aggregate all profiling events recorded by RPC threads.

        These attributes are assigned on exiting context.

        Attributes:
            function_events (torch.autograd.profiler.EventList).  It's a list that has helper
            methods, like 1) show record items in a pretty-print table.
            2) do averaging by grouping on keys. 3) and more.

            process_global_function_events (List[torch.autograd.profiler.FunctionEvent]).
            It's a list of ``FunctionEvent`` elements. Every element is a profiling result
            of an RPC request handling within the profiling range.
        """
        if not self.enabled:
            return

        process_global_events = _disable_server_process_global_profiler()

        # Every element in this list is a thread profiling result from an RPC request handling.
        process_global_function_events = []
        for thread_local_events in process_global_events:
            # Parse from ``Event``s to ``FunctionEvent``s.
            thread_local_function_events = (
                torch.autograd.profiler_legacy._parse_legacy_records(
                    thread_local_events
                )
            )
            thread_local_function_events.sort(
                key=lambda function_event: [
                    function_event.time_range.start,
                    -(function_event.time_range.end),
                ]
            )
            process_global_function_events.append(thread_local_function_events)

        flattened_function_events = list(
            itertools.chain.from_iterable(process_global_function_events)
        )
        self.function_events = torch.autograd.profiler_util.EventList(
            flattened_function_events,
            use_device="cuda" if self.use_cuda else None,
            profile_memory=self.profile_memory,
        )
        self.function_events._build_tree()

        self.process_global_function_events = process_global_function_events

        return False
