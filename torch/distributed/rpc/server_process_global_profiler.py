#!/usr/bin/python3
# mypy: allow-untyped-defs

from torch.autograd.profiler import profile

__all__: list[str] = []


class _server_process_global_profile(profile):
    """
    It has the same API as ``torch.autograd.profiler.profile`` class,
    except that it enables profiling on all threads running RPC server request callbacks.

    Context manager that manages autograd profiler state and holds a summary of results.
    Under the hood it relies on Kineto to record events of functions being executed in C++
    and exposes those events to Python. You can wrap any code into it and it will
    only report runtime of PyTorch functions.
    Note: profiling is process-global: enabling it here records events for every thread
    running RPC server-side request callbacks.

    Args:
        enabled (bool, optional): Setting this to False makes this context manager a no-op.
            Default: ``True``.

        use_device (str, optional): Enables timing of device events for the given device
            type (e.g. ``"cuda"`` or ``"npu"``). Adds approximately 4us of overhead to
            each tensor operation. Default: ``None``

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

    .. warning::
        Enabling memory profiling incurs additional profiler overhead

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
        # Backward compatibility: the legacy ``use_cuda`` flag maps to the
        # device-agnostic ``use_device`` argument of the Kineto profiler.
        use_cuda = kwargs.pop("use_cuda", None)
        if use_cuda is not None:
            kwargs.setdefault("use_device", "cuda" if use_cuda else None)
        super().__init__(*args, **kwargs)

    def __enter__(self):
        """
        Turn on server-side process-global profiling.
        The Kineto profiler is process-global: starting it here records events for
        every thread running RPC server-side request callbacks.
        """
        if not self.enabled:
            return

        if self.entered:  # type: ignore[has-type]
            raise RuntimeError("autograd profiler traces are not reentrant")
        # NOTE: do NOT set ``self.entered`` here — the Kineto profiler manages
        # this flag itself (it raises "not reentrant" if it is already True).

        return super().__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Turn off server-side process-global profiling.
        Aggregate all profiling events recorded by RPC threads (grouped by thread id).

        These attributes are assigned on exiting context.

        Attributes:
            function_events (torch.autograd.profiler_util.EventList).  It's a list that has helper
            methods, like 1) show record items in a pretty-print table.
            2) do averaging by grouping on keys. 3) and more.

            process_global_function_events (List[torch.autograd.profiler_util.EventList]).
            Every element is the profiling result of one thread within the profiling
            range, approximating the per-RPC-request results of the legacy profiler.
        """
        if not self.enabled:
            return False

        super().__exit__(exc_type, exc_val, exc_tb)

        # Group the collected events by thread to approximate the per-thread
        # profiling results previously produced by the legacy mechanism.
        threads: dict = {}
        for function_event in self.function_events:
            threads.setdefault(getattr(function_event, "thread", None), []).append(
                function_event
            )

        process_global_function_events = []
        for thread_local_function_events in threads.values():
            thread_local_function_events.sort(
                key=lambda function_event: [
                    function_event.time_range.start,
                    -(function_event.time_range.end),
                ]
            )
            process_global_function_events.append(thread_local_function_events)

        self.process_global_function_events = process_global_function_events

        return False
