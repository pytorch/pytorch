(_torch_nccl_environment_variables)=
# PYTORCH NCCL Backend Environment Variables

These variables configure the `nccl`, `nccl-lazy` and `nccl-legacy` backends. Each entry below states which backends read it: **nccl-legacy only**, **All NCCL backends**, or the backends it applies to.

`nccl-legacy` records into the flight recorder natively and dumps it itself. For `nccl`, `nccl-lazy` and other backends without native support, `FlightRecorderHook` records and dumps the trace. The recorder reads the buffer size, C++ stack and dump file variables; the dump-on-failure variables are read by `nccl-legacy` or by `FlightRecorderHook`, as noted per entry. Where both are read, the `TORCH_FR_*` name takes precedence over its `TORCH_NCCL_*` alias.

```{list-table}
:header-rows: 1

* - **Variable**
  - **Description**
* - ``TORCH_DIST_NAN_CHECK``
  - **All backends.** If set to ``1``, attach a backend-agnostic NaN check to every new process group; input buffers of collectives are checked for NaNs. Use this with ``nccl``, which ignores ``TORCH_NCCL_NAN_CHECK``. Default: ``0``.
* - ``TORCH_NCCL_ASYNC_ERROR_HANDLING``
  - **All NCCL backends.** Control how we perform Async Error Handling with NCCL when an exception is observed in watchdog. If set to 0, no handling of asynchronous NCCL errors. If set to 1, aborting NCCL communicator and tearing down process upon error. If set to 2, only abort NCCL communicator and if set to 3, tearing down process without aborting NCCL communicator. By default, it is set to 3; ``torchrun`` sets it to 1 for its workers if it is unset.
* - ``TORCH_NCCL_HIGH_PRIORITY``
  - **All NCCL backends.** Control whether to use high priority stream for the NCCL communicator. Default: ``0``.
* - ``TORCH_NCCL_BLOCKING_WAIT``
  - **All NCCL backends.** Control whether or not wait() is blocking or non-blocking. Default: ``0``.
* - ``TORCH_NCCL_USE_COMM_NONBLOCKING``
  - **All NCCL backends.** If set to ``1``, create NCCL communicators in nonblocking mode. An explicit ``blocking`` value in the NCCL config takes precedence. Default: ``0``.
* - ``TORCH_NCCL_NONBLOCKING_TIMEOUT``
  - **nccl-legacy only.** Timeout in seconds for nonblocking NCCL communicator operations (init, abort, finalize). Default: ``1800``.
* - ``TORCH_NCCL_RANKS_PER_ROOT``
  - **All NCCL backends.** Number of ranks each root covers during scalable NCCL communicator init; groups larger than this use multiple roots. ``nccl`` rejects values less than 1. Default: ``128``.
* - ``TORCH_NCCL_BCAST_UNIQUEID``
  - **nccl-legacy only.** Whether to broadcast the NCCL unique ID through the store when creating a communicator. Default: ``1``.
* - ``TORCH_NCCL_CUDA_EVENT_CACHE``
  - **All NCCL backends.** Whether to reuse CUDA events through a cache instead of creating and destroying them per collective. Destroying a CUDA event while the CUDA global lock is held can hang. Default: ``1``.
* - ``TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK``
  - **nccl-legacy only.** If set to ``1``, register CUDA caching allocator hooks so that allocated and freed segments are registered and deregistered with all NCCL communicators. Default: ``0``.
* - ``TORCH_NCCL_SHOW_EAGER_INIT_P2P_SERIALIZATION_WARNING``
  - **nccl-legacy only.** Whether to warn that point-to-point operations are serialized when the communicator is eagerly initialized. Default: ``1``.
* - ``TORCH_NCCL_DESYNC_DEBUG``
  - **nccl-legacy only.** Control whether Desync Debug is enabled. This is helpful in figuring out the culprit rank of collective desync.
* - ``TORCH_NCCL_ENABLE_TIMING``
  - **All NCCL backends.** If set to ``1``, enable recording start-events for all ProcessGroupNCCL collectives, and compute accurate collective timing per-collective. Default: ``0``.
* - ``TORCH_NCCL_ENABLE_MONITORING``
  - **nccl-legacy only.** If set to ``1``, enable monitoring thread which aborts the process when the ProcessGroupNCCL Watchdog thread gets stuck and no heartbeat is detected after TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC. This can happen due to calling CUDA/NCCL APIs that may hang. It is Useful to prevent jobs being stuck for a prolonged time than necessary tying up cluster resources. Default: ``1``.
* - ``TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC``
  - **nccl-legacy only.** Control the watchdog heartbeat timeout period after which the monitoring thread will abort the process. Default: ``480``.
* - ``TORCH_NCCL_PROPAGATE_ERROR``
  - **nccl-legacy only.** If set to ``1``, propagate NCCL errors to all ranks through the store. Default: ``0``.
* - ``TORCH_NCCL_RETHROW_CUDA_ERRORS``
  - **nccl-legacy only.** Whether the watchdog rethrows CUDA errors. Default: ``1``.
* - ``TORCH_NCCL_TEARDOWN_ON_TIMEOUT``
  - **nccl-legacy only.** If set to ``1``, a collective timeout tears down the process even when ``TORCH_NCCL_RETHROW_CUDA_ERRORS=0``. Only applies when ``TORCH_NCCL_ASYNC_ERROR_HANDLING`` is 1 or 3. Default: ``0``.
* - ``TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN``
  - **nccl-legacy only.** Whether to log C++ stack traces on unclean shutdown. Default: ``1``.
* - ``TORCH_NCCL_COORD_CHECK_MILSEC``
  - **All NCCL backends.** Control the interval inside the monitoring thread to check the coordinated signal from other ranks, e.g. to dump the debugging information. ``nccl-legacy`` reads only this name; for ``nccl`` it is the alias of ``TORCH_FR_COORD_CHECK_MILSEC``. Default value is 1000 ms.
* - ``TORCH_FR_COORD_CHECK_MILSEC``
  - **`nccl`, `nccl-lazy` and other hooked backends.** Interval in milliseconds at which each process polls the store for a flight recorder dump signal from another rank. Falls back to ``TORCH_NCCL_COORD_CHECK_MILSEC``. Default: ``1000``.
* - ``TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC``
  - **All NCCL backends.** Control how much extra time we will wait for dumping the debugging info before we exit and throws timeout exception. ``nccl`` waits this long before aborting the process; its flight recorder dump reads ``TORCH_FR_WAIT_TIMEOUT_DUMP_MILSEC`` first. Default: ``15000``.
* - ``TORCH_FR_WAIT_TIMEOUT_DUMP_MILSEC``
  - **`nccl`, `nccl-lazy` and other hooked backends.** Time limit in milliseconds for a flight recorder dump on failure. Falls back to ``TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC``. Default: ``15000``.
* - ``TORCH_FR_DUMP_ON_TIMEOUT``
  - **`nccl`, `nccl-lazy` and other hooked backends.** Whether to dump the flight recorder trace when a collective times out or fails. Falls back to ``TORCH_NCCL_DUMP_ON_TIMEOUT``. Default: ``1``.
* - ``TORCH_NCCL_DUMP_ON_TIMEOUT``
  - **All NCCL backends.** Control whether dumping debug info on watchdog timeout or exception is detected. ``nccl-legacy`` reads only this name; for ``nccl`` it is the alias of ``TORCH_FR_DUMP_ON_TIMEOUT``. Has no effect if the trace buffer size is 0. Default: ``1``.
* - ``TORCH_FR_BUFFER_SIZE``, ``TORCH_NCCL_TRACE_BUFFER_SIZE``
  - **All NCCL backends.** The maximum number of events we store in the flight recorder's ring buffer. One event could be the start or end of a collective, for example. Set to 0 to disable recording. Default: ``2000``. ``nccl-legacy`` decides whether to dump, and whether to create the ``TORCH_NCCL_DEBUG_INFO_PIPE_FILE`` pipe, from ``TORCH_NCCL_TRACE_BUFFER_SIZE`` alone, so ``TORCH_FR_BUFFER_SIZE=0`` stops recording but ``nccl-legacy`` still dumps an empty trace. For ``nccl``, the first name that is set controls both recording and dumping.
* - ``TORCH_FR_CPP_STACK``, ``TORCH_NCCL_TRACE_CPP_STACK``
  - **All NCCL backends.** Whether to collect cpp stack traces for flight recorder. Default: ``0``.
* - ``TORCH_FR_DUMP_TEMP_FILE``, ``TORCH_NCCL_DEBUG_INFO_TEMP_FILE``
  - **All NCCL backends.** Prefix of the file into which the debugging info is dumped; the rank is appended. Default: ``$XDG_CACHE_HOME/torch/comm_lib_trace_rank_`` (``XDG_CACHE_HOME`` defaults to ``~/.cache``).
* - ``TORCH_FR_DUMP_DYNAMIC_FILE_NAME``
  - **All NCCL backends.** If set to ``1``, reread ``TORCH_FR_DUMP_TEMP_FILE`` at each dump and use its value as the full file path, without appending the rank. The ``TORCH_NCCL_DEBUG_INFO_TEMP_FILE`` alias is not reread. Default: ``0``.
* - ``TORCH_INCLUDE_STACK_TRACE``
  - **All NCCL backends.** Whether a flight recorder dump on timeout or failure includes stack traces. If the dump does not finish within the wait timeout, it is retried without them. Default: ``1``.
* - ``TORCH_INCLUDE_ONLY_ACTIVE``
  - **All NCCL backends.** Whether a flight recorder dump includes only collectives that have not completed. Default: ``0``.
* - ``TORCH_NCCL_EXTRA_DUMP_ON_EXEC``
  - **nccl-legacy only.** If set to ``1``, also dump the debugging info when the process group is aborted or its watchdog hits an exception. Default: ``0``.
* - ``TORCH_NCCL_DEBUG_INFO_PIPE_FILE``
  - **nccl-legacy only.** The pipe file to trigger debugging dump manually, write anything into the pipe would trigger the dump.
* - ``TORCH_NCCL_NAN_CHECK``
  - **nccl-legacy only.** Control whether to enable NAN check for the input, Error would be thrown if NAN is detected. For other backends, including ``nccl``, use ``TORCH_DIST_NAN_CHECK``. Default: ``0``.
* - ``TORCH_NCCL2_RECONFIGURE_SHRINK_GROW``
  - **`nccl` and `nccl-lazy` only.** If set to ``1``, ``reconfigure`` uses NCCL's shrink/grow fast path (NCCL 2.29 or later) when NCCL's rank order matches c10d's; otherwise it creates a new communicator. Set the same value on every rank. Default: ``0``.
```
