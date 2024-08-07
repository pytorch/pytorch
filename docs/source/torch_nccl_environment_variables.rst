.. _torch_nccl_environment_variables:

PYTORCH ProcessGroupNCCL Environment Variables
==============================================
For more information on the environment variables, see `ProcessGroupNCCL Environment Variables <https://github.com/pytorch/pytorch/blob/main/torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>`_.

.. list-table::
  :header-rows: 1

  * - Variable
    - Description
  * - ``TORCH_NCCL_HIGH_PRIORITY``
    - Control whether to use high priority stream for the NCCL communicator.
  * - ``TORCH_NCCL_BLOCKING_WAIT``
    - Control whether or not wait() is blocking or non-blocking.
  * - ``TORCH_NCCL_DUMP_ON_TIMEOUT``
    - Control whether dumping debug info on watchdog timeout or exception is detected. This variable must be set together with TORCH_NCCL_TRACE_BUFFER_SIZE larger than 0.
  * - ``TORCH_NCCL_DESYNC_DEBUG``
    - Control whether Desync Debug is enabled. This is helpful in figuring out the culprit rank of collective desync.
  * - ``TORCH_NCCL_ENABLE_TIMING``
    - If set to ``1``, enable recording start-events for all ProcessGroupNCCL collectives, and compute accurate collective timing per-collective.
  * - ``TORCH_NCCL_ENABLE_MONITORING``
    - If set to ``1``,enable monitoring thread which aborts the process when the ProcessGroupNCCL Watchdog thread gets stuck and no heartbeat is detected after TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC. This can happen due to calling CUDA/NCCL APIs that may hang. It is Useful to prevent jobs being stuck for a prolonged time than necessary tying up cluster resources.
  * - ``TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC``
    - Control the watchdog heartbeat timeout period after which the monitoring thread will abort the process.
  * - ``TORCH_NCCL_TRACE_BUFFER_SIZE``
    - The maximum number of events we store in the flight recorder's ring buffer. One event could be the start or end of a collective, for example. Set to 0 to disable the tracebuffer and debugging info dump.
  * - ``TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC``
    - Control how much extra time we will wait for dumping the debugging info before we exit and throws timeout exception.
  * - ``TORCH_NCCL_DEBUG_INFO_TEMP_FILE``
    - The file into which the debugging info would be dumped.
  * - ``TORCH_NCCL_DEBUG_INFO_PIPE_FILE``
    - The pipe file to trigger debugging dump manually, write anything into the pipe would trigger the dump.
  * - ``TORCH_NCCL_NAN_CHECK``
    - Control whether to enable NAN check for the input, Error would be thrown if NAN is detected.
