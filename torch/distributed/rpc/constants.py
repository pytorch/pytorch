from datetime import timedelta

from . import (
    _DEFAULT_INIT_METHOD,
    _DEFAULT_NUM_SEND_RECV_THREADS,
    _DEFAULT_NUM_WORKER_THREADS,
    _DEFAULT_RPC_TIMEOUT_SEC,
    _UNSET_RPC_TIMEOUT,
)


# For any RpcAgent.
DEFAULT_RPC_TIMEOUT_SEC = _DEFAULT_RPC_TIMEOUT_SEC
DEFAULT_INIT_METHOD = _DEFAULT_INIT_METHOD
DEFAULT_SHUTDOWN_TIMEOUT = 5.0

# For ProcessGroupAgent.
DEFAULT_NUM_SEND_RECV_THREADS = _DEFAULT_NUM_SEND_RECV_THREADS
# For TensorPipeAgent.
DEFAULT_NUM_WORKER_THREADS = _DEFAULT_NUM_WORKER_THREADS
# Ensure that we don't time out when there are long periods of time without
# any operations against the underlying ProcessGroup.
DEFAULT_PROCESS_GROUP_TIMEOUT = timedelta(milliseconds=2 ** 31 - 1)
# Value indicating that timeout is not set for RPC call, and the default should be used.
UNSET_RPC_TIMEOUT = _UNSET_RPC_TIMEOUT
