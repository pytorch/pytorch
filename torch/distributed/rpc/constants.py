from datetime import timedelta
from typing import List
from torch._C._distributed_rpc import (
    _DEFAULT_INIT_METHOD,
    _DEFAULT_NUM_WORKER_THREADS,
    _DEFAULT_RPC_TIMEOUT_SEC,
    _UNSET_RPC_TIMEOUT,
)


# For any RpcAgent.
DEFAULT_RPC_TIMEOUT_SEC: float = _DEFAULT_RPC_TIMEOUT_SEC
DEFAULT_INIT_METHOD: str = _DEFAULT_INIT_METHOD
DEFAULT_SHUTDOWN_TIMEOUT: float = 0

# For TensorPipeAgent.
DEFAULT_NUM_WORKER_THREADS: int = _DEFAULT_NUM_WORKER_THREADS
# Ensure that we don't time out when there are long periods of time without
# any operations against the underlying ProcessGroup.
DEFAULT_PROCESS_GROUP_TIMEOUT: timedelta = timedelta(milliseconds=2 ** 31 - 1)
# Value indicating that timeout is not set for RPC call, and the default should be used.
UNSET_RPC_TIMEOUT: float = _UNSET_RPC_TIMEOUT

__all__: List[str] = []
