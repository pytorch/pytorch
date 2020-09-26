from . import _TensorPipeRpcBackendOptionsBase
from . import constants as rpc_contants

from typing import List


class TensorPipeRpcBackendOptions(_TensorPipeRpcBackendOptionsBase):
    r"""
    The backend options for
    :class:`~torch.distributed.rpc.TensorPipeAgent`, derived from
    :class:`~torch.distributed.rpc.RpcBackendOptions`.

    Arguments:
        num_worker_threads (int, optional): The number of threads in the
            thread-pool used by
            :class:`~torch.distributed.rpc.TensorPipeAgent` to execute
            requests (default: 16).
        rpc_timeout (float, optional): The default timeout, in seconds,
            for RPC requests (default: 60 seconds). If the RPC has not
            completed in this timeframe, an exception indicating so will
            be raised. Callers can override this timeout for individual
            RPCs in :meth:`~torch.distributed.rpc.rpc_sync` and
            :meth:`~torch.distributed.rpc.rpc_async` if necessary.
        init_method (str, optional): The URL to initialize the distributed
            store used for rendezvous. It takes any value accepted for the
            same argument of :meth:`~torch.distributed.init_process_group`
            (default: ``env://``).
    """
    def __init__(
        self,
        *,
        num_worker_threads: int = rpc_contants.DEFAULT_NUM_WORKER_THREADS,
        rpc_timeout: float = rpc_contants.DEFAULT_RPC_TIMEOUT_SEC,
        init_method: str = rpc_contants.DEFAULT_INIT_METHOD,
        _transports: List = None,
        _channels: List = None,
    ):
        super().__init__(
            num_worker_threads,
            _transports,
            _channels,
            rpc_timeout,
            init_method
        )
