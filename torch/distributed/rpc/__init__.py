from __future__ import absolute_import, division, print_function, unicode_literals

import sys
import torch


from . import backend_registry
from .constants import DEFAULT_RPC_TIMEOUT, DEFAULT_NUM_SEND_RECV_THREADS


def is_available():
    return sys.version_info >= (3, 0) and hasattr(torch._C, "_rpc_init")


if is_available() and not torch._C._rpc_init():
    raise RuntimeError("Failed to initialize torch.distributed.rpc")


if is_available():
    from .api import _init_rpc
    from .api import *  # noqa: F401
    import torch.distributed.autograd

    def init_model_parallel(
        self_name,
        backend=backend_registry.BackendType.PROCESS_GROUP,
        init_method=None,
        self_rank=-1,
        worker_name_to_id=None,
        num_send_recv_threads=DEFAULT_NUM_SEND_RECV_THREADS,
        rpc_timeout=DEFAULT_RPC_TIMEOUT,
    ):
        r"""
        Initializes model parallel primitives such as the local rpc agent
        and distributed autograd.

        Initializes the local RPC agent which immediately makes the current
        process ready to send and receive RPCs. The caller needs to make
        sure the specified backend is properly intialized before calling
        this method. For example, to use ``pg`` (ProcessGroup) backend,
        ``init_process_group`` must be invoked prior to this method.

        Arguments:
            backend (Enum): type of RPC backend implementation.
                        Currently, process group backend is the only
                        available backend implementation. (default:
                        ``RpcBackend.PROCESS_GROUP``).
            self_name (str): a globally unique name of this node. (e.g.,
                        ``Trainer3``, ``ParameterServer2``, ``Master``,
                        ``Worker1``) Name can only contain number, alphabet,
                        underscore, and/or dash, and must be shorter than
                        128 characters.
            self_rank (int): a globally unique id/rank of this node.
            init_method(str): backend specific init arguments.
            num_send_recv_threads(int): Number of threads for send/recv work.
            rpc_timeout (datetime.timedelta): Timeout for RPCs. Defaults to 10 seconds.
        """
        # Rendezvous.
        world_size = len(worker_name_to_id)
        rendezvous_iterator = torch.distributed.rendezvous(
            init_method, rank=self_rank, world_size=world_size
        )
        store, _, _ = next(rendezvous_iterator)

        # Initialize autograd before RPC since _init_rpc guarantees all
        # processes sync via the store. If we initialize autograd after RPC,
        # there could be a race where some nodes might have initialized autograd
        # and others might not have. As a result, a node calling
        # torch.distributed.autograd.backward() would run into errors since
        # other nodes might not have been initialized.
        torch.distributed.autograd._init(worker_name_to_id[self_name])

        # Initialize RPC.
        _init_rpc(
            backend,
            store,
            self_name,
            self_rank,
            worker_name_to_id,
            num_send_recv_threads,
            rpc_timeout,
        )
