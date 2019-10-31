from __future__ import absolute_import, division, print_function, unicode_literals

import sys

from . import backend_registry


if sys.version_info >= (3, 0):
    from . import api
    from .api import _init_rpc
    from .api import *  # noqa: F401
    import torch.distributed.autograd

    def init_model_parallel(
        self_name,
        backend=backend_registry.BackendType.PROCESS_GROUP,
        init_method=None,
        self_rank=-1,
        worker_name_to_id=None,
        num_send_recv_threads=4,
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
        """
        # Rendezvous.
        world_size = len(worker_name_to_id)
        rendezvous_iterator = torch.distributed.rendezvous(
            init_method, rank=self_rank, world_size=world_size
        )
        store, _, _ = next(rendezvous_iterator)

        # Initialize RPC.
        _init_rpc(
            backend,
            store,
            self_name,
            self_rank,
            worker_name_to_id,
            num_send_recv_threads,
        )

        # Initialize Autograd.
        torch.distributed.autograd._init(api._agent.get_worker_info().id)
