from __future__ import absolute_import, division, print_function, unicode_literals

import numbers
import sys

import torch
import torch.distributed as dist

from . import backend_registry


def is_available():
    return sys.version_info >= (3, 0) and hasattr(torch._C, "_rpc_init")


if is_available() and not torch._C._rpc_init():
    raise RuntimeError("Failed to initialize torch.distributed.rpc")


if is_available():
    from .api import _init_rpc_backend
    from .api import *  # noqa: F401
    import torch.distributed.autograd

    def init_rpc(
        name,
        backend=backend_registry.BackendType.PROCESS_GROUP,
        init_method=None,
        rank=-1,
        world_size=None,
        rpc_backend_options=None,
    ):
        r"""
        Initializes RPC primitives such as the local RPC agent
        and distributed autograd.

        Initializes the local RPC agent which immediately makes the current
        process ready to send and receive RPCs. This method also properly
        initializes a default process group backend that uses gloo for
        collective communication.

        Arguments:
            backend (Enum): type of RPC backend implementation.
                        Currently, process group backend is the only
                        available backend implementation. (default:
                        ``RpcBackend.PROCESS_GROUP``).
            name (str): a globally unique name of this node. (e.g.,
                        ``Trainer3``, ``ParameterServer2``, ``Master``,
                        ``Worker1``) Name can only contain number, alphabet,
                        underscore, and/or dash, and must be shorter than
                        128 characters.
            init_method(str): backend specific init arguments.
            rank (int): a globally unique id/rank of this node.
            world_size (int): The number of workers in the group.
            rpc_backend_options (RpcBackendOptions): The options passed to RpcAgent
                consturctor.
        """
        # Rendezvous.
        rendezvous_iterator = torch.distributed.rendezvous(
            init_method, rank=rank, world_size=world_size
        )
        store, _, _ = next(rendezvous_iterator)

        # Initialize autograd before RPC since _init_rpc_backend guarantees all
        # processes sync via the store. If we initialize autograd after RPC,
        # there could be a race where some nodes might have initialized autograd
        # and others might not have. As a result, a node calling
        # torch.distributed.autograd.backward() would run into errors since
        # other nodes might not have been initialized.
        torch.distributed.autograd._init(rank)

        # Initialize RPC.
        _init_rpc_backend(backend, store, name, rank, world_size, rpc_backend_options)
