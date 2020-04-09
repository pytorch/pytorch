from __future__ import absolute_import, division, print_function, unicode_literals

import numbers
import sys

import torch
import torch.distributed as dist


def is_available():
    return sys.version_info >= (3, 0) and hasattr(torch._C, "_rpc_init")


if is_available() and not torch._C._rpc_init():
    raise RuntimeError("Failed to initialize torch.distributed.rpc")


if is_available():
    from . import api, backend_registry
    from .api import *  # noqa: F401
    import torch.distributed.autograd as dist_autograd

    def init_rpc(
        name,
        backend=backend_registry.BackendType.PROCESS_GROUP,
        rank=-1,
        world_size=None,
        rpc_backend_options=None,
    ):
        r"""
        Initializes RPC primitives such as the local RPC agent
        and distributed autograd.

        Initializes the local RPC agent which immediately makes the current
        process ready to send and receive RPCs. This method also properly
        initializes a default process group backend that uses Gloo for
        communication.

        Arguments:
            backend (Enum): type of RPC backend implementation. Currently,
                process group backend is the only available backend
                implementation. (default: ``RpcBackend.PROCESS_GROUP``).
            name (str): a globally unique name of this node. (e.g.,
                ``Trainer3``, ``ParameterServer2``, ``Master``, ``Worker1``)
                Name can only contain number, alphabet, underscore, and/or dash,
                and must be shorter than 128 characters.
            rank (int): a globally unique id/rank of this node.
            world_size (int): The number of workers in the group.
            rpc_backend_options (RpcBackendOptions): The options passed to
                RpcAgent constructor. It contains RpcAgent specific
                initialization configurations. By default, it contains
                ``rpc_timeout = timedelta(seconds=60)``,
                ``init_method = "env://"``, ``num_send_recv_threads = 4`` for
                process group agent. If using the default
                ``rpc_backend_options``, RPC would initialize the underlying
                process group backend using ``init_method = "env://"``,
                meaning that environment variables ``MASTER_ADDRESS`` and
                ``MASTER_PORT`` needs to be set properly. See
                :class:`~torch.distributed.rpc.ProcessGroupRpcBackendOptions`
                for examples.
        """

        if not rpc_backend_options:
            # default construct a set of RPC backend options.
            rpc_backend_options = backend_registry.construct_rpc_backend_options(
                backend
            )

        # Rendezvous.
        # This rendezvous state sometimes is destroyed before all processes
        # finishing handshaking. To avoid that issue, we make it global to
        # keep it alive.
        global rendezvous_iterator
        rendezvous_iterator = torch.distributed.rendezvous(
            rpc_backend_options.init_method, rank=rank, world_size=world_size
        )
        store, _, _ = next(rendezvous_iterator)

        # Initialize autograd before RPC since _init_rpc_backend guarantees all
        # processes sync via the store. If we initialize autograd after RPC,
        # there could be a race where some nodes might have initialized autograd
        # and others might not have. As a result, a node calling
        # torch.distributed.autograd.backward() would run into errors since
        # other nodes might not have been initialized.
        dist_autograd._init(rank)

        # Initialize RPC.
        api._init_rpc_backend(backend, store, name, rank, world_size, rpc_backend_options)


    @api._require_initialized
    def _get_debug_info():
        from . import _rref_context_get_debug_info
        info = _rref_context_get_debug_info()
        info.update(api._get_current_rpc_agent().get_debug_info())
        info.update(dist_autograd._get_debug_info())
        return info
