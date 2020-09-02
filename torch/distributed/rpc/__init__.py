from __future__ import absolute_import, division, print_function, unicode_literals

import torch
import torch.distributed as dist
import threading

_init_counter = 0
_init_counter_lock = threading.Lock()

def is_available():
    return hasattr(torch._C, "_rpc_init")


if is_available() and not torch._C._rpc_init():
    raise RuntimeError("Failed to initialize torch.distributed.rpc")




if is_available():
    from . import api, backend_registry, functions, _set_profiler_node_id
    from . import (
        _disable_jit_rref_pickle,
        _enable_jit_rref_pickle,
        _set_and_start_rpc_agent,
    )  # noqa: F401
    from .api import *  # noqa: F401
    from .options import TensorPipeRpcBackendOptions  # noqa: F401
    from .backend_registry import BackendType
    from .server_process_global_profiler import (
        _server_process_global_profile,
    )
    import torch.distributed.autograd as dist_autograd
    from torch._rref import RRef, is_rref

    import numbers


    def init_rpc(
        name,
        backend=BackendType.TENSORPIPE,
        rank=-1,
        world_size=None,
        rpc_backend_options=None,
    ):
        r"""
        Initializes RPC primitives such as the local RPC agent
        and distributed autograd, which immediately makes the current
        process ready to send and receive RPCs.

        Arguments:
            backend (BackendType, optional): The type of RPC backend
                implementation. Supported values include
                ``BackendType.TENSORPIPE`` (the default) and
                ``BackendType.PROCESS_GROUP``. See :ref:`rpc-backends` for more
                information.
            name (str): a globally unique name of this node. (e.g.,
                ``Trainer3``, ``ParameterServer2``, ``Master``, ``Worker1``)
                Name can only contain number, alphabet, underscore, colon,
                and/or dash, and must be shorter than 128 characters.
            rank (int): a globally unique id/rank of this node.
            world_size (int): The number of workers in the group.
            rpc_backend_options (RpcBackendOptions, optional): The options
                passed to the RpcAgent constructor. It must be an agent-specific
                subclass of :class:`~torch.distributed.rpc.RpcBackendOptions`
                and contains agent-specific initialization configurations. By
                default, for all agents, it sets the default timeout to 60
                seconds and performs the rendezvous with an underlying process
                group initialized using ``init_method = "env://"``,
                meaning that environment variables ``MASTER_ADDR`` and
                ``MASTER_PORT`` need to be set properly. See
                :ref:`rpc-backends` for more information and find which options
                are available.
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

        # Use a PrefixStore to distinguish multiple invocations.
        with _init_counter_lock:
            global _init_counter
            store = dist.PrefixStore(str('rpc_prefix_{}'.format(_init_counter)), store)
            _init_counter += 1

        # Initialize autograd before RPC since _init_rpc_backend guarantees all
        # processes sync via the store. If we initialize autograd after RPC,
        # there could be a race where some nodes might have initialized autograd
        # and others might not have. As a result, a node calling
        # torch.distributed.autograd.backward() would run into errors since
        # other nodes might not have been initialized.
        dist_autograd._init(rank)

        _set_profiler_node_id(rank)
        # Initialize RPC.
        _init_rpc_backend(backend, store, name, rank, world_size, rpc_backend_options)


    def _validate_rpc_args(backend, store, name, rank, world_size, rpc_backend_options):
        type_mapping = {
            backend: backend_registry.BackendType,
            store: dist.Store,
            name: str,
            rank: numbers.Integral,
            world_size: numbers.Integral,
            rpc_backend_options: RpcBackendOptions,
        }
        for arg, arg_type in type_mapping.items():
            if not isinstance(arg, arg_type):
                raise RuntimeError(
                    "Argument {} must be of type {} but got type {}".format(
                        arg, arg_type, type(arg)
                    )
                )


    def _init_rpc_backend(
        backend=backend_registry.BackendType.TENSORPIPE,
        store=None,
        name=None,
        rank=-1,
        world_size=-1,
        rpc_backend_options=None,
    ):

        _validate_rpc_args(backend, store, name, rank, world_size, rpc_backend_options)

        if _is_current_rpc_agent_set():
            raise RuntimeError("RPC is already initialized")

        # Initialize RPC.
        rpc_agent = backend_registry.init_backend(
            backend,
            store=store,
            name=name,
            rank=rank,
            world_size=world_size,
            rpc_backend_options=rpc_backend_options,
        )

        api._init_rpc_states(rpc_agent)


    @api._require_initialized
    def _get_debug_info():
        from . import _rref_context_get_debug_info
        info = _rref_context_get_debug_info()
        info.update(api._get_current_rpc_agent().get_debug_info())
        info.update(dist_autograd._get_debug_info())
        return info
