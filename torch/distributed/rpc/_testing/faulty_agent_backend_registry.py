#!/usr/bin/env python3

import torch.distributed as dist
import torch.distributed.rpc as rpc

def _faulty_tensorpipe_construct_rpc_backend_options_handler(
    rpc_timeout,
    init_method,
    num_worker_threads,
    messages_to_fail,
    messages_to_delay,
    num_fail_sends,
    **kwargs
):
    from . import FaultyTensorPipeRpcBackendOptions

    return FaultyTensorPipeRpcBackendOptions(
        num_worker_threads=num_worker_threads,
        rpc_timeout=rpc_timeout,
        init_method=init_method,
        messages_to_fail=messages_to_fail,
        messages_to_delay=messages_to_delay,
        num_fail_sends=num_fail_sends,
    )


def _faulty_tensorpipe_init_backend_handler(
    store, name, rank, world_size, rpc_backend_options
):
    from . import FaultyTensorPipeAgent
    from . import FaultyTensorPipeRpcBackendOptions
    from torch.distributed.rpc import api

    if not isinstance(store, dist.Store):
        raise TypeError(f"`store` must be a c10d::Store. {store}")

    if not isinstance(
        rpc_backend_options, FaultyTensorPipeRpcBackendOptions
    ):
        raise TypeError(
            f"`rpc_backend_options` must be a `FaultyTensorPipeRpcBackendOptions`. {rpc_backend_options}"
        )

    agent = FaultyTensorPipeAgent(
        store,
        name,
        rank,
        world_size,
        rpc_backend_options,
        {},  # reverse_device_map
        [],  # devices
    )
    api._init_rpc_states(agent)

    return agent


rpc.backend_registry.register_backend(
    "FAULTY_TENSORPIPE",
    _faulty_tensorpipe_construct_rpc_backend_options_handler,
    _faulty_tensorpipe_init_backend_handler,
)
