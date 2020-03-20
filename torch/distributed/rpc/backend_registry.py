from __future__ import absolute_import, division, print_function, unicode_literals

import collections
import datetime
import enum

import torch.distributed as dist
import torch.distributed.distributed_c10d as dc10d

from . import constants as rpc_constants


BackendValue = collections.namedtuple(
    "BackendValue", ["construct_rpc_backend_options_handler", "init_backend_handler"]
)


def _backend_type_repr(self):
    return "BackendType." + self.name


# Create an enum type, `BackendType`, with empty members.
BackendType = enum.Enum(value="BackendType", names={})
BackendType.__repr__ = _backend_type_repr


def register_backend(
    backend_name, construct_rpc_backend_options_handler, init_backend_handler
):
    """Registers a new RPC backend.

    Arguments:
        backend_name (str): backend string to identify the handler.
        construct_rpc_backend_options_handler (function):
            Handler that is invoked when
            rpc_backend.construct_rpc_backend_options(**dict) is called.
        init_backend_handler (function): Handler that is invoked when the
            `_init_rpc_backend()` function is called with a backend.
             This returns the agent.
    """
    global BackendType
    if backend_name in BackendType.__members__.keys():
        raise RuntimeError("RPC backend {}: already registered".format(backend_name))
    # Create a new enum type, `BackendType`, with extended members.
    existing_enum_dict = {member.name: member.value for member in BackendType}
    extended_enum_dict = dict(
        {
            backend_name: BackendValue(
                construct_rpc_backend_options_handler=construct_rpc_backend_options_handler,
                init_backend_handler=init_backend_handler,
            )
        },
        **existing_enum_dict
    )
    BackendType = enum.Enum(value="BackendType", names=extended_enum_dict)
    BackendType.__repr__ = _backend_type_repr
    return BackendType[backend_name]


def construct_rpc_backend_options(
    backend,
    rpc_timeout=rpc_constants.DEFAULT_RPC_TIMEOUT,
    init_method=rpc_constants.DEFAULT_INIT_METHOD,
    **kwargs
):
    if not isinstance(rpc_timeout, datetime.timedelta):
        raise RuntimeError("`rpc_timeout` must be a `datetime.timedelta`.")

    return backend.value.construct_rpc_backend_options_handler(
        rpc_timeout, init_method, **kwargs
    )


def init_backend(backend, *args, **kwargs):
    return backend.value.init_backend_handler(*args, **kwargs)


def _process_group_construct_rpc_backend_options_handler(
    rpc_timeout,
    init_method,
    num_send_recv_threads=rpc_constants.DEFAULT_NUM_SEND_RECV_THREADS,
    **kwargs
):
    from . import ProcessGroupRpcBackendOptions

    return ProcessGroupRpcBackendOptions(
        rpc_timeout=rpc_timeout,
        init_method=init_method,
        num_send_recv_threads=num_send_recv_threads
    )


def _process_group_init_backend_handler(
    store, name, rank, world_size, rpc_backend_options
):
    from . import ProcessGroupAgent

    # Initialize ProcessGroup.
    if dist.is_initialized():
        raise RuntimeError(
            "Default process group must not be initialized before init_rpc."
        )

    process_group_timeout = rpc_constants.DEFAULT_PROCESS_GROUP_TIMEOUT

    dist.init_process_group(
        backend=dist.Backend.GLOO,
        store=store,
        rank=rank,
        world_size=world_size,
        timeout=process_group_timeout,
    )

    try:
        group = dc10d._get_default_group()
        assert group is not None, "Failed to initialize default ProcessGroup."

        if (rank != -1) and (rank != group.rank()):
            raise RuntimeError(
                "rank argument {} doesn't match pg rank {}".format(rank, group.rank())
            )
        if (world_size != -1) and (world_size != group.size()):
            raise RuntimeError(
                "world_size argument {} doesn't match pg size {}".format(
                    world_size, group.size()
                )
            )
        # TODO: add try-except and destroy _agent in all processes if any fails.
        return ProcessGroupAgent(
            name,
            group,
            rpc_backend_options.num_send_recv_threads,
            rpc_backend_options.rpc_timeout,
        )
    except Exception as ex:
        dist.destroy_process_group()
        raise ex


register_backend(
    "PROCESS_GROUP",
    _process_group_construct_rpc_backend_options_handler,
    _process_group_init_backend_handler,
)
