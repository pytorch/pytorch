
import collections
from datetime import timedelta
import enum

import torch
import torch.distributed as dist

from . import api
from . import constants as rpc_constants


BackendValue = collections.namedtuple(
    "BackendValue", ["construct_rpc_backend_options_handler", "init_backend_handler"]
)


def _backend_type_repr(self):
    return "BackendType." + self.name


_backend_type_doc = """
    An enum class of available backends.

    PyTorch ships with two builtin backends: ``BackendType.TENSORPIPE`` and
    ``BackendType.PROCESS_GROUP``. Additional ones can be registered using the
    :func:`~torch.distributed.rpc.backend_registry.register_backend` function.
"""

# Create an enum type, `BackendType`, with empty members.
# Can't handle Function Enum API (mypy bug #9079)
BackendType = enum.Enum(value="BackendType", names=dict())  # type: ignore[misc]
# Unable to assign a function a method (mypy bug #2427)
BackendType.__repr__ = _backend_type_repr  # type: ignore[assignment]
BackendType.__doc__ = _backend_type_doc

def backend_registered(backend_name):
    """
    Checks if backend_name is registered as an RPC backend.

    Args:
        backend_name (str): string to identify the RPC backend.
    Returns:
        True if the backend has been registered with ``register_backend``, else
        False.
    """
    return backend_name in BackendType.__members__.keys()


def register_backend(
    backend_name, construct_rpc_backend_options_handler, init_backend_handler
):
    """Registers a new RPC backend.

    Args:
        backend_name (str): backend string to identify the handler.
        construct_rpc_backend_options_handler (function):
            Handler that is invoked when
            rpc_backend.construct_rpc_backend_options(**dict) is called.
        init_backend_handler (function): Handler that is invoked when the
            `_init_rpc_backend()` function is called with a backend.
             This returns the agent.
    """
    global BackendType
    if backend_registered(backend_name):
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
    # Can't handle Function Enum API (mypy bug #9079)
    BackendType = enum.Enum(value="BackendType", names=extended_enum_dict)  # type: ignore[misc]
    # Unable to assign a function a method (mypy bug #2427)
    BackendType.__repr__ = _backend_type_repr  # type: ignore[assignment]
    BackendType.__doc__ = _backend_type_doc
    return BackendType[backend_name]


def construct_rpc_backend_options(
    backend,
    rpc_timeout=rpc_constants.DEFAULT_RPC_TIMEOUT_SEC,
    init_method=rpc_constants.DEFAULT_INIT_METHOD,
    **kwargs
):

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

def _init_process_group(store, rank, world_size):
    # Initialize ProcessGroup.
    process_group_timeout = rpc_constants.DEFAULT_PROCESS_GROUP_TIMEOUT

    # We're using a bunch of private APIs here since `new_group` requires the
    # default group to be initialized.
    group = dist.ProcessGroupGloo(store, rank, world_size, process_group_timeout)

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
    return group

def _process_group_init_backend_handler(
    store, name, rank, world_size, rpc_backend_options
):
    from . import ProcessGroupRpcBackendOptions
    from . import ProcessGroupAgent

    if not isinstance(store, dist.Store):
        raise TypeError("`store` must be a c10d::Store. {}".format(store))

    if not isinstance(
        rpc_backend_options, ProcessGroupRpcBackendOptions
    ):
        raise TypeError(
            "`rpc_backend_options` must be a `ProcessGroupRpcBackendOptions`. {}".format(
                rpc_backend_options
            )
        )

    group = _init_process_group(store, rank, world_size)

    # TODO: add try-except and destroy _agent in all processes if any fails.
    return ProcessGroupAgent(
        store,
        name,
        group,
        rpc_backend_options.num_send_recv_threads,
        timedelta(seconds=rpc_backend_options.rpc_timeout),
    )


register_backend(
    "PROCESS_GROUP",
    _process_group_construct_rpc_backend_options_handler,
    _process_group_init_backend_handler,
)

def _tensorpipe_construct_rpc_backend_options_handler(
    rpc_timeout,
    init_method,
    num_worker_threads=rpc_constants.DEFAULT_NUM_WORKER_THREADS,
    _transports=None,
    _channels=None,
    **kwargs
):
    from . import TensorPipeRpcBackendOptions

    return TensorPipeRpcBackendOptions(
        rpc_timeout=rpc_timeout,
        init_method=init_method,
        num_worker_threads=num_worker_threads,
        _transports=_transports,
        _channels=_channels,
    )


# detect if any worker has invalid device_map configurations, and return
# names of failed workers
def _tensorpipe_check_device_maps(agent, device_maps):
    if device_maps is None:
        device_maps = {}

    def check_one_worker(name, device_maps, all_device_counts):
        device_count = all_device_counts[name]
        wrong_worker_names = set(device_maps) - set(all_device_counts)
        if wrong_worker_names:
            raise ValueError(f"Wrong worker names: {wrong_worker_names}")
        for worker_name in all_device_counts:
            remote_device_count = all_device_counts[worker_name]
            if worker_name in device_maps:
                device_map = device_maps[worker_name]
                key_set = set(device_map.keys())
                val_set = set(device_map.values())
                if not all([
                    len(device_map) == len(key_set),
                    len(device_map) == len(val_set),  # check 1-to-1 mapping
                    min(key_set) >= 0,
                    max(key_set) < device_count,  # check local range
                    min(val_set) >= 0,
                    max(val_set) < remote_device_count  # check remote range
                ]):
                    raise ValueError(
                        f"Invalid device_map configuration on {name}:\n"
                        f"device_maps = {device_maps}"
                    )

    gathered = api._all_gather([torch.cuda.device_count(), device_maps])
    all_device_counts = {name: gathered[name][0] for name in gathered}
    all_device_maps = {name: gathered[name][1] for name in gathered}
    for worker_name in all_device_maps:
        worker_device_maps = all_device_maps[worker_name]
        check_one_worker(worker_name, worker_device_maps, all_device_counts)

    # passed all checked, construct reverse mapping for return values
    reverse_device_maps = {}
    local_name = api.get_worker_info().name
    for worker_name in all_device_maps:
        remote_device_maps = all_device_maps[worker_name]
        if local_name in remote_device_maps:
            remote_device_map = remote_device_maps[local_name]
            reverse_device_maps[worker_name] = {
                remote_device_map[k]: k for k in remote_device_map
            }

    agent._set_reverse_device_maps(reverse_device_maps)


def _tensorpipe_init_backend_handler(store, name, rank, world_size, rpc_backend_options):
    from . import TensorPipeRpcBackendOptions
    from . import TensorPipeAgent

    if not isinstance(store, dist.Store):
        raise TypeError("`store` must be a c10d::Store. {}".format(store))

    if not isinstance(
        rpc_backend_options, TensorPipeRpcBackendOptions
    ):
        raise TypeError(
            "`rpc_backend_options` must be a `TensorPipeRpcBackendOptions`. {}".format(
                rpc_backend_options
            )
        )

    if torch.cuda.is_available():
        # It's necessary to initialize PyTorch CUDA states here (e.g.,
        # CUDACachingAllocator). If this is missing, we could hit errors like
        # "allocator not initialized", because other processes might send
        # CUDA-related RPC request to this process before user code in this
        # process initializes its PyTorch CUDA states.
        torch.cuda.init()

    # The agent's join method is required to behave like a barrier and perform
    # collective operations, for which it relies on a process group, instead of
    # re-implementing this on top of RPCs.

    group = _init_process_group(store, rank, world_size)

    # TODO: add try-except and destroy _agent in all processes if any fails.
    agent = TensorPipeAgent(
        store, name, rank, world_size, group, rpc_backend_options
    )

    api._init_rpc_states(agent)

    try:
        _tensorpipe_check_device_maps(agent, rpc_backend_options.device_maps)
        agent.join()
    except Exception:
        api.shutdown()
        raise

    return agent


register_backend(
    "TENSORPIPE",
    _tensorpipe_construct_rpc_backend_options_handler,
    _tensorpipe_init_backend_handler,
)
