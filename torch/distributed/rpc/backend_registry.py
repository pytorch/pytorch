
import collections
import enum
from typing import Dict, List, Set, Tuple

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

    PyTorch ships with a builtin ``BackendType.TENSORPIPE`` backend.
    Additional ones can be registered using the
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

# Create a placeholder for PROCESS_GROUP BackendType
# which is deprecated. The backend type can be accessed, but
# will throw an error when the user tries to use it in init_rpc
register_backend(
    "PROCESS_GROUP",
    lambda *args: None,
    lambda *args: None,
)

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


def _tensorpipe_validate_devices(devices, device_count):
    return all(
        d.type == "cpu" or (d.type == "cuda" and 0 <= d.index < device_count)
        for d in devices
    )


# detect if any worker has invalid device_map configurations, and return
# reverse device maps
def _tensorpipe_exchange_and_check_all_device_maps(
    my_name, my_device_count, my_device_maps, my_devices, group
):
    gathered: List[Tuple[
        str, int, Dict[str, Dict[torch.device, torch.device]], List[torch.device]
    ]] = [("", 0, {}, []) for _ in range(group.size())]
    dist.all_gather_object(
        gathered, (my_name, my_device_count, my_device_maps, my_devices), group
    )
    all_names = [name for name, _, _, _ in gathered]
    all_device_counts = {name: count for name, count, _, _ in gathered}
    all_device_maps = {name: map_ for name, _, map_, _ in gathered}
    all_devices = {name: devices for name, _, _, devices in gathered}

    for node in all_names:
        devices = all_devices[node]
        if len(set(devices)) != len(devices):
            raise ValueError(
                f"Node {node} has duplicated devices\n"
                f"devices = {devices}"
            )
        if not _tensorpipe_validate_devices(devices, all_device_counts[node]):
            raise ValueError(
                f"Node {node} has devices with invalid indices\n"
                f"devices = {devices}\n"
                f"device count = {all_device_counts[node]}"
            )

    for source_node in all_names:
        if not set(all_device_maps[source_node].keys()).issubset(all_names):
            raise ValueError(
                f"Node {source_node} has invalid target node names in its device maps\n"
                f"device maps = {all_device_maps[source_node].keys()}\n"
                f"node names = {all_names}"
            )
        for target_node, map_ in all_device_maps[source_node].items():
            if len(set(map_.values())) != len(map_):
                raise ValueError(
                    f"Node {source_node} has duplicated target devices "
                    f"in its device map for {target_node}\n"
                    f"device map = {map_}"
                )
            if all_devices[source_node]:
                if not set(map_.keys()).issubset(all_devices[source_node]):
                    raise ValueError(
                        f"Node {source_node} has unexpected source devices "
                        f"in its device map for {target_node}\n"
                        f"device map = {map_}\n"
                        f"devices = {all_devices[source_node]}"
                    )
            elif not _tensorpipe_validate_devices(
                map_.keys(), all_device_counts[source_node]
            ):
                raise ValueError(
                    f"Node {source_node} has source devices with invalid indices "
                    f"in its device map for {target_node}\n"
                    f"device map = {map_}\n"
                    f"device count = {all_device_counts[source_node]}"
                )
            if all_devices[target_node]:
                if not set(map_.values()).issubset(all_devices[target_node]):
                    raise ValueError(
                        f"Node {source_node} has unexpected target devices "
                        f"in its device map for {target_node}\n"
                        f"device map = {map_}\n"
                        f"devices = {all_devices[target_node]}"
                    )
            elif not _tensorpipe_validate_devices(
                map_.values(), all_device_counts[target_node]
            ):
                raise ValueError(
                    f"Node {source_node} has target devices with invalid indices "
                    f"in its device map for {target_node}\n"
                    f"device map = {map_}\n"
                    f"device count = {all_device_counts[target_node]}"
                )

    # passed all checked, construct reverse mapping for return values
    reverse_device_maps: Dict[str, Dict[torch.device, torch.device]] = {}
    for node in all_names:
        if my_name in all_device_maps[node]:
            reverse_device_maps[node] = {
                v: k for k, v in all_device_maps[node][my_name].items()
            }

    if not my_devices:
        devices_set: Set[torch.device] = set()
        for _, map_ in my_device_maps.items():
            devices_set.update(map_.keys())
        for _, map_ in reverse_device_maps.items():
            devices_set.update(map_.keys())
        devices_set.discard(torch.device("cpu"))
        my_devices = list(devices_set)
    my_devices = sorted(my_devices, key=lambda d: d.index)

    return reverse_device_maps, my_devices


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

    # The agent's join method is required to behave like a barrier and perform
    # collective operations, for which it relies on a process group, instead of
    # re-implementing this on top of RPCs.

    group = _init_process_group(store, rank, world_size)

    if torch.cuda.is_available():
        # It's necessary to initialize PyTorch CUDA states here (e.g.,
        # CUDACachingAllocator). If this is missing, we could hit errors like
        # "allocator not initialized", because other processes might send
        # CUDA-related RPC request to this process before user code in this
        # process initializes its PyTorch CUDA states.
        torch.cuda.init()
        device_count = torch.cuda.device_count()
    else:
        device_count = 0

    reverse_device_maps, devices = _tensorpipe_exchange_and_check_all_device_maps(
        name,
        device_count,
        rpc_backend_options.device_maps,
        rpc_backend_options.devices,
        group,
    )

    # TODO: add try-except and destroy _agent in all processes if any fails.
    agent = TensorPipeAgent(
        store,
        name,
        rank,
        world_size,
        group,
        rpc_backend_options,
        reverse_device_maps,
        devices,
    )

    api._init_rpc_states(agent)

    # Run one dummy round of RPC to initialize channels/transports. Without
    # this, it's easy to hit timeout in rpc.shutdown() if there is no other RPC
    # on that process before rpc.shutdown(), as the agent initialization can
    # take longer than 5s.
    api._all_gather(None, timeout=rpc_constants.DEFAULT_RPC_TIMEOUT_SEC)
    # Need a barrier here to make sure no peers leave before the rank0 finishes
    # _all_gather
    group.barrier().wait()

    return agent


register_backend(
    "TENSORPIPE",
    _tensorpipe_construct_rpc_backend_options_handler,
    _tensorpipe_init_backend_handler,
)
