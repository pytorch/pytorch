# mypy: allow-untyped-defs


import collections
import enum
from typing import cast

import torch
import torch.distributed as dist

from . import api, constants as rpc_constants
from ._utils import _group_membership_management, _update_group_membership


__all__ = [
    "backend_registered",
    "register_backend",
    "construct_rpc_backend_options",
    "init_backend",
    "BackendValue",
    "BackendType",
]

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
BackendType = enum.Enum(value="BackendType", names={})  # type: ignore[misc]
# Unable to assign a function a method (mypy bug #2427)
BackendType.__repr__ = _backend_type_repr  # type: ignore[assignment]

if BackendType.__doc__:
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
    return backend_name in BackendType.__members__


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
        raise RuntimeError(f"RPC backend {backend_name}: already registered")
    # Create a new enum type, `BackendType`, with extended members.
    existing_enum_dict = {member.name: member.value for member in BackendType}
    extended_enum_dict = dict(
        {
            backend_name: BackendValue(
                construct_rpc_backend_options_handler=construct_rpc_backend_options_handler,
                init_backend_handler=init_backend_handler,
            )
        },
        **existing_enum_dict,
    )
    # Can't handle Function Enum API (mypy bug #9079)
    BackendType = enum.Enum(value="BackendType", names=extended_enum_dict)  # type: ignore[misc]
    # Unable to assign a function a method (mypy bug #2427)
    BackendType.__repr__ = _backend_type_repr  # type: ignore[assignment]
    if BackendType.__doc__:
        BackendType.__doc__ = _backend_type_doc
    # pyrefly: ignore [unsupported-operation]
    return BackendType[backend_name]


def construct_rpc_backend_options(
    backend,
    rpc_timeout=rpc_constants.DEFAULT_RPC_TIMEOUT_SEC,
    init_method=rpc_constants.DEFAULT_INIT_METHOD,
    **kwargs,
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
        raise RuntimeError(f"rank argument {rank} doesn't match pg rank {group.rank()}")
    if (world_size != -1) and (world_size != group.size()):
        raise RuntimeError(
            f"world_size argument {world_size} doesn't match pg size {group.size()}"
        )
    return group


def _tensorpipe_construct_rpc_backend_options_handler(
    rpc_timeout,
    init_method,
    num_worker_threads=rpc_constants.DEFAULT_NUM_WORKER_THREADS,
    _transports=None,
    _channels=None,
    **kwargs,
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
    gathered: list[
        tuple[str, int, dict[str, dict[torch.device, torch.device]], list[torch.device]]
    ] = [("", 0, {}, []) for _ in range(group.size())]
    dist.all_gather_object(
        gathered, (my_name, my_device_count, my_device_maps, my_devices), group
    )
    all_names = [name for name, _, _, _ in gathered]
    all_device_counts = {name: count for name, count, _, _ in gathered}
    all_device_maps = {name: map_ for name, _, map_, _ in gathered}
    all_devices = {name: devices for name, _, _, devices in gathered}

    _validate_device_maps(all_names, all_device_counts, all_device_maps, all_devices)

    # passed all checked, construct reverse mapping and get list of devices handled by this agent
    reverse_device_maps = _create_reverse_mapping(my_name, all_names, all_device_maps)
    my_devices = _create_device_list(my_devices, my_device_maps, reverse_device_maps)
    return reverse_device_maps, my_devices


def _validate_device_maps(
    all_names, all_device_counts, all_device_maps, all_devices, is_static_group=True
):
    for node in all_names:
        devices = all_devices[node]
        if len(set(devices)) != len(devices):
            raise ValueError(f"Node {node} has duplicated devices\ndevices = {devices}")
        if not _tensorpipe_validate_devices(devices, all_device_counts[node]):
            raise ValueError(
                f"Node {node} has devices with invalid indices\n"
                f"devices = {devices}\n"
                f"device count = {all_device_counts[node]}"
            )

    for source_node in all_names:
        # For dynamic group (non-static) do not check the target node name since it may not have joined yet
        if is_static_group and not set(all_device_maps[source_node].keys()).issubset(
            all_names
        ):
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
            if all_devices.get(target_node, []):
                if not set(map_.values()).issubset(all_devices[target_node]):
                    raise ValueError(
                        f"Node {source_node} has unexpected target devices "
                        f"in its device map for {target_node}\n"
                        f"device map = {map_}\n"
                        f"devices = {all_devices[target_node]}"
                    )
            elif target_node in all_device_counts and not _tensorpipe_validate_devices(
                map_.values(), all_device_counts[target_node]
            ):
                raise ValueError(
                    f"Node {source_node} has target devices with invalid indices "
                    f"in its device map for {target_node}\n"
                    f"device map = {map_}\n"
                    f"device count = {all_device_counts[target_node]}"
                )


def _create_device_list(my_devices, my_device_maps, reverse_device_maps):
    if not my_devices:
        devices_set: set[torch.device] = set()
        for map_ in my_device_maps.values():
            devices_set.update(map_.keys())
        for map_ in reverse_device_maps.values():
            devices_set.update(map_.keys())
        devices_set.discard(torch.device("cpu"))
        my_devices = list(devices_set)
    my_devices = sorted(my_devices, key=lambda d: d.index)
    return my_devices


def _create_reverse_mapping(my_name, all_names, all_device_maps):
    reverse_device_maps: dict[str, dict[torch.device, torch.device]] = {}
    for node in all_names:
        if my_name in all_device_maps[node]:
            reverse_device_maps[node] = {
                v: k for k, v in all_device_maps[node][my_name].items()
            }
    return reverse_device_maps


def _get_device_infos():
    from . import TensorPipeAgent

    agent = cast(TensorPipeAgent, api._get_current_rpc_agent())
    opts = agent._get_backend_options()
    device_count = torch.cuda.device_count()
    if torch.cuda.is_available() and opts.devices:
        torch.cuda.init()
    return device_count, opts.device_maps, opts.devices


def _set_devices_and_reverse_device_map(agent):
    from . import TensorPipeAgent

    agent = cast(TensorPipeAgent, agent)
    # Group state is retrieved from local agent
    # On initialization, tensorpipe agent retrieves information from all existing workers, so group state is valid
    my_worker_info = agent.get_worker_info()
    my_name = my_worker_info.name
    all_worker_infos = agent.get_worker_infos()
    # One round to get device_maps of all workers and construct reverse device maps
    all_device_counts, all_device_maps, all_devices, all_names = {}, {}, {}, []
    for worker_info in all_worker_infos:
        worker_name = worker_info.name
        if worker_name != my_name:
            # TODO: make async?
            device_count, device_map, devices = api.rpc_sync(
                worker_name, _get_device_infos
            )
        else:
            opts = agent._get_backend_options()
            device_count, device_map, devices = (
                torch.cuda.device_count(),
                opts.device_maps,
                opts.devices,
            )
        all_device_counts[worker_name] = device_count
        all_device_maps[worker_name] = device_map
        all_devices[worker_name] = devices
        all_names.append(worker_name)

    _validate_device_maps(
        all_names,
        all_device_counts,
        all_device_maps,
        all_devices,
        is_static_group=False,
    )
    reverse_device_maps = _create_reverse_mapping(my_name, all_names, all_device_maps)

    # Perform RPC call to all workers, including itself, to include newly joined worker information and device maps
    for worker_name in all_names:
        # Set device list for each worker
        all_devices[worker_name] = _create_device_list(
            all_devices[worker_name], all_device_maps[worker_name], reverse_device_maps
        )
        api.rpc_sync(
            worker_name,
            _update_group_membership,
            args=(my_worker_info, all_devices[worker_name], reverse_device_maps, True),
        )


def _tensorpipe_init_backend_handler(
    store, name, rank, world_size, rpc_backend_options
):
    from . import TensorPipeAgent, TensorPipeRpcBackendOptions

    if not isinstance(store, dist.Store):
        raise TypeError(f"`store` must be a c10d::Store. {store}")

    if not isinstance(rpc_backend_options, TensorPipeRpcBackendOptions):
        raise TypeError(
            f"`rpc_backend_options` must be a `TensorPipeRpcBackendOptions`. {rpc_backend_options}"
        )

    device_count = torch.cuda.device_count()

    is_static_group = bool(world_size)
    # world_size is specified so this is a static group (ranks cannot join and leave)
    if is_static_group:
        # The agent's join method is required to behave like a barrier and perform
        # collective operations, for which it relies on a process group, instead of
        # re-implementing this on top of RPCs.
        group = _init_process_group(store, rank, world_size)

        reverse_device_maps, devices = _tensorpipe_exchange_and_check_all_device_maps(
            name,
            device_count,
            rpc_backend_options.device_maps,
            rpc_backend_options.devices,
            group,
        )

        if torch.cuda.is_available() and devices:
            # It's necessary to initialize PyTorch CUDA states here (e.g.,
            # CUDACachingAllocator). If this is missing, we could hit errors like
            # "allocator not initialized", because other processes might send
            # CUDA-related RPC request to this process before user code in this
            # process initializes its PyTorch CUDA states.
            torch.cuda.init()

        # TODO: add try-except and destroy _agent in all processes if any fails.
        agent = TensorPipeAgent(
            store,
            name,
            rank,
            world_size,
            rpc_backend_options,
            reverse_device_maps,
            devices,
        )

        api._init_rpc_states(agent)

        # Run one dummy round of RPC to initialize channels/transports. Without
        # this, it's easy to hit timeout in rpc.shutdown() if there is no other RPC
        # on that process before rpc.shutdown(), as the agent initialization can
        # take longer than 5s.
        api._all_gather(None, timeout=rpc_backend_options.rpc_timeout)
        # Need a barrier here to make sure no peers leave before the rank0 finishes
        # _all_gather
        group.barrier().wait()

        return agent
    # initialization for dynamic rpc (ranks can join and leave)
    else:
        with _group_membership_management(store, name, True):
            # Construct TPAgent with empty reverse_device_map and devices
            # these properties will be updated after initialization
            agent = TensorPipeAgent(
                store,
                name,
                rank,
                world_size,
                rpc_backend_options,
                {},
                [],
            )
            api._init_rpc_states(agent)

            try:
                # Notify all workers in group this rank has joined and set devices and reverse_device_map
                # This is a synchronous operation that completes once all existing ranks are updated
                _set_devices_and_reverse_device_map(agent)
            except Exception:
                api.shutdown()
                raise
            return agent


register_backend(
    "TENSORPIPE",
    _tensorpipe_construct_rpc_backend_options_handler,
    _tensorpipe_init_backend_handler,
)
