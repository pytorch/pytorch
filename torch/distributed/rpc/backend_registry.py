
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

def _tensorpipe_check_local_device_maps(name, options):
    # Check local devices in device_maps and devices are all valid.
    local_devices = set(options.devices) if options.devices else set()
    device_maps = options.device_maps
    for worker_name in device_maps:
        device_map = device_maps[worker_name]
        key_set = set(device_map.keys())
        val_set = set(device_map.values())
        if not all([
            len(key_set) == len(device_map),
            len(val_set) == len(device_map),
        ]):
            raise ValueError(
                f"Invalid device_map configuration for {worker_name}, "
                f"not 1-to-1 mapping:\ndevice_maps = {device_map}"
            )
        local_devices.update(key_set)

    if not all(
        (0 <= d.index < torch.cuda.device_count() if d.type == "cuda" else True)
        for d in local_devices
    ):
        raise ValueError(
            f"Invalid device in TensorPipe options on {name}:\n"
            f"device_maps = {options.device_maps},\n"
            f"devices = {options.devices}"
        )

# detect if any worker has invalid device_map configurations, and return
# names of failed workers
def _tensorpipe_check_remote_device_maps(agent, options):
    device_maps = options.device_maps
    if device_maps is None:
        device_maps = {}

    def check_one_worker(name, device_maps, all_device_counts):
        device_count = all_device_counts[name]
        wrong_worker_names = set(device_maps) - set(all_device_counts)
        if wrong_worker_names:
            raise ValueError(f"Wrong worker names: {wrong_worker_names}")
        for remote_name in all_device_counts:
            remote_device_count = all_device_counts[remote_name]
            if remote_name in device_maps:
                device_map = device_maps[remote_name]
                val_set = set(device_map.values())
                if not all(
                    (0 <= d.index < remote_device_count if d.type == "cuda" else True)
                    for d in val_set
                ):
                    raise ValueError(
                        f"Invalid device_map configuration on {name} "
                        f"for {remote_name}, remote device out of range:\n"
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

    if world_size:
        is_static_group = True
    else:
        is_static_group = False

    # world_size is specified so this is a static group (ranks cannot join and leave)
    if is_static_group:
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
        # TODO: retrieve token from store to signal start of rank join/leave critical section
        token = f"TokenOnWorker{rank}"

        while True:
            returned = store.compare_set("init_rpc_token", "", token).decode()
            if returned == token:
                print(f"{rank} got token")
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

                # Validate devices and device_maps locally for current rank
                _tensorpipe_check_local_device_maps(name, rpc_backend_options)

                # Construct TPAgent with empty reverse_device_map and devices
                # these two properties will be updated after construction
                print(f"{rank}: begin construct TPAgent")
                agent = TensorPipeAgent(
                    store,
                    name,
                    rank,
                    world_size,
                    rpc_backend_options,
                    {},
                    [],
                )
                print(f"{rank}: finish construct TPAgent")

                try:
                    # TODO: Notify all workers in group this rank has joined and set devices and reverse_device_map
                    # This is a synchronous operation that completes once all existing ranks are updated
                    # _tensorpipe_check_remote_device_maps(agent, rpc_backend_options)
                    pass
                except Exception:
                    api.shutdown()
                    raise

                # finish initialization
                break
            else:
                from datetime import timedelta
                store.wait([returned], timedelta(seconds=15))

        # TODO: update from store to signal end of rank join/leave critical section
        store.set("init_rpc_token", "")
        store.set(token, "1")

        return agent

register_backend(
    "TENSORPIPE",
    _tensorpipe_construct_rpc_backend_options_handler,
    _tensorpipe_init_backend_handler,
)
