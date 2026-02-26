"""
In NUMA (Non-Uniform Memory Access) systems, accessing memory on remote NUMA
nodes incurs additional latency. PyTorch provides NUMA binding utilities to
promote memory locality by binding worker processes to CPUs near their assigned GPUs.

In practice, NUMA binding typically results in 1-10% overall performance improvements,
but some workloads may obtain much greater benefits or none at all.

To enable NUMA binding, use the ``--numa-binding`` flag with :ref:`torchrun <launcher-api>`, e.g.:

.. code-block:: bash

    torchrun --numa-binding=node --nproc_per_node=8 train.py

Alternatively, pass :class:`NumaOptions` to ``LaunchConfig``
when using ``elastic_launch``.

See :class:`AffinityMode` for available binding modes.
"""

import os
import shutil
import traceback
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from enum import Enum
from functools import wraps
from logging import getLogger
from typing import ParamSpec, TypeVar

import torch
from torch._utils_internal import signpost_event


__all__ = [
    "AffinityMode",
    "NumaOptions",
]

logger = getLogger(__name__)


class AffinityMode(str, Enum):
    NODE = "node"
    """
    Each worker process and its threads will be bound to all the CPUs
    on the NUMA node containing the GPU whose local index equals the worker's local rank.
    If in doubt, use this option rather than the others.

    **Ex.:** If GPU 3 (i.e. ``torch.device("cuda:3")``) lives on NUMA node 1, then the worker
    whose local rank is 3 will only be able to run on the CPUs of NUMA node 1.
    """

    SOCKET = "socket"
    """
    Each worker process and its threads will be bound to all the CPUs on all the NUMA nodes of the
    socket containing the GPU whose local index equals the worker's local rank.

    **Ex.:** If socket 0 contains GPU 3 and NUMA nodes 0-1, then the worker whose
    local rank is 3 will be bound to the CPUs of NUMA nodes 0-1.

    For cases where there is only one NUMA node per socket anyway, this is equivalent to NODE.
    """

    EXCLUSIVE = "exclusive"
    """
    Each worker process and its threads will be bound to an exclusive subset of CPUs
    on the NUMA node containing the GPU whose local index equals the worker's local rank.
    The CPUs on the NUMA node are divided evenly among all GPUs on that node, so no two
    workers share the same CPU cores.

    **Ex.:** If NUMA node 1 has 16 physical cores and GPUs 2 and 3, then the worker whose
    local rank is 2 will be bound to cores 0-7, and the worker whose local rank is 3 will
    be bound to cores 8-15.
    """

    CORE_COMPLEX = "core-complex"
    """
    Each worker process and its threads will be bound to a single core complex (a group of cores
    sharing the same L3 cache) on the NUMA node containing the GPU whose local index equals
    the worker's local rank. Each worker is bound to a different core complex when possible.

    **Ex.:** If NUMA node 1 has two core complexes (cores 0-7 sharing one L3 cache, cores 8-15
    sharing another) and GPUs 2 and 3, then the worker whose local rank is 2 will be bound to
    cores 0-7, and the worker whose local rank is 3 will be bound to cores 8-15.
    """


@dataclass(frozen=True)
class NumaOptions:
    affinity_mode: AffinityMode

    should_fall_back_if_binding_fails: bool = False
    """
    If ``True``, we will silence any exceptions that occur during NUMA binding itself
    rather than raising them.

    There are no expected exceptions, so avoid using this option. Its purpose is simply
    to mitigate crash risk while conducting mass rollouts of NUMA binding.
    """


def _maybe_wrap_command_args_with_numa_binding(
    command_args: tuple[str, ...],
    *,
    gpu_index: int,
    numa_options: NumaOptions | None,
) -> tuple[str, ...]:
    """
    Wraps command arguments with numactl to apply NUMA CPU binding.

    This function prepends numactl with appropriate CPU affinity flags to the
    provided command arguments, binding the process to CPUs associated with
    the specified GPU's NUMA node.

    Args:
        command_args: The original command arguments to wrap.
        gpu_index: The index of the GPU that will be used by the subprocess.
        numa_options: Configuration for NUMA binding behavior. If None, returns
            the original command_args unchanged.

    Returns:
        Tuple of command arguments, potentially wrapped with numactl for NUMA binding.
        Returns the original command_args if numa_options is None or if binding fails
        and fallback is enabled.
    """
    if numa_options is None:
        return command_args

    kwargs = {
        "command_args": command_args,
        "gpu_index": gpu_index,
        "numa_options": asdict(numa_options),
    }

    try:
        logical_cpu_indices = _get_validated_logical_cpus_to_bind_to(
            gpu_index=gpu_index,
            numa_options=numa_options,
        )

        wrapped_command_args = _assemble_numactl_command_args(
            original_command_args=command_args,
            logical_cpu_indices=logical_cpu_indices,
        )
        signpost_event(
            category="numa_binding",
            name="apply_success",
            parameters={
                **kwargs,
                "wrapped_command": wrapped_command_args,
            },
        )
        return wrapped_command_args
    except Exception:
        # pyrefly: ignore [bad-argument-type]
        _handle_exception(numa_options=numa_options, logger_kwargs=kwargs)
        return command_args


_TParams = ParamSpec("_TParams")
_TReturn = TypeVar("_TReturn")


def _maybe_wrap_with_numa_binding(
    func: Callable[_TParams, _TReturn],
    *,
    gpu_index: int,
    numa_options: NumaOptions | None,
) -> Callable[_TParams, _TReturn]:
    """
    Wraps a function to apply NUMA CPU binding before execution.

    This decorator applies NUMA CPU affinity to all threads in the current process
    before calling the wrapped function, binding them to CPUs associated with the
    specified GPU's NUMA node.

    Args:
        func: The function to wrap with NUMA binding.
        gpu_index: The index of the GPU that will be used.
        numa_options: Configuration for NUMA binding behavior. If None, returns
            the original function unchanged.

    Returns:
        A wrapped function that applies NUMA binding before execution, or the
        original function if numa_options is None.
    """
    if numa_options is None:
        return func

    @wraps(func)
    def wrapped(*args: _TParams.args, **kwargs: _TParams.kwargs) -> _TReturn:
        _maybe_apply_numa_binding_to_current_process(
            gpu_index=gpu_index,
            # pyrefly: ignore [bad-argument-type]
            numa_options=numa_options,
        )
        return func(*args, **kwargs)

    return wrapped


def _maybe_apply_numa_binding_to_current_process(
    *, gpu_index: int, numa_options: NumaOptions
) -> None:
    kwargs = {
        "gpu_index": gpu_index,
        "numa_options": asdict(numa_options),
    }

    try:
        logical_cpu_indices = _get_validated_logical_cpus_to_bind_to(
            gpu_index=gpu_index,
            numa_options=numa_options,
        )

        _bind_all_threads_in_current_process_to_logical_cpus(
            logical_cpu_indices=logical_cpu_indices
        )

        signpost_event(
            category="numa_binding",
            name="apply_success",
            parameters={
                **kwargs,
                "logical_cpu_indices": _get_ranges_str_from_ints(logical_cpu_indices),
            },
        )
    except Exception:
        # pyrefly: ignore [bad-argument-type]
        _handle_exception(numa_options=numa_options, logger_kwargs=kwargs)


def _assemble_numactl_command_args(
    *, original_command_args: tuple[str, ...], logical_cpu_indices: set[int]
) -> tuple[str, ...]:
    return (
        "numactl",
        f"--physcpubind={_get_ranges_str_from_ints(logical_cpu_indices)}",
        *original_command_args,
    )


def _handle_exception(
    *, numa_options: NumaOptions, logger_kwargs: dict[str, object]
) -> None:
    signpost_event(
        category="numa_binding",
        name="apply_exception",
        parameters={
            **logger_kwargs,
            "traceback": traceback.format_exc(),
        },
    )
    logger.exception("Failed to apply NUMA binding for input=%r", logger_kwargs)
    if numa_options.should_fall_back_if_binding_fails:
        logger.warning(
            "Continuing executing without applying NUMA binding, despite exception %s",
            traceback.format_exc(),
        )
        return
    # This function is called within an except block, so silence the warning
    # about raise without an exception.
    raise  # noqa: PLE0704


def _get_validated_logical_cpus_to_bind_to(
    *,
    gpu_index: int,
    numa_options: NumaOptions,
) -> set[int]:
    logical_cpu_indices = _get_logical_cpus_to_bind_to(
        gpu_index=gpu_index, numa_options=numa_options
    )
    _raise_if_binding_invalid(logical_cpu_indices=logical_cpu_indices)

    return logical_cpu_indices


def _raise_if_binding_invalid(*, logical_cpu_indices: set[int]) -> None:
    # NOTE: numactl CLI is only actually necessary for the str entrypoint path,
    # but for simplicity we will just check it no matter what.
    if shutil.which("numactl") is None:
        raise RuntimeError("numactl CLI is required for NUMA binding")

    if not logical_cpu_indices:
        raise RuntimeError("Must bind to a non-empty set of CPU indices")


def _bind_all_threads_in_current_process_to_logical_cpus(
    *, logical_cpu_indices: set[int]
) -> None:
    # Save the original affinity of the main thread before changing it
    # pyrefly: ignore [missing-attribute]
    original_main_thread_affinity = os.sched_getaffinity(0)  # type: ignore[attr-defined]

    # 0 represents the current thread.
    # This is outside the try/except because the main thread should always bind successfully.
    # pyrefly: ignore [missing-attribute]
    os.sched_setaffinity(0, logical_cpu_indices)  # type: ignore[attr-defined]

    for tid_str in os.listdir("/proc/self/task"):
        try:
            tid = int(tid_str)
            # pyrefly: ignore [missing-attribute]
            tid_affinity = os.sched_getaffinity(tid)  # type: ignore[attr-defined]

            # Defensive check to ensure we do not overwrite affinity on any threads
            # that have already had their affinity set elsewhere.
            if tid_affinity == original_main_thread_affinity:
                # pyrefly: ignore [missing-attribute]
                os.sched_setaffinity(tid, logical_cpu_indices)  # type: ignore[attr-defined]
        except Exception:
            # Thread may have exited or otherwise become invalid
            pass


def _get_logical_cpus_to_bind_to(
    *,
    gpu_index: int,
    numa_options: NumaOptions,
) -> set[int]:
    """
    Args:
        gpu_index: The index of the GPU that will be used by the subprocess.
            Example: 0
        numa_options: See NumaOptions for details.

    Returns:
        Set of logical CPU indices to bind to.
    """
    if numa_options.affinity_mode == AffinityMode.NODE:
        logical_cpus = _node_get_logical_cpus_to_bind_to(gpu_index=gpu_index)
    elif numa_options.affinity_mode == AffinityMode.SOCKET:
        logical_cpus = _socket_get_logical_cpus_to_bind_to(gpu_index=gpu_index)
    elif numa_options.affinity_mode == AffinityMode.EXCLUSIVE:
        logical_cpus = _exclusive_get_logical_cpus_to_bind_to(gpu_index=gpu_index)
    elif numa_options.affinity_mode == AffinityMode.CORE_COMPLEX:
        logical_cpus = _core_complex_get_logical_cpus_to_bind_to(gpu_index=gpu_index)
    else:
        raise ValueError(f"Affinity mode {numa_options.affinity_mode} not supported.")

    return logical_cpus


def _node_get_logical_cpus_to_bind_to(*, gpu_index: int) -> set[int]:
    """
    Core logic of 'node' numa strategy.
    """
    numa_node_index = _get_numa_node_index_for_gpu_index(gpu_index=gpu_index)

    return _get_allowed_logical_cpu_indices_for_numa_node(
        numa_node_index=numa_node_index
    )


def _socket_get_logical_cpus_to_bind_to(*, gpu_index: int) -> set[int]:
    """
    Core logic of 'socket' numa strategy.
    """
    numa_node_index_of_gpu = _get_numa_node_index_for_gpu_index(gpu_index=gpu_index)
    socket_index = _get_socket_index_for_numa_node(
        numa_node_index=numa_node_index_of_gpu
    )
    numa_node_indices = _get_numa_node_indices_for_socket_index(
        socket_index=socket_index
    )

    logical_cpus = set()
    for numa_node_index in numa_node_indices:
        logical_cpus.update(
            _get_allowed_logical_cpu_indices_for_numa_node(
                numa_node_index=numa_node_index
            )
        )

    return logical_cpus


def _exclusive_get_logical_cpus_to_bind_to(*, gpu_index: int) -> set[int]:
    """
    Core logic of 'exclusive' numa strategy.
    """
    numa_node_index = _get_numa_node_index_for_gpu_index(gpu_index=gpu_index)

    gpu_indices = _get_gpu_indices_for_numa_node(numa_node_index=numa_node_index)
    gpu_indices = sorted(gpu_indices)
    original_gpu_relative_index = gpu_indices.index(gpu_index)

    allowed_logical_cpu_indices = _get_allowed_logical_cpu_indices_for_numa_node(
        numa_node_index=numa_node_index
    )

    # Arbitrarily use the min logical cpu index on the physical core to
    # represent the physical core.
    physical_core_to_allowed_logical_cpu_indices = _group_by(
        allowed_logical_cpu_indices,
        lambda logical_cpu_index: min(
            _get_logical_cpu_indices_sharing_same_physical_core_as(
                logical_cpu_index=logical_cpu_index
            )
        ),
    )
    # Sort the dict for consistency (dicts maintain order in Python)
    physical_core_to_allowed_logical_cpu_indices = dict(
        sorted(physical_core_to_allowed_logical_cpu_indices.items())
    )

    num_physical_cores_per_gpu = len(
        physical_core_to_allowed_logical_cpu_indices
    ) // len(gpu_indices)
    # Often, the number of physical cores will not be perfectly divisible by the number
    # of GPUs. In those cases, give the lowest GPU indices an extra core
    num_gpus_to_give_one_extra_physical_core = len(
        physical_core_to_allowed_logical_cpu_indices
    ) % len(gpu_indices)

    if num_physical_cores_per_gpu < 1:
        raise RuntimeError(
            f"There are only {len(physical_core_to_allowed_logical_cpu_indices)} physical cores on {numa_node_index=},"
            + f" but there are {len(gpu_indices)} GPUs associated with this NUMA node."
        )

    # Compute slice indices for this GPU
    start = original_gpu_relative_index * num_physical_cores_per_gpu + min(
        original_gpu_relative_index, num_gpus_to_give_one_extra_physical_core
    )
    end = (
        start
        + num_physical_cores_per_gpu
        + (
            1
            if original_gpu_relative_index < num_gpus_to_give_one_extra_physical_core
            else 0
        )
    )

    # Slice and flatten the logical CPUs from the selected physical cores
    logical_cpu_indices_for_original_gpu = {
        logical_cpu_index
        for logical_cpu_indices in list(
            physical_core_to_allowed_logical_cpu_indices.values()
        )[start:end]
        for logical_cpu_index in logical_cpu_indices
    }

    return logical_cpu_indices_for_original_gpu


def _core_complex_get_logical_cpus_to_bind_to(*, gpu_index: int) -> set[int]:
    """
    Core logic of 'core-complex' numa strategy.

    Each GPU is assigned a full core complex (group of cores sharing L3 cache)
    within its affined NUMA node.
    """
    numa_node_index = _get_numa_node_index_for_gpu_index(gpu_index=gpu_index)

    gpu_indices = _get_gpu_indices_for_numa_node(numa_node_index=numa_node_index)
    gpu_indices = sorted(gpu_indices)
    original_gpu_relative_index = gpu_indices.index(gpu_index)

    allowed_logical_cpu_indices = _get_allowed_logical_cpu_indices_for_numa_node(
        numa_node_index=numa_node_index
    )

    # Arbitrarily use the min logical cpu index on the max level cache
    # to represent the max level cache.
    max_level_cache_to_allowed_logical_cpu_indices = _group_by(
        allowed_logical_cpu_indices,
        lambda logical_cpu_index: min(
            _get_logical_cpus_sharing_same_max_level_cache_as(
                logical_cpu_index=logical_cpu_index
            )
        ),
    )

    max_level_cache_to_allowed_logical_cpu_indices = dict(
        sorted(
            max_level_cache_to_allowed_logical_cpu_indices.items(),
            # First, prioritize caches with more available cpus
            # Second, prioritize lower index cpus (just for clarity/consistency)
            key=lambda item: (-len(item[1]), item[0]),
        )
    )

    cache_index_for_original_gpu = original_gpu_relative_index % len(
        max_level_cache_to_allowed_logical_cpu_indices
    )
    logical_cpu_indices_for_original_gpu = list(
        max_level_cache_to_allowed_logical_cpu_indices.values()
    )[cache_index_for_original_gpu]

    return logical_cpu_indices_for_original_gpu


K = TypeVar("K")
V = TypeVar("V")


def _group_by(values: Iterable[V], get_key: Callable[[V], K]) -> dict[K, set[V]]:
    """
    Groups elements with same key into sets.
    """
    key_to_values: defaultdict[K, set[V]] = defaultdict(set)
    for value in values:
        key = get_key(value)
        key_to_values[key].add(value)
    return key_to_values


def _get_logical_cpu_indices_sharing_same_physical_core_as(
    *, logical_cpu_index: int
) -> set[int]:
    thread_siblings_list_absolute_path = (
        f"/sys/devices/system/cpu/cpu{logical_cpu_index}/topology/thread_siblings_list"
    )
    with open(thread_siblings_list_absolute_path) as f:
        return _get_set_of_int_from_ranges_str(f.read())


def _get_logical_cpus_sharing_same_max_level_cache_as(
    *, logical_cpu_index: int
) -> set[int]:
    cpu_cache_dir_absolute_path = (
        f"/sys/devices/system/cpu/cpu{logical_cpu_index}/cache"
    )

    max_level = -1
    logical_cpus_sharing_max_level_cache = set()
    for entry in os.listdir(cpu_cache_dir_absolute_path):
        if not entry.startswith("index") or not entry[5:].isdecimal():
            continue
        cache_index_absolute_path = os.path.join(cpu_cache_dir_absolute_path, entry)

        # Filter out other cache types like Instruction
        type_absolute_path = os.path.join(cache_index_absolute_path, "type")
        with open(type_absolute_path) as type_file:
            if type_file.read().strip() not in {"Unified", "Data"}:
                continue

        level_absolute_path = os.path.join(cache_index_absolute_path, "level")
        with open(level_absolute_path) as level_file:
            level = int(level_file.read())
        if level <= max_level:
            continue

        max_level = level
        shared_cpu_list_absolute_path = os.path.join(
            cache_index_absolute_path, "shared_cpu_list"
        )
        with open(shared_cpu_list_absolute_path) as share_cpu_list_file:
            logical_cpus_sharing_max_level_cache = _get_set_of_int_from_ranges_str(
                share_cpu_list_file.read()
            )

    return logical_cpus_sharing_max_level_cache


def _get_allowed_logical_cpu_indices_for_numa_node(*, numa_node_index: int) -> set[int]:
    all_cpu_indices = _get_cpu_indices_for_numa_node_MAYBE_NOT_ALLOWED(
        numa_node_index=numa_node_index
    )
    allowed_cpu_indices = _get_allowed_cpu_indices_for_current_thread()
    return all_cpu_indices & allowed_cpu_indices


def _get_cpu_indices_for_numa_node_MAYBE_NOT_ALLOWED(
    *, numa_node_index: int
) -> set[int]:
    """
    Returns:
        Indices of all CPUs associated with numa_node_index. However, the list
        is not filtered based on whether the thread is allowed to use them.
    """
    cpulist_absolute_path = f"/sys/devices/system/node/node{numa_node_index}/cpulist"
    try:
        with open(cpulist_absolute_path) as f:
            cpu_range_str = f.read()
    except FileNotFoundError as e:
        raise RuntimeError(
            f"Could not determine CPUs corresponding to {numa_node_index=}."
        ) from e
    return _get_set_of_int_from_ranges_str(cpu_range_str)


def _get_gpu_count() -> int:
    return torch.cuda.device_count()


def _get_numa_node_index_for_gpu_index(*, gpu_index: int) -> int:
    device_properties = torch.cuda.get_device_properties(gpu_index)

    domain = device_properties.pci_domain_id  # type: ignore[attr-defined]
    bus = device_properties.pci_bus_id  # type: ignore[attr-defined]
    device = device_properties.pci_device_id  # type: ignore[attr-defined]

    # Format to sysfs PCI address: "0000:dc:00.0"
    pci_addr = f"{domain:04x}:{bus:02x}:{device:02x}.0"

    pci_numa_node_absolute_path = f"/sys/bus/pci/devices/{pci_addr}/numa_node"
    with open(pci_numa_node_absolute_path) as f:
        # In systems with only one NUMA node, this will
        # often be saved as -1. In those cases, there is obviously
        # at least one numa node, 0, so we use that.
        return max(int(f.read().strip()), 0)


def _get_gpu_indices_for_numa_node(*, numa_node_index: int) -> set[int]:
    return {
        gpu_index
        for gpu_index in range(_get_gpu_count())
        if _get_numa_node_index_for_gpu_index(gpu_index=gpu_index) == numa_node_index
    }


def _get_socket_index_for_numa_node(*, numa_node_index: int) -> int:
    arbitrary_cpu_index = _get_arbitrary_allowed_cpu_index_for_numa_node(
        numa_node_index=numa_node_index
    )

    return _get_socket_index_for_cpu(cpu_index=arbitrary_cpu_index)


def _get_socket_index_for_cpu(*, cpu_index: int) -> int:
    package_id_absolute_path = (
        f"/sys/devices/system/cpu/cpu{cpu_index}/topology/physical_package_id"
    )
    try:
        with open(package_id_absolute_path) as f:
            return int(f.read().strip())
    except FileNotFoundError as e:
        raise RuntimeError(f"Could not determine socket for {cpu_index=}") from e


def _get_arbitrary_allowed_cpu_index_for_numa_node(*, numa_node_index: int) -> int:
    return min(
        _get_allowed_logical_cpu_indices_for_numa_node(numa_node_index=numa_node_index)
    )


def _get_set_of_int_from_ranges_str(ranges_str: str) -> set[int]:
    """
    Util for parsing a string of int ranges, as in a sysfs file.

    Args:
        ranges_str: E.g., "0-2,4,6-7"

    Returns:
        E.g., {0, 1, 2, 4, 6, 7}
    """
    ints: set[int] = set()
    for range_str in ranges_str.split(","):
        range_str = range_str.strip()
        if not range_str:
            continue
        if "-" in range_str:
            start_str, end_str = range_str.split("-")
            start, end = int(start_str), int(end_str)
            ints.update(range(start, end + 1))
        else:
            ints.add(int(range_str))
    return ints


def _get_ranges_str_from_ints(ints: Iterable[int]) -> str:
    """
    Convert a set of integers to a compact string with ranges.

    Args:
        ints: E.g., {0, 1, 2, 4, 6, 7}

    Returns:
        E.g., "0-2,4,6-7"
    """
    if not ints:
        return ""

    sorted_ints = sorted(ints)
    ranges = []
    start = prev = sorted_ints[0]

    for num in sorted_ints[1:]:
        if num == prev + 1:
            prev = num
        else:
            if start == prev:
                ranges.append(f"{start}")
            else:
                ranges.append(f"{start}-{prev}")
            start = prev = num

    # Append the last range
    if start == prev:
        ranges.append(f"{start}")
    else:
        ranges.append(f"{start}-{prev}")

    return ",".join(ranges)


def _get_systemwide_numa_node_indices() -> set[int]:
    with open("/sys/devices/system/node/possible") as f:
        possible_nodes_str = f.read()

    return _get_set_of_int_from_ranges_str(possible_nodes_str)


def _get_numa_node_indices_for_socket_index(*, socket_index: int) -> set[int]:
    systemwide_numa_node_indices = _get_systemwide_numa_node_indices()

    matching_numa_node_indices = set()
    for numa_node_index in systemwide_numa_node_indices:
        arbitrary_cpu_index = _get_arbitrary_allowed_cpu_index_for_numa_node(
            numa_node_index=numa_node_index
        )
        if socket_index == _get_socket_index_for_cpu(cpu_index=arbitrary_cpu_index):
            matching_numa_node_indices.add(numa_node_index)

    return matching_numa_node_indices


def _get_allowed_cpu_indices_for_current_thread() -> set[int]:
    # 0 denotes current thread
    # pyrefly: ignore [missing-attribute]
    return os.sched_getaffinity(0)  # type:ignore[attr-defined]
