import os
import shutil
import subprocess
import traceback
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional, TypeVar

import torch
from torch._utils_internal import signpost_event
from torch.distributed.elastic.utils.logging import get_logger


__all__ = [
    "maybe_wrap_with_numa_bindings",
    "AffinityMode",
    "NumaOptions",
]


_NUMACTL_COMMAND = "numactl"

logger = get_logger(__file__)


class AffinityMode(str, Enum):
    """
    See behavior description for each affinity mode
    in torch.distributed.run.
    """

    NODE = "node"
    SOCKET = "socket"
    EXCLUSIVE = "exclusive"
    CORE_COMPLEX = "core-complex"


@dataclass(frozen=True)
class NumaOptions:
    affinity_mode: AffinityMode
    """
    If true, we will silently return the original command if any of the following occur:
    - An exception is raised as we compute the wrapped command.
    - During a dry run of the wrapped command, numactl fails for any reason.

    You should avoid using this option! It is only intended as a safety mechanism for facilitating
    mass rollouts of numa binding.
    """
    should_fall_back_if_binding_fails: bool = False


def maybe_wrap_with_numa_bindings(
    *,
    entrypoint: str,
    local_rank_to_args: dict[int, tuple],
    numa_options: Optional[NumaOptions],
) -> tuple[str, dict[int, tuple]]:
    """
    Args:
        entrypoint: The entrypoint to the program, such as might be input to Popen.
            Example: "python"
        local_rank_to_args: A mapping from local rank to args for the entrypoint.
            Example: {0: ("trainer.py",)}
        numa_options: See NumaOptions for details.

    Returns:
        A tuple of (entrypoint, local_rank_to_args), basically transforming the inputs,
        where the entrypoint and args may now involve numa binding.
        Example: ("numactl", {"0": ("--cpunodebind=0", "--preferred=0", "python", "trainer.py")})
    """
    if numa_options is None:
        return (entrypoint, local_rank_to_args)

    wrapped_local_rank_to_args = {}
    for local_rank, args in local_rank_to_args.items():
        try:
            numactl_command_options = _maybe_get_numactl_options(
                command_args=(entrypoint, *[str(arg) for arg in args]),
                gpu_index=local_rank,
                numa_options=numa_options,
            )
        except Exception:
            if numa_options.should_fall_back_if_binding_fails:
                # NOTE: If any element of the batch fails to apply NUMA bindings
                # for any reason, we do not apply NUMA bindings to any element of the batch,
                # for maximum safety. This only applies if fallback is enabled.
                return (entrypoint, local_rank_to_args)
            raise
        wrapped_local_rank_to_args[local_rank] = (
            *numactl_command_options,
            entrypoint,
            *args,
        )
    return (_NUMACTL_COMMAND, wrapped_local_rank_to_args)


def _maybe_get_numactl_options(
    *,
    command_args: tuple[str, ...],
    gpu_index: int,
    numa_options: NumaOptions,
) -> tuple[str, ...]:
    """
    Args:
        command_args: The args for a command, such as might be input to Popen.
            Example: ("python", "trainer.py")
        gpu_index: The index of the GPU that will be used by the subprocess which executes command_args.
            Example: 0
        numa_options: See NumaOptions for details.

    Returns:
        Depending on numa_options, something like
            ("--cpunodebind=0", "--preferred=0")
    """
    try:
        _raise_if_numactl_not_available()
        if numa_options.affinity_mode == AffinityMode.NODE:
            numactl_command_options = _get_node_numactl_options(gpu_index=gpu_index)
        elif numa_options.affinity_mode == AffinityMode.SOCKET:
            numactl_command_options = _get_socket_numactl_options(gpu_index=gpu_index)
        elif numa_options.affinity_mode == AffinityMode.EXCLUSIVE:
            numactl_command_options = _get_exclusive_numactl_options(
                gpu_index=gpu_index
            )
        elif numa_options.affinity_mode == AffinityMode.CORE_COMPLEX:
            numactl_command_options = _get_core_complex_numactl_options(
                gpu_index=gpu_index
            )
        else:
            raise ValueError(
                f"Affinity mode {numa_options.affinity_mode} not supported."
            )

        if numa_options.should_fall_back_if_binding_fails:
            _raise_if_numactl_fails_dry_run(numactl_options=numactl_command_options)
        signpost_event(
            category="numa_binding",
            name="wrap_command_success",
            parameters={
                "original_command_args": command_args,
                "gpu_index": gpu_index,
                "numa_options": numa_options,
                "numactl_command_options": numactl_command_options,
            },
        )
        return numactl_command_options
    except Exception:
        signpost_event(
            category="numa_binding",
            name="wrap_command_exception",
            parameters={
                "traceback": traceback.format_exc(),
                "original_command_args": command_args,
                "gpu_index": gpu_index,
                "numa_options": numa_options,
            },
        )
        logger.exception(
            """Failed to wrap command with NUMA bindings.
            Input:
                command_args=%r,
                gpu_index=%d,
                numa_options=%r,
        """,
            command_args,
            gpu_index,
            numa_options,
        )
        raise


def _raise_if_numactl_fails_dry_run(*, numactl_options: tuple[str, ...]) -> None:
    noop_args = _get_assembled_command_from_pieces(
        # Execute arbitrary noop
        command_args=("true",),
        numactl_options=numactl_options,
    )
    try:
        subprocess.run(
            noop_args,
            stdout=subprocess.DEVNULL,
            # These allow us to capture the stderr as text
            stderr=subprocess.PIPE,
            text=True,
            # Raise exception if nonzero exit status.
            check=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"""Our binding logic inferred to prepend your command with options {noop_args[:-1]}.
            Before doing that, we did a noop dry run with args {noop_args}, but that command failed.
            This should NOT happen, and likely suggests a bug in pytorch's numa binding logic.

            The {_NUMACTL_COMMAND} command itself had this stderr:

            {e.stderr}
            """
        ) from e


def _get_assembled_command_from_pieces(
    *, command_args: tuple[str, ...], numactl_options: tuple[str, ...]
) -> tuple[str, ...]:
    # Syntax for invoking a command but with numactl activated is numactl <args> command <args>
    return (_NUMACTL_COMMAND, *numactl_options, *command_args)


def _raise_if_numactl_not_available() -> None:
    if not shutil.which(_NUMACTL_COMMAND):
        raise RuntimeError(
            f"{_NUMACTL_COMMAND} shell command is required for NUMA bindings."
        )


def _get_node_numactl_options(*, gpu_index: int) -> tuple[str, ...]:
    """
    Core logic of 'node' numa strategy.

    Returns options to be used with numactl. E.g.,
    ("--cpunodebind=0", "--preferred=0").
    """
    numa_node_index = _get_numa_node_index_for_gpu_index(gpu_index=gpu_index)

    return (
        f"--cpunodebind={numa_node_index}",
        f"--preferred={numa_node_index}",
    )


def _get_socket_numactl_options(*, gpu_index: int) -> tuple[str, ...]:
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
    numa_node_indices_str = _get_ranges_str_from_ints(numa_node_indices)

    return (
        f"--cpunodebind={numa_node_indices_str}",
        (
            f"--preferred-many={numa_node_indices_str}"
            if len(numa_node_indices) > 1
            else f"--preferred={numa_node_indices_str}"
        ),
    )


def _get_exclusive_numactl_options(*, gpu_index: int) -> tuple[str, ...]:
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
    logical_cpu_indices_for_original_gpu = (
        logical_cpu_index
        for logical_cpu_indices in list(
            physical_core_to_allowed_logical_cpu_indices.values()
        )[start:end]
        for logical_cpu_index in logical_cpu_indices
    )

    return (
        f"--physcpubind={_get_ranges_str_from_ints(logical_cpu_indices_for_original_gpu)}",
        f"--preferred={numa_node_index}",
    )


def _get_core_complex_numactl_options(*, gpu_index: int) -> tuple[str, ...]:
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

    return (
        f"--physcpubind={_get_ranges_str_from_ints(logical_cpu_indices_for_original_gpu)}",
        f"--preferred={numa_node_index}",
    )


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
    allowed_cpu_indices = _get_allowed_cpu_indices_for_current_process()
    return all_cpu_indices & allowed_cpu_indices


def _get_cpu_indices_for_numa_node_MAYBE_NOT_ALLOWED(
    *, numa_node_index: int
) -> set[int]:
    """
    Returns:
        Indices of all CPUs associated with numa_node_index. However, the list
        is not filtered based on whether the process is allowed to use them.
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


def _get_allowed_cpu_indices_for_current_process() -> set[int]:
    # 0 denotes current process
    return os.sched_getaffinity(0)
