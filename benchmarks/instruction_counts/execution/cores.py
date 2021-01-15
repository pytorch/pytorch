"""Handle assignment of benchmark tasks to particular cores."""
import json
import multiprocessing
import re
import subprocess
import threading
from typing import Dict, List, Optional, Tuple


CPU_COUNT: int = multiprocessing.cpu_count()

# We don't want to completely saturate the CPU since the main process does some
# work as well. (And we don't want to contend with the benchmarks.) To account
# for this, we reserve at most `CPU_COUNT - SLACK` cores for benchmarks. In
# practice, completely saturating the CPU doesn't actually reduce overall time.
SLACK: int = min(CPU_COUNT - 1, int(CPU_COUNT * 0.15), 6)


def get_numa_information() -> Tuple[Tuple[int, int], ...]:
    """Determine CPU core layout.

    When benchmarking, it is important to consider CPU groupings in order to
    obtain low distortion results. Adjacent cores often share caches, while
    CPUs in different sockets have to go through slower communication paths.
    (QPI, HyperTransport, etc.) Similar asymmetries are also present when
    reading and writing to memory.

    If an A/B test pins one experiment to cores within a NUMA node and the
    other to cores spanning multiple nodes, it will systematically bias the
    comparison.

    The `numactl` command allows one to pin a process to a single NUMA node,
    however we want finer grained (per core) isolation as well. To that end,
    we directly extract and manage core placement to ensure sensible
    architecture specific behavior.
    """
    lscpu_proc = subprocess.run(
        ["lscpu", "--json"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        shell=False,
        timeout=1,
        encoding="utf-8",
    )
    assert not lscpu_proc.returncode

    info_json: Dict[str, List[Dict[str, str]]] = json.loads(lscpu_proc.stdout)
    numa_groups: List[Tuple[int, int]] = []
    i: Dict[str, str]
    for i in info_json["lscpu"]:
        if re.match(r"^NUMA node[0-9]+ CPU\(s\):$", i["field"]):
            for group_str in i["data"].split(","):
                group_match = re.match(r"^([0-9]+)-([0-9]+)$", group_str)
                assert group_match
                lower = int(group_match.groups()[0])
                upper = int(group_match.groups()[1])

                assert 0 <= lower < CPU_COUNT
                assert 0 <= upper < CPU_COUNT
                assert upper >= lower
                numa_groups.append((lower, upper))
    assert numa_groups, "Could not find any NUMA groups"
    return tuple(numa_groups)


class _NUMA_Node:
    def __init__(self, min_core_id: int, max_core_id: int) -> None:
        assert min_core_id >= 0
        assert max_core_id >= min_core_id
        assert max_core_id < CPU_COUNT

        self._min_core_id: int = min_core_id
        self._max_core_id: int = max_core_id
        self._num_cores = max_core_id - min_core_id + 1
        self._available: List[bool] = [
            True for _ in range(min_core_id, min_core_id + self._num_cores)]

        self._reservations: Dict[str, Tuple[int, ...]] = {}

    @property
    def num_available(self) -> int:
        return sum(self._available)

    def reserve(self, n: int) -> Optional[str]:
        """Simple first-fit policy.

        If successful, return a Tuple[int, int] of [lower_core, upper_core].
        Otherwise, return None. (Similar to the allocator pattern of returning
        nullptr if unable to allocate.)
        """
        for lower_index in range(self._num_cores - n + 1):
            indices = tuple(range(lower_index, lower_index + n))
            if all(self._available[i] for i in indices):
                for i in indices:
                    self._available[i] = False

                lower_core = indices[0] + self._min_core_id
                upper_core = indices[-1] + self._min_core_id
                key = f"{lower_core}-{upper_core}" if n > 1 else f"{lower_core}"
                self._reservations[key] = indices
                return key
        return None

    def release(self, key: str) -> None:
        for i in self._reservations[key]:
            self._available[i] = True
        self._reservations.pop(key)


class CorePool:
    def __init__(self, slack: int = SLACK) -> None:
        self._slack: int = slack
        numa_groups: Tuple[Tuple[int, int], ...]
        try:
            numa_groups = get_numa_information()
        except Exception as e:
            print(f"Failed to determine NUMA information: {e}")
            print("Benchmark will not be NUMA aware")
            numa_groups = ((0, CPU_COUNT - 1),)

        self._nodes: Tuple[_NUMA_Node, ...] = tuple(
            _NUMA_Node(min_core_id, max_core_id)
            for min_core_id, max_core_id in numa_groups
        )

        self._reservation_nodes: Dict[str, _NUMA_Node] = {}
        self._lock: threading.Lock = threading.Lock()

    def reserve(self, n: int) -> Optional[str]:
        """Try each NUMA node in order and see if a reservation is possible."""
        assert n > 0
        with self._lock:
            if sum(n.num_available for n in self._nodes) - n <= self._slack:
                return None

            for node in self._nodes:
                reservation: Optional[str] = node.reserve(n)
                if reservation is None:
                    continue

                self._reservation_nodes[reservation] = node
                return reservation
            return None

    def release(self, key: str) -> None:
        with self._lock:
            self._reservation_nodes[key].release(key)
