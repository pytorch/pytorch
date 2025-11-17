# Owner(s): ["oncall: distributed"]

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from typing import Any, Optional
from unittest import skipUnless
from unittest.mock import mock_open, patch

import torch
from torch._utils_internal import signpost_event
from torch.distributed.elastic.multiprocessing import DefaultLogsSpecs, start_processes
from torch.distributed.elastic.multiprocessing.api import _wrap
from torch.numa.binding import (
    _bind_all_threads_in_current_process_to_logical_cpus,
    _get_ranges_str_from_ints,
    _get_set_of_int_from_ranges_str,
    AffinityMode,
    NumaOptions,
)
from torch.testing._internal.common_utils import run_tests, TestCase


@dataclass(frozen=True)
class MockDeviceProperties:
    name: str
    major: int
    minor: int
    total_memory: str
    multi_processor_count: int
    uuid: str
    pci_bus_id: int
    pci_device_id: int
    pci_domain_id: int
    L2_cache_size: str


_real_open = open


@skipUnless(sys.platform == "linux", "Only linux currently supported")
@skipUnless(
    torch.distributed.is_available(), "Need access to some distributed submodules"
)
class NumaBindingTest(TestCase):
    def setUp(self) -> None:
        super().setUp()

        self._mock_file_path_to_contents: dict[str, str] = {}
        self._mock_device_properties: list[MockDeviceProperties] = []
        self._mock_num_logical_cpus = 0
        self._mock_num_numa_nodes = 0
        self._mock_num_sockets = 0

        self._context_managers_to_apply_to_all_tests = [
            patch("torch.cuda.device_count", self._mock_device_count),
            patch("torch.cuda.get_device_properties", self._mock_get_device_properties),
            patch("torch.cuda.is_available", self._mock_is_available),
            # Implicitly used by dynamo
            patch("torch.cuda.get_rng_state"),
            patch("builtins.open", new=self._mock_open),
            patch("os.listdir", new=self._mock_listdir),
            patch("os.sched_getaffinity", new=self._mock_sched_getaffinity),
            patch("torch.numa.binding.signpost_event", self._mock_signpost_event),
            patch("torch.numa.binding.shutil.which", return_value="/usr/bin/numactl"),
        ]

        for context_manager in self._context_managers_to_apply_to_all_tests:
            context_manager.__enter__()

    def tearDown(self) -> None:
        for context_manager in self._context_managers_to_apply_to_all_tests:
            context_manager.__exit__(None, None, None)
        super().tearDown()

    def _mock_signpost_event(self, *args, **kwargs) -> None:
        # Please keep these parameters JSON serializable for logging purposes
        json.dumps(kwargs["parameters"])
        return signpost_event(*args, **kwargs)

    def _add_mock_hardware(
        self,
        *,
        num_sockets: int,
        num_numa_nodes_per_socket: int,
        num_gpus_per_numa_node: int,
        num_l3_caches_per_numa_node: int,
        num_physical_core_per_l3_cache: int,
    ) -> None:
        """
        It's not fun, but we mock everything down to sysfs level
        to make sure we get really thorough coverage.
        """
        for socket_index in range(num_sockets):
            for numa_node_index in range(
                self._mock_num_numa_nodes,
                self._mock_num_numa_nodes + num_numa_nodes_per_socket,
            ):
                self._mock_file_contents(
                    file_path=f"/sys/devices/system/node/node{numa_node_index}/cpulist",
                    contents=f"{self._mock_num_logical_cpus}-"
                    + f"{self._mock_num_logical_cpus + num_l3_caches_per_numa_node * num_physical_core_per_l3_cache * 2 - 1}",
                )
                for gpu_index in range(
                    len(self._mock_device_properties),
                    len(self._mock_device_properties) + num_gpus_per_numa_node,
                ):
                    device_properties = MockDeviceProperties(
                        name=f"mock_gpu_{gpu_index}",
                        major=8,
                        minor=0,
                        total_memory="512GB",
                        multi_processor_count=256,
                        uuid=f"mock_gpu_uuid_{gpu_index}",
                        pci_bus_id=gpu_index,
                        pci_device_id=gpu_index,
                        pci_domain_id=gpu_index,
                        L2_cache_size="40MB",
                    )
                    self._mock_device_properties.append(device_properties)
                    pci_numa_node_path = (
                        self._get_corresponding_pci_numa_node_file_path(
                            device_properties=device_properties
                        )
                    )
                    self._mock_file_contents(
                        file_path=pci_numa_node_path,
                        contents=str(numa_node_index),
                    )

                for _ in range(num_l3_caches_per_numa_node):
                    lowest_logical_cpu_index_on_l3 = self._mock_num_logical_cpus
                    highest_logical_cpu_index_on_l3 = (
                        self._mock_num_logical_cpus
                        + 2 * num_physical_core_per_l3_cache
                        - 1
                    )
                    for logical_cpu_index in range(
                        self._mock_num_logical_cpus,
                        self._mock_num_logical_cpus
                        # Assume hyperthreaded
                        + 2 * num_physical_core_per_l3_cache,
                    ):
                        thread_siblings_range_str = (
                            f"{logical_cpu_index - 1}-{logical_cpu_index}"
                            if logical_cpu_index % 2
                            else f"{logical_cpu_index}-{logical_cpu_index + 1}"
                        )
                        self._mock_file_contents(
                            file_path=f"/sys/devices/system/cpu/cpu{logical_cpu_index}/topology/thread_siblings_list",
                            contents=thread_siblings_range_str,
                        )
                        # Unrelated file our logic should know to skip
                        self._mock_file_contents(
                            file_path=f"/sys/devices/system/cpu/cpu{logical_cpu_index}/cache/paulwuzhere",
                            contents="Data",
                        )
                        self._mock_file_contents(
                            file_path=f"/sys/devices/system/cpu/cpu{logical_cpu_index}/topology/physical_package_id",
                            contents=str(socket_index),
                        )
                        for cache_level in range(5):
                            self._mock_file_contents(
                                file_path=f"/sys/devices/system/cpu/cpu{logical_cpu_index}/cache/index{cache_level}/type",
                                contents="ShouldSkip" if cache_level == 4 else "Data",
                            )
                            self._mock_file_contents(
                                file_path=f"/sys/devices/system/cpu/cpu{logical_cpu_index}/cache/index{cache_level}/level",
                                contents=str(cache_level),
                            )
                            self._mock_file_contents(
                                file_path=f"/sys/devices/system/cpu/cpu{logical_cpu_index}/cache/index{cache_level}/shared_cpu_list",
                                contents=(
                                    f"{lowest_logical_cpu_index_on_l3}-{highest_logical_cpu_index_on_l3}"
                                    if cache_level == 3
                                    # Assume L1-2 are per physical core
                                    else thread_siblings_range_str
                                ),
                            )
                        self._mock_num_logical_cpus += 1
                self._mock_num_numa_nodes += 1
            self._mock_num_sockets += 1
        self._mock_file_contents(
            file_path="/sys/devices/system/node/possible",
            contents=f"0-{self._mock_num_numa_nodes - 1}",
        )

    def _mock_is_available(self) -> bool:
        return len(self._mock_device_properties) > 0

    def _get_corresponding_pci_numa_node_file_path(
        self, *, device_properties: MockDeviceProperties
    ) -> str:
        pci_addr = (
            f"{device_properties.pci_domain_id:04x}:"
            + f"{device_properties.pci_bus_id:02x}:{device_properties.pci_device_id:02x}.0"
        )
        return f"/sys/bus/pci/devices/{pci_addr}/numa_node"

    def _mock_file_contents(self, *, file_path: str, contents: str) -> None:
        self._mock_file_path_to_contents[file_path] = contents

    def _mock_device_count(self) -> int:
        return len(self._mock_device_properties)

    def _mock_get_device_properties(self, index: int) -> MockDeviceProperties:
        return self._mock_device_properties[index]

    def _mock_open(self, path: str, *args, **kwargs) -> Any:
        if path in self._mock_file_path_to_contents:
            return mock_open(read_data=self._mock_file_path_to_contents[path])()
        if isinstance(path, str) and path.startswith("/sys/"):
            raise FileNotFoundError(f"File {path} was not mocked.")
        # Looks like CI is calling open and intending to open an actual file in some places.
        # Need this to make the CI pass.
        return _real_open(path, *args, **kwargs)

    def _mock_listdir(self, target_path: str) -> set[str]:
        if not target_path.endswith("/"):
            target_path += "/"
        return {
            mock_path.split(target_path)[1].split("/")[0]
            for mock_path in self._mock_file_path_to_contents
            if mock_path.startswith(target_path)
        }

    def _mock_sched_getaffinity(self, pid: int) -> set[int]:
        return set(range(self._mock_num_logical_cpus))

    def _start_processes_for_str_entrypoint_and_get_command_args(
        self, *, numa_options: Optional[NumaOptions], target_local_rank: int
    ) -> tuple[str, ...]:
        with patch(
            "torch.distributed.elastic.multiprocessing.subprocess_handler.subprocess_handler.Popen"
        ) as mock_popen:
            start_processes(
                name="test_process",
                entrypoint="echo",
                args=dict.fromkeys(
                    range(self._mock_device_count()), ("Hello, world!",)
                ),
                envs={
                    i: {"LOCAL_RANK": str(i)} for i in range(self._mock_device_count())
                },
                logs_specs=DefaultLogsSpecs(),
                numa_options=numa_options,
            )
            call_args = next(
                call_args
                for call_args in mock_popen.call_args_list
                if call_args.kwargs.get("env", {}).get("LOCAL_RANK")
                == str(target_local_rank)
            )
            return call_args.kwargs["args"]

    def _start_processes_for_callable_entrypoint_and_get_sched_setaffinity_cpus(
        self, *, numa_options: Optional[NumaOptions], target_local_rank: int
    ) -> Optional[set[int]]:
        target_sched_setaffinity_logical_cpu_indices = None

        def mock_sched_setaffinity(*args, **kwargs) -> None:
            nonlocal target_sched_setaffinity_logical_cpu_indices
            target_sched_setaffinity_logical_cpu_indices = args[1]

        def dummy_fn():
            pass

        import torch.multiprocessing as mp

        ctx = mp.get_context()
        mock_queue = ctx.SimpleQueue()
        mock_event = ctx.Event()

        with patch("os.sched_setaffinity", mock_sched_setaffinity):
            mock_event.set()  # Prevent hanging
            # This is the entrypoint for subprocesses with Callable entrypoints
            _wrap(
                local_rank=target_local_rank,
                fn=dummy_fn,
                args={target_local_rank: ()},
                envs={target_local_rank: {}},
                stdout_redirects={target_local_rank: ""},
                stderr_redirects={target_local_rank: ""},
                ret_vals={target_local_rank: mock_queue},
                queue_finished_reading_event=mock_event,
                numa_options=numa_options,
            )

        return target_sched_setaffinity_logical_cpu_indices

    def test_node_numa_binding(self) -> None:
        self._add_mock_hardware(
            num_sockets=4,
            num_numa_nodes_per_socket=2,
            num_gpus_per_numa_node=2,
            num_l3_caches_per_numa_node=4,
            num_physical_core_per_l3_cache=2,
        )

        command_args = self._start_processes_for_str_entrypoint_and_get_command_args(
            numa_options=NumaOptions(affinity_mode=AffinityMode.NODE),
            target_local_rank=11,
        )
        self.assertEqual(
            command_args,
            # There are 8 numa nodes and 2 GPUs per numa node, so GPU 11 would be
            # on numa node 11 // 2 = 5.
            # Each numa node has 4 * 2 * 2 = 16 logical CPUs
            # Numa node 5 has CPUs 80-95
            ("numactl", "--physcpubind=80-95", "echo", "Hello, world!"),
        )

    def test_no_numa_binding_if_numa_options_not_provided(self) -> None:
        self._add_mock_hardware(
            num_sockets=4,
            num_numa_nodes_per_socket=2,
            num_gpus_per_numa_node=2,
            num_l3_caches_per_numa_node=4,
            num_physical_core_per_l3_cache=2,
        )

        command_args = self._start_processes_for_str_entrypoint_and_get_command_args(
            numa_options=None, target_local_rank=11
        )
        self.assertEqual(
            command_args,
            ("echo", "Hello, world!"),
        )

    def test_default_numa_binding(self) -> None:
        # Inner import to avoid crashing if not torch.distributed.is_available()
        from torch.distributed.launcher.api import LaunchConfig

        self._add_mock_hardware(
            num_sockets=1,
            num_numa_nodes_per_socket=1,
            num_gpus_per_numa_node=1,
            num_l3_caches_per_numa_node=1,
            num_physical_core_per_l3_cache=1,
        )

        with patch(
            "torch.distributed.launcher.api.get_default_numa_options",
            return_value=NumaOptions(
                affinity_mode=AffinityMode.NODE, should_fall_back_if_binding_fails=True
            ),
        ):
            launch_config = LaunchConfig(
                min_nodes=1,
                max_nodes=1,
                nproc_per_node=1,
                # Don't provide numa_options
            )
        self.assertEqual(
            launch_config.numa_options,
            NumaOptions(
                affinity_mode=AffinityMode.NODE, should_fall_back_if_binding_fails=True
            ),
        )

    def test_fallback(self) -> None:
        self._add_mock_hardware(
            num_sockets=2,
            num_numa_nodes_per_socket=1,
            num_gpus_per_numa_node=1,
            num_l3_caches_per_numa_node=1,
            num_physical_core_per_l3_cache=1,
        )

        with (
            patch("torch.numa.binding.signpost_event") as signpost_patch,
            patch(
                "torch.numa.binding._get_numa_node_index_for_gpu_index",
                side_effect=Exception("Mock exception!"),
            ),
        ):
            command_args = (
                self._start_processes_for_str_entrypoint_and_get_command_args(
                    numa_options=NumaOptions(
                        affinity_mode=AffinityMode.NODE,
                        should_fall_back_if_binding_fails=True,
                    ),
                    target_local_rank=0,
                )
            )
        self.assertIn(
            "Mock exception!",
            signpost_patch.call_args.kwargs["parameters"]["traceback"],
        )
        self.assertEqual(
            command_args,
            ("echo", "Hello, world!"),
        )

    def test_fallback_if_numactl_not_available(self) -> None:
        self._add_mock_hardware(
            num_sockets=2,
            num_numa_nodes_per_socket=1,
            num_gpus_per_numa_node=1,
            num_l3_caches_per_numa_node=1,
            num_physical_core_per_l3_cache=1,
        )

        with (
            patch("torch.numa.binding.signpost_event") as signpost_patch,
            patch("torch.numa.binding.shutil.which", return_value=None),
        ):
            command_args = (
                self._start_processes_for_str_entrypoint_and_get_command_args(
                    numa_options=NumaOptions(
                        affinity_mode=AffinityMode.NODE,
                        should_fall_back_if_binding_fails=True,
                    ),
                    target_local_rank=0,
                )
            )
        self.assertIn(
            "numactl CLI is required for NUMA binding",
            signpost_patch.call_args.kwargs["parameters"]["traceback"],
        )
        self.assertEqual(
            command_args,
            ("echo", "Hello, world!"),
        )

    def test_explicit_numa_options_overrides_default(self) -> None:
        # Inner import to avoid crashing if not torch.distributed.is_available()
        from torch.distributed.launcher.api import LaunchConfig

        with patch(
            "torch.distributed.launcher.api.get_default_numa_options",
            return_value=NumaOptions(affinity_mode=AffinityMode.NODE),
        ):
            launch_config = LaunchConfig(
                min_nodes=1,
                max_nodes=1,
                nproc_per_node=1,
                numa_options=NumaOptions(affinity_mode=AffinityMode.EXCLUSIVE),
            )
        self.assertEqual(
            launch_config.numa_options,
            NumaOptions(affinity_mode=AffinityMode.EXCLUSIVE),
        )

    def test_nproc_must_equal_cuda_device_count_to_use_default_numa_options(
        self,
    ) -> None:
        # Inner import to avoid crashing if not torch.distributed.is_available()
        from torch.distributed.launcher.api import LaunchConfig

        self._add_mock_hardware(
            num_sockets=1,
            num_numa_nodes_per_socket=1,
            num_gpus_per_numa_node=1,
            num_l3_caches_per_numa_node=1,
            num_physical_core_per_l3_cache=1,
        )

        with patch(
            "torch.distributed.launcher.api.get_default_numa_options"
        ) as mock_get_default_numa_options:
            launch_config = LaunchConfig(
                min_nodes=1,
                max_nodes=1,
                nproc_per_node=2,
            )
            mock_get_default_numa_options.assert_not_called()
            self.assertIsNone(launch_config.numa_options)

    def test_socket_numa_binding_with_multiple_numa_per_socket(self) -> None:
        self._add_mock_hardware(
            num_sockets=4,
            num_numa_nodes_per_socket=2,
            num_gpus_per_numa_node=2,
            num_l3_caches_per_numa_node=4,
            num_physical_core_per_l3_cache=2,
        )

        command_args = self._start_processes_for_str_entrypoint_and_get_command_args(
            numa_options=NumaOptions(affinity_mode=AffinityMode.SOCKET),
            target_local_rank=15,
        )
        self.assertEqual(
            command_args,
            # GPU 15 is on numa node 15 // 2 = 7, which is on socket 3 (numa nodes 6 and 7)
            # Each numa node has 4 * 2 * 2 = 16 logical CPUs
            # Numa nodes 6 and 7 have CPUs 96-111 and 112-127
            ("numactl", "--physcpubind=96-127", "echo", "Hello, world!"),
        )

    def test_socket_numa_binding_with_single_numa_per_socket(self) -> None:
        self._add_mock_hardware(
            num_sockets=4,
            num_numa_nodes_per_socket=1,
            num_gpus_per_numa_node=2,
            num_l3_caches_per_numa_node=4,
            num_physical_core_per_l3_cache=2,
        )

        command_args = self._start_processes_for_str_entrypoint_and_get_command_args(
            numa_options=NumaOptions(affinity_mode=AffinityMode.SOCKET),
            target_local_rank=7,
        )
        self.assertEqual(
            command_args,
            # GPU 7 is on numa node 7 // 2 = 3, which is socket 3 by itself
            # Each numa node has 4 * 2 * 2 = 16 logical CPUs
            # Numa node 3 has CPUs 48-63
            ("numactl", "--physcpubind=48-63", "echo", "Hello, world!"),
        )

    def test_exclusive_numa_binding(self) -> None:
        self._add_mock_hardware(
            num_sockets=2,
            num_numa_nodes_per_socket=1,
            num_gpus_per_numa_node=2,
            num_l3_caches_per_numa_node=1,
            num_physical_core_per_l3_cache=3,
        )

        command_args_0 = self._start_processes_for_str_entrypoint_and_get_command_args(
            numa_options=NumaOptions(affinity_mode=AffinityMode.EXCLUSIVE),
            target_local_rank=0,
        )
        self.assertEqual(
            command_args_0,
            # Gets an extra physical core due to odd number of physical cores on numa node
            # 3 physical cores total, 2 GPUs: GPU 0 gets 2 physical cores (CPUs 0-3)
            ("numactl", "--physcpubind=0-3", "echo", "Hello, world!"),
        )

        command_args_1 = self._start_processes_for_str_entrypoint_and_get_command_args(
            numa_options=NumaOptions(affinity_mode=AffinityMode.EXCLUSIVE),
            target_local_rank=1,
        )
        self.assertEqual(
            command_args_1,
            # Does not get an extra physical core, since the 1st GPU already took the extra.
            # GPU 1 gets 1 physical core (CPUs 4-5)
            ("numactl", "--physcpubind=4-5", "echo", "Hello, world!"),
        )

    def test_exclusive_raises_if_too_few_physical_cores(self) -> None:
        self._add_mock_hardware(
            num_sockets=2,
            num_numa_nodes_per_socket=1,
            num_gpus_per_numa_node=2,
            num_l3_caches_per_numa_node=1,
            num_physical_core_per_l3_cache=1,
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "There are only 1 physical cores on numa_node_index=0, but there are 2 GPUs associated with this NUMA node.",
        ):
            self._start_processes_for_str_entrypoint_and_get_command_args(
                numa_options=NumaOptions(affinity_mode=AffinityMode.EXCLUSIVE),
                target_local_rank=1,
            )

    def test_core_complex_numa_binding_with_extra_l3(self) -> None:
        self._add_mock_hardware(
            num_sockets=2,
            num_numa_nodes_per_socket=1,
            num_gpus_per_numa_node=2,
            num_l3_caches_per_numa_node=3,
            num_physical_core_per_l3_cache=3,
        )

        command_args = self._start_processes_for_str_entrypoint_and_get_command_args(
            numa_options=NumaOptions(affinity_mode=AffinityMode.CORE_COMPLEX),
            target_local_rank=3,
        )
        self.assertEqual(
            command_args,
            # GPU 3 is on numa node 3 // 2 = 1, relative GPU index is 3 % 2 = 1
            # The second L3 on the second numa node (numa node 1)
            # Second numa node starts at CPU 18, second L3 cache is CPUs 24-29
            ("numactl", "--physcpubind=24-29", "echo", "Hello, world!"),
        )

    def test_core_complex_numa_binding_with_fewer_l3_than_gpu(self) -> None:
        self._add_mock_hardware(
            num_sockets=2,
            num_numa_nodes_per_socket=1,
            num_gpus_per_numa_node=2,
            num_l3_caches_per_numa_node=1,
            num_physical_core_per_l3_cache=3,
        )

        command_args = self._start_processes_for_str_entrypoint_and_get_command_args(
            numa_options=NumaOptions(affinity_mode=AffinityMode.CORE_COMPLEX),
            target_local_rank=3,
        )
        self.assertEqual(
            command_args,
            # GPU 3 is on numa node 3 // 2 = 1, relative GPU index is 3 % 2 = 1
            # With 1 L3 cache per numa node, GPU 3 uses L3 cache index 1 % 1 = 0 (the only cache)
            # Second numa node starts at CPU 6, single L3 cache spans CPUs 6-11
            ("numactl", "--physcpubind=6-11", "echo", "Hello, world!"),
        )

    def test_core_complex_prefers_caches_with_more_cpus(self) -> None:
        self._add_mock_hardware(
            num_sockets=1,
            num_numa_nodes_per_socket=1,
            num_gpus_per_numa_node=1,
            num_l3_caches_per_numa_node=3,
            num_physical_core_per_l3_cache=3,
        )

        # Only some subset of the CPUs are available this time.
        with patch("os.sched_getaffinity", return_value={0, 4, 6, 7, 9}):
            command_args = (
                self._start_processes_for_str_entrypoint_and_get_command_args(
                    numa_options=NumaOptions(affinity_mode=AffinityMode.CORE_COMPLEX),
                    target_local_rank=0,
                )
            )

        # Binds to the second L3 because it has the most available CPUs
        self.assertEqual(
            command_args,
            ("numactl", "--physcpubind=6-7,9", "echo", "Hello, world!"),
        )

    def test_core_complex_tiebreak_prefers_lower_cache_key(self) -> None:
        """
        When several max‑level caches expose the same number of logical CPUs,
        prioritize binding to caches with lower cpu indices first.
        """
        self._add_mock_hardware(
            num_sockets=1,
            num_numa_nodes_per_socket=1,
            num_gpus_per_numa_node=1,
            num_l3_caches_per_numa_node=2,
            num_physical_core_per_l3_cache=1,
        )

        command_args = self._start_processes_for_str_entrypoint_and_get_command_args(
            numa_options=NumaOptions(affinity_mode=AffinityMode.CORE_COMPLEX),
            target_local_rank=0,
        )
        # 1 numa node, 2 L3 caches, 1 physical core per L3 cache = 2 logical CPUs per cache
        # L3 cache 0: CPUs 0-1, L3 cache 1: CPUs 2-3
        # Both have same number of CPUs, so prefer lower cache key (0)
        self.assertEqual(
            command_args,
            ("numactl", "--physcpubind=0-1", "echo", "Hello, world!"),
        )

    def test_binds_to_node_0_if_node_stored_as_minus_one(self) -> None:
        self._add_mock_hardware(
            num_sockets=1,
            num_numa_nodes_per_socket=1,
            num_gpus_per_numa_node=1,
            num_l3_caches_per_numa_node=1,
            num_physical_core_per_l3_cache=1,
        )

        device_0_properties = self._mock_get_device_properties(0)
        # Overwrite the existing mock file
        self._mock_file_contents(
            file_path=self._get_corresponding_pci_numa_node_file_path(
                device_properties=device_0_properties
            ),
            contents="-1",
        )

        command_args = self._start_processes_for_str_entrypoint_and_get_command_args(
            numa_options=NumaOptions(affinity_mode=AffinityMode.NODE),
            target_local_rank=0,
        )
        # GPU 0 has numa node stored as -1, which is treated as numa node 0
        # Each numa node has 1 * 1 * 2 = 2 logical CPUs
        # Numa node 0 has CPUs 0-1
        self.assertEqual(
            command_args,
            ("numactl", "--physcpubind=0-1", "echo", "Hello, world!"),
        )

    def test_callable_entrypoint_basic(self) -> None:
        self._add_mock_hardware(
            num_sockets=4,
            num_numa_nodes_per_socket=2,
            num_gpus_per_numa_node=2,
            num_l3_caches_per_numa_node=4,
            num_physical_core_per_l3_cache=2,
        )

        bound_logical_cpu_indices = self._start_processes_for_callable_entrypoint_and_get_sched_setaffinity_cpus(
            numa_options=NumaOptions(affinity_mode=AffinityMode.NODE),
            target_local_rank=11,
        )
        self.assertEqual(
            bound_logical_cpu_indices,
            # There are 8 numa nodes and 2 GPUs per numa node, so GPU 11 would be
            # on numa node 11 // 2 = 5.
            # Each numa node has 4 * 2 * 2 = 16 logical CPUs
            # Numa node 5 has CPUs 80-95
            set(range(80, 96)),
        )

    def test_raises_if_binding_to_empty_set(self) -> None:
        self._add_mock_hardware(
            num_sockets=1,
            num_numa_nodes_per_socket=1,
            num_gpus_per_numa_node=1,
            num_l3_caches_per_numa_node=1,
            num_physical_core_per_l3_cache=1,
        )

        with (
            patch(
                "torch.numa.binding._get_logical_cpus_to_bind_to", return_value=set()
            ),
            self.assertRaisesRegex(
                RuntimeError, "Must bind to a non-empty set of CPU indices"
            ),
        ):
            self._start_processes_for_callable_entrypoint_and_get_sched_setaffinity_cpus(
                numa_options=NumaOptions(affinity_mode=AffinityMode.NODE),
                target_local_rank=0,
            )

    def test_get_set_of_int_from_ranges_str(self) -> None:
        self.assertEqual(
            _get_set_of_int_from_ranges_str("0-2,4,6-7"), {0, 1, 2, 4, 6, 7}
        )

    def test_get_range_str_from_ints(self) -> None:
        self.assertEqual(_get_ranges_str_from_ints([7, 0, 1, 6, 2, 4]), "0-2,4,6-7")

    def test_bind_all_threads_in_current_process_to_logical_cpus(self) -> None:
        self._add_mock_hardware(
            num_sockets=1,
            num_numa_nodes_per_socket=1,
            num_gpus_per_numa_node=1,
            num_l3_caches_per_numa_node=1,
            num_physical_core_per_l3_cache=1,
        )

        self._mock_file_contents(file_path="/proc/self/task/8675309", contents="")
        self._mock_file_contents(file_path="/proc/self/task/1111111", contents="")
        self._mock_file_contents(
            # The exception from casting not_an_integer to int should get silenced.
            file_path="/proc/self/task/not_an_integer",
            contents="",
        )

        # Mock the original affinity for all threads
        original_main_thread_affinity = set(range(8))
        manually_reaffinitized_thread_affinity = {0}

        def mock_sched_getaffinity_impl(tid: int) -> set[int]:
            if tid == 1111111:
                return manually_reaffinitized_thread_affinity
            return original_main_thread_affinity

        call_order = []

        def mock_sched_setaffinity_impl(tid: int, cpus: set[int]) -> None:
            call_order.append((tid, cpus))

        with (
            patch("os.sched_getaffinity", side_effect=mock_sched_getaffinity_impl),
            patch(
                "os.sched_setaffinity", side_effect=mock_sched_setaffinity_impl
            ) as mock_sched_setaffinity,
        ):
            _bind_all_threads_in_current_process_to_logical_cpus(
                logical_cpu_indices={2, 0, 9}  # arbitrary
            )

        # Should set affinity for main thread (0) and thread 8675309 (same affinity as main)
        # Should NOT set affinity for thread 1111111 (manually reaffinitized)
        self.assertEqual(mock_sched_setaffinity.call_count, 2)
        mock_sched_setaffinity.assert_any_call(0, {2, 0, 9})
        mock_sched_setaffinity.assert_any_call(8675309, {2, 0, 9})

        # Verify that thread 1111111 was NOT reaffinitized
        for tid, _ in call_order:
            self.assertNotEqual(
                tid, 1111111, "Thread 1111111 should not have been reaffinitized"
            )


if __name__ == "__main__":
    run_tests()
