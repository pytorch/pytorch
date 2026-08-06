# Owner(s): ["module: intel"]

import shutil
import subprocess
import tempfile
import unittest

from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    IS_LINUX,
    parametrize,
    run_tests,
    TestCase,
)


@unittest.skipIf(not IS_LINUX, "Only works on linux")
@instantiate_parametrized_tests
class TestTorchrun(TestCase):
    def setUp(self):
        super().setUp()
        self._test_dir = tempfile.mkdtemp(prefix=self.__class__.__name__)

    def tearDown(self):
        shutil.rmtree(self._test_dir)

    def test_cpu_info(self):
        lscpu_info = """# The following is the parsable format, which can be fed to other
# programs. Each different item in every column has an unique ID
# starting from zero.
# CPU,Core,Socket,Node
0,0,0,0
1,1,0,0
2,2,0,0
3,3,0,0
4,4,1,1
5,5,1,1
6,6,1,1
7,7,1,1
8,0,0,0
9,1,0,0
10,2,0,0
11,3,0,0
12,4,1,1
13,5,1,1
14,6,1,1
15,7,1,1
"""
        from torch.backends.xeon.run_cpu import _CPUinfo

        cpuinfo = _CPUinfo(lscpu_info)
        if cpuinfo._physical_core_nums() != 8:
            raise AssertionError(
                f"Expected 8 physical cores, got {cpuinfo._physical_core_nums()}"
            )
        if cpuinfo._logical_core_nums() != 16:
            raise AssertionError(
                f"Expected 16 logical cores, got {cpuinfo._logical_core_nums()}"
            )
        if cpuinfo.get_node_physical_cores(0) != [0, 1, 2, 3]:
            raise AssertionError(
                f"Expected [0, 1, 2, 3], got {cpuinfo.get_node_physical_cores(0)}"
            )
        if cpuinfo.get_node_physical_cores(1) != [4, 5, 6, 7]:
            raise AssertionError(
                f"Expected [4, 5, 6, 7], got {cpuinfo.get_node_physical_cores(1)}"
            )
        if cpuinfo.get_node_logical_cores(0) != [0, 1, 2, 3, 8, 9, 10, 11]:
            raise AssertionError(
                f"Expected [0, 1, 2, 3, 8, 9, 10, 11], got {cpuinfo.get_node_logical_cores(0)}"
            )
        if cpuinfo.get_node_logical_cores(1) != [4, 5, 6, 7, 12, 13, 14, 15]:
            raise AssertionError(
                f"Expected [4, 5, 6, 7, 12, 13, 14, 15], got {cpuinfo.get_node_logical_cores(1)}"
            )
        if cpuinfo.get_all_physical_cores() != [0, 1, 2, 3, 4, 5, 6, 7]:
            raise AssertionError(
                f"Expected [0, 1, 2, 3, 4, 5, 6, 7], got {cpuinfo.get_all_physical_cores()}"
            )
        expected_logical = [0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 6, 7, 12, 13, 14, 15]
        if cpuinfo.get_all_logical_cores() != expected_logical:
            raise AssertionError(
                f"Expected {expected_logical}, got {cpuinfo.get_all_logical_cores()}"
            )
        if cpuinfo.numa_aware_check([0, 1, 2, 3]) != [0]:
            raise AssertionError(
                f"Expected [0], got {cpuinfo.numa_aware_check([0, 1, 2, 3])}"
            )
        if cpuinfo.numa_aware_check([4, 5, 6, 7]) != [1]:
            raise AssertionError(
                f"Expected [1], got {cpuinfo.numa_aware_check([4, 5, 6, 7])}"
            )
        if cpuinfo.numa_aware_check([2, 3, 4, 5]) != [0, 1]:
            raise AssertionError(
                f"Expected [0, 1], got {cpuinfo.numa_aware_check([2, 3, 4, 5])}"
            )

    @parametrize(
        "core_list,expected",
        [
            ("0", [0]),
            ("0,1,2,3", [0, 1, 2, 3]),
            ("0, 1, 2, 3", [0, 1, 2, 3]),
            ("0-3", [0, 1, 2, 3]),
            ("5-5", [5]),
            ("0-31,65,92-94", list(range(32)) + [65, 92, 93, 94]),
            (" 0-2 , 8 ", [0, 1, 2, 8]),
            ("8,0-2", [0, 1, 2, 8]),
            ("0-2,1-3", [0, 1, 2, 3]),
            ("0,0,1", [0, 1]),
            ("0-3,", [0, 1, 2, 3]),
        ],
    )
    def test_parse_core_list(self, core_list, expected):
        from torch.backends.xeon.run_cpu import _parse_core_list

        self.assertEqual(_parse_core_list(core_list), expected)

    @parametrize(
        "core_list", ["", " ", ",", "3-0", "0-", "-3", "a", "0-a", "0--3", "1.5", "0 1"]
    )
    def test_parse_core_list_invalid(self, core_list):
        from torch.backends.xeon.run_cpu import _parse_core_list

        with self.assertRaises(ValueError):
            _parse_core_list(core_list)

    def test_core_list_ranges(self):
        cmds = []
        with subprocess.Popen(
            f"python -m torch.backends.xeon.run_cpu --core-list 0-1,3 --ncores-per-instance 2 \
            --use-default-allocator --disable-iomp --disable-numactl \
            --log-path {self._test_dir} --no-python pwd",
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        ) as p:
            for line in p.stdout.readlines():
                line = str(line, "utf-8").strip()
                if "taskset" in line:
                    cmds.append(line[line.index("taskset") :])
        # 3 cores with 2 cores per instance yields a single instance; the
        # contiguous pair collapses back into a range for taskset.
        self.assertEqual(len(cmds), 1)
        self.assertTrue(cmds[0].startswith("taskset -c 0-1 "), cmds[0])

    def test_multi_threads(self):
        num = 0
        with subprocess.Popen(
            f"python -m torch.backends.xeon.run_cpu --ninstances 4 --use-default-allocator \
            --disable-iomp --disable-numactl --disable-taskset --log-path {self._test_dir} --no-python pwd",
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        ) as p:
            for line in p.stdout.readlines():
                segs = str(line, "utf-8").strip().split("-")
                if segs[-1].strip() == "pwd":
                    num += 1
        if num != 4:
            raise AssertionError(
                f"Failed to launch multiple instances for inference, got {num}"
            )


if __name__ == "__main__":
    run_tests()
