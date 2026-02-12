# Owner(s): ["module: intel"]

import shutil
import subprocess
import tempfile
import unittest

from torch.testing._internal.common_utils import IS_LINUX, run_tests, TestCase


@unittest.skipIf(not IS_LINUX, "Only works on linux")
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
