# Owner(s): ["module: intel"]

import shutil
import subprocess
import tempfile
import unittest

from torch.testing._internal.common_utils import IS_LINUX, run_tests, TestCase


@unittest.skipIf(not IS_LINUX, "Only works on linux")
class TestTorchrun(TestCase):
    def setUp(self):
        self._test_dir = tempfile.mkdtemp(prefix=self.__class__.__name__)

    def tearDown(self):
        shutil.rmtree(self._test_dir)

    def test_cpu_info(self):
        lscpu_info = """
CPU NODE SOCKET CORE L1d:L1i:L2:L3 ONLINE    MAXMHZ   MINMHZ      MHZ
  0    0      0    0 0:0:0:0          yes 5000.0000 800.0000 2400.000
  1    0      0    1 0:0:0:0          yes 5000.0000 800.0000 2400.000
  2    0      0    2 0:0:0:0          yes 5000.0000 800.0000 2400.000
  3    0      0    3 0:0:0:0          yes 5000.0000 800.0000 2400.000
  4    1      1    4 0:0:0:0          yes 5000.0000 800.0000 2400.000
  5    1      1    5 0:0:0:0          yes 5000.0000 800.0000 2400.000
  6    1      1    6 0:0:0:0          yes 5000.0000 800.0000 2400.000
  7    1      1    7 0:0:0:0          yes 5000.0000 800.0000 2400.000
  8    0      0    0 0:0:0:0          yes 5000.0000 800.0000 2400.000
  9    0      0    1 0:0:0:0          yes 5000.0000 800.0000 2400.000
 10    0      0    2 0:0:0:0          yes 5000.0000 800.0000 2400.000
 11    0      0    3 0:0:0:0          yes 5000.0000 800.0000 2400.000
 12    1      1    4 0:0:0:0          yes 5000.0000 800.0000 2400.000
 13    1      1    5 0:0:0:0          yes 5000.0000 800.0000 2400.000
 14    1      1    6 0:0:0:0          yes 5000.0000 800.0000 2400.000
 15    1      1    7 0:0:0:0          yes 5000.0000 800.0000 2400.000
"""
        from torch.backends.xeon._cpu_info import CPUPoolList

        cpupool = CPUPoolList(lscpu_txt=lscpu_info)
        assert [c.cpu for c in cpupool.pool_all] == [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,  # noqa: Q003
            9,
            10,
            11,
            12,
            13,
            14,
            15,
        ]
        assert [c.cpu for c in cpupool.pool_all if c.is_physical_core] == [
            0,
            1,
            2,
            3,
            4,
            5,  # noqa: Q003
            6,
            7,
        ]
        assert [c.cpu for c in cpupool.pool_all if c.node == 0] == [
            0,
            1,
            2,
            3,
            8,
            9,
            10,
            11,
        ]
        assert [c.cpu for c in cpupool.pool_all if c.node == 1] == [
            4,
            5,
            6,
            7,
            12,
            13,
            14,
            15,
        ]
        assert [
            c.cpu for c in cpupool.pool_all if c.node == 0 and c.is_physical_core
        ] == [0, 1, 2, 3]
        assert [
            c.cpu for c in cpupool.pool_all if c.node == 1 and c.is_physical_core
        ] == [4, 5, 6, 7]

    def test_multi_threads_module(self):
        num = 0
        with subprocess.Popen(
            f'python -m torch.backends.xeon.run_cpu --ninstances 4 --memory-allocator default \
            --omp-runtime default --multi-task-manager none --log-dir {self._test_dir} --no-python echo "test"',
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        ) as p:
            for line in p.stdout.readlines():
                segs = str(line, "utf-8").strip().split(":")
                if segs[-1].strip() == "test":
                    num += 1
        assert num == 4, "Failed to launch multiple instances for inference"

    def test_multi_threads_command(self):
        num = 0
        with subprocess.Popen(
            f'torch-xeon-launcher --ninstances 4 --memory-allocator default \
            --omp-runtime default --multi-task-manager none --log-dir {self._test_dir} --no-python echo "test"',
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        ) as p:
            for line in p.stdout.readlines():
                segs = str(line, "utf-8").strip().split(":")
                if segs[-1].strip() == "test":
                    num += 1
        assert num == 4, "Failed to launch multiple instances for inference"


if __name__ == "__main__":
    run_tests()
