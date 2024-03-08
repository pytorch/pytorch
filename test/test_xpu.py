# Owner(s): ["module: intel"]

import sys
import unittest

import torch
from torch.testing._internal.common_utils import NoTest, run_tests, TEST_XPU, TestCase

if not TEST_XPU:
    print("XPU not available, skipping tests", file=sys.stderr)
    TestCase = NoTest  # noqa: F811

TEST_MULTIXPU = torch.xpu.device_count() > 1

cpu_device = torch.device("cpu")
xpu_device = torch.device("xpu")

class TestXpu(TestCase):
    def test_device_behavior(self):
        current_device = torch.xpu.current_device()
        torch.xpu.set_device(current_device)
        self.assertEqual(current_device, torch.xpu.current_device())

    @unittest.skipIf(not TEST_MULTIXPU, "only one GPU detected")
    def test_multi_device_behavior(self):
        current_device = torch.xpu.current_device()
        target_device = (current_device + 1) % torch.xpu.device_count()

        with torch.xpu.device(target_device):
            self.assertEqual(target_device, torch.xpu.current_device())
        self.assertEqual(current_device, torch.xpu.current_device())

        with torch.xpu._DeviceGuard(target_device):
            self.assertEqual(target_device, torch.xpu.current_device())
        self.assertEqual(current_device, torch.xpu.current_device())

    def test_get_device_properties(self):
        current_device = torch.xpu.current_device()
        device_properties = torch.xpu.get_device_properties(current_device)
        self.assertEqual(device_properties, torch.xpu.get_device_properties(None))
        self.assertEqual(device_properties, torch.xpu.get_device_properties())

        device_name = torch.xpu.get_device_name(current_device)
        self.assertEqual(device_name, torch.xpu.get_device_name(None))
        self.assertEqual(device_name, torch.xpu.get_device_name())

        device_capability = torch.xpu.get_device_capability(current_device)
        self.assertTrue(device_capability["max_work_group_size"] > 0)
        self.assertTrue(device_capability["max_num_sub_groups"] > 0)

    def test_wrong_xpu_fork(self):
        stderr = TestCase.runWithPytorchAPIUsageStderr(
            """\
import torch
from torch.multiprocessing import Process
def run(rank):
    torch.xpu.set_device(rank)
if __name__ == "__main__":
    size = 2
    processes = []
    for rank in range(size):
        # it would work fine without the line below
        torch.xpu.set_device(0)
        p = Process(target=run, args=(rank,))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
"""
        )
        self.assertRegex(stderr, "Cannot re-initialize XPU in forked subprocess.")

    def test_streams(self):
        s0 = torch.xpu.Stream()
        torch.xpu.set_stream(s0)
        s1 = torch.xpu.current_stream()
        self.assertEqual(s0, s1)
        s2 = torch.xpu.Stream()
        self.assertFalse(s0 == s2)
        torch.xpu.set_stream(s2)
        with torch.xpu.stream(s0):
            self.assertEqual(s0, torch.xpu.current_stream())
        self.assertEqual(s2, torch.xpu.current_stream())

    def test_stream_priority(self):
        low, high = torch.xpu.Stream.priority_range()
        s0 = torch.xpu.Stream(device=0, priority=low)

        self.assertEqual(low, s0.priority)
        self.assertEqual(torch.device("xpu:0"), s0.device)

        s1 = torch.xpu.Stream(device=0, priority=high)

        self.assertEqual(high, s1.priority)
        self.assertEqual(torch.device("xpu:0"), s1.device)

    def test_stream_event_repr(self):
        s = torch.xpu.current_stream()
        self.assertTrue("torch.xpu.Stream" in str(s))
        e = torch.xpu.Event()
        self.assertTrue("torch.xpu.Event(uninitialized)" in str(e))
        s.record_event(e)
        self.assertTrue("torch.xpu.Event" in str(e))

    def test_events(self):
        stream = torch.xpu.current_stream()
        event = torch.xpu.Event()
        self.assertTrue(event.query())
        stream.record_event(event)
        event.synchronize()
        self.assertTrue(event.query())

    def test_generator(self):
        torch.manual_seed(2024)
        g_state0 = torch.xpu.get_rng_state()
        torch.manual_seed(1234)
        g_state1 = torch.xpu.get_rng_state()
        self.assertNotEqual(g_state0, g_state1)

        torch.xpu.manual_seed(2024)
        g_state2 = torch.xpu.get_rng_state()
        self.assertEqual(g_state0, g_state2)

        torch.xpu.set_rng_state(g_state1)
        self.assertEqual(g_state1, torch.xpu.get_rng_state())

        torch.manual_seed(1234)
        torch.xpu.set_rng_state(g_state0)
        self.assertEqual(2024, torch.xpu.initial_seed())

    # Simple cases for operators
    # TODO: Reusing PyTorch test_ops.py to improve coverage

    # Binary
    def test_add(self, dtype=torch.float):
        a_cpu = torch.randn(2, 3)
        b_cpu = torch.randn(2, 3)
        a_xpu = a_cpu.to(xpu_device)
        b_xpu = b_cpu.to(xpu_device)
        c_xpu = a_xpu + b_xpu
        c_cpu = a_cpu + b_cpu
        self.assertEqual(c_cpu, c_xpu.to(cpu_device))

    def test_sub(self, dtype=torch.float):
        a_cpu = torch.randn(2, 3)
        b_cpu = torch.randn(2, 3)
        a_xpu = a_cpu.to(xpu_device)
        b_xpu = b_cpu.to(xpu_device)
        c_xpu = a_xpu - b_xpu
        c_cpu = a_cpu - b_cpu
        self.assertEqual(c_cpu, c_xpu.to(cpu_device))

    def test_mul(self, dtype=torch.float):
        a_cpu = torch.randn(2, 3)
        b_cpu = torch.randn(2, 3)
        a_xpu = a_cpu.to(xpu_device)
        b_xpu = b_cpu.to(xpu_device)
        c_xpu = a_xpu * b_xpu
        c_cpu = a_cpu * b_cpu
        self.assertEqual(c_cpu, c_xpu.to(cpu_device))

    def test_div(self, dtype=torch.float):
        a_cpu = torch.randn(2, 3)
        b_cpu = torch.randn(2, 3)
        a_xpu = a_cpu.to(xpu_device)
        b_xpu = b_cpu.to(xpu_device)
        c_xpu = a_xpu / b_xpu
        c_cpu = a_cpu / b_cpu
        self.assertEqual(c_cpu, c_xpu.to(cpu_device))

    # Resize/View
    def test_resize(self, dtype=torch.float):
        x = torch.ones([2, 2, 4, 3], device=xpu_device, dtype=dtype)
        x.resize_(1, 2, 3, 4)
        y = torch.ones([2, 2, 4, 3], device=cpu_device, dtype=dtype)
        y.resize_(1, 2, 3, 4)
        self.assertEqual(y, x.cpu())

    def test_view_as_real(self, dtype=torch.cfloat):
        a_cpu = torch.randn(2, 3, 4, dtype=dtype)
        a_xpu = a_cpu.to(xpu_device)
        b_cpu = torch.view_as_real(a_cpu)
        b_xpu = torch.view_as_real(a_xpu)
        self.assertEqual(b_cpu, b_xpu.to(cpu_device))

    def test_view_as_complex(self, dtype=torch.float):
        a_cpu = torch.randn(109, 2, dtype=dtype)
        a_xpu = a_cpu.to(xpu_device)
        b_cpu = torch.view_as_complex(a_cpu)
        b_xpu = torch.view_as_complex(a_xpu)
        self.assertEqual(b_cpu, b_xpu.to(cpu_device))

    def test_view(self, dtype=torch.float):
        a_cpu = torch.randn(2, 3, 4, dtype=dtype)
        a_xpu = a_cpu.to(xpu_device)
        a_xpu = a_xpu.view(4, 3, 2)
        b_xpu = torch.full_like(a_xpu, 1)
        c_cpu = torch.ones([4, 3, 2])
        assert b_xpu.shape[0] == 4
        assert b_xpu.shape[1] == 3
        self.assertEqual(c_cpu, b_xpu.to(cpu_device))

    # Unary
    def test_abs(self, dtype=torch.float):
        data = [
            [
                -0.2911,
                -1.3204,
                -2.6425,
                -2.4644,
                -0.6018,
                -0.0839,
                -0.1322,
                -0.4713,
                -0.3586,
                -0.8882,
                0.0000,
                0.0000,
                1.1111,
                2.2222,
                3.3333,
            ]
        ]
        excepted = [
            [
                0.2911,
                1.3204,
                2.6425,
                2.4644,
                0.6018,
                0.0839,
                0.1322,
                0.4713,
                0.3586,
                0.8882,
                0.0000,
                0.0000,
                1.1111,
                2.2222,
                3.3333,
            ]
        ]
        x_dpcpp = torch.tensor(data, device=xpu_device)
        y = torch.tensor(excepted, device=xpu_device)
        y_dpcpp = torch.abs(x_dpcpp)
        self.assertEqual(y.to(cpu_device), y_dpcpp.to(cpu_device))

    # Copy/Clone
    def test_copy_and_clone(self, dtype=torch.float):
        a_cpu = torch.randn(16, 64, 28, 28)
        b_cpu = torch.randn(16, 64, 28, 28)
        a_xpu = a_cpu.to(xpu_device)
        b_xpu = b_cpu.to(xpu_device)
        # naive
        b_cpu.copy_(a_cpu)
        b_xpu.copy_(a_xpu)
        self.assertEqual(b_cpu, b_xpu.to(cpu_device))
        # clone + permutation
        b_cpu = a_cpu.clone(memory_format=torch.channels_last)
        b_xpu = a_xpu.clone(memory_format=torch.channels_last)
        self.assertEqual(b_cpu, b_xpu.to(cpu_device))

    # Loops kernel
    def test_loops(self, dtype=torch.float):
        test_shapes = [
            [[23, 72, 72], [5184, 72, 1], [23, 72, 72], [5184, 72, 1]],
            [[23, 16, 16], [23, 1, 16]],
            [[23, 16, 17], [23, 1, 17]],
            [[1, 72, 72], [23, 72, 72]],
            [[23, 72, 1], [23, 72, 72]],
            [[23000, 72, 72], [5184, 72, 1], [23000, 72, 72], [5184, 72, 1]],
            [[16, 16, 256, 256], [16, 16, 256, 256]],
            [[16, 16, 512, 512], [16, 1, 1, 512]],
            [[4, 15000, 3], [105000, 1, 15000], [4, 1, 3], [3, 3, 1]],
            [[16, 16, 512, 513], [16, 1, 1, 513]],
            [[28, 4096, 9], [36864, 9, 1], [28, 4096, 1], [4096, 1, 1]],
        ]
        for shape in test_shapes:
            if len(shape) == 2:
                a = torch.randn(shape[0], dtype=dtype)
                b = torch.randn(shape[1], dtype=dtype)
            elif len(shape) == 4:
                a = torch.as_strided(
                    torch.randn(shape[0][0] * shape[1][0]), shape[0], shape[1]
                )
                b = torch.as_strided(
                    torch.randn(shape[2][0] * shape[3][0]), shape[2], shape[3]
                )
            a_xpu = a.xpu()
            b_xpu = b.xpu()
            c = a + b
            c_xpu = a_xpu + b_xpu
            self.assertEqual(c, c_xpu.cpu())

if __name__ == "__main__":
    run_tests()
