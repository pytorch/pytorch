# Owner(s): ["module: intel"]

import sys
import unittest

import torch
import numpy as np
import math
from math import inf, nan, isnan

from torch.testing._internal.common_utils import NoTest, run_tests, TEST_XPU, TestCase
from torch.testing._internal.common_device_type import instantiate_device_type_tests, dtypes, precisionOverride
from torch.testing._internal.common_dtype import floating_and_complex_types_and

if not TEST_XPU:
    print("XPU not available, skipping tests", file=sys.stderr)
    TestCase = NoTest  # noqa: F811

TEST_MULTIXPU = torch.xpu.device_count() > 1

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

class TestBasicGEMM(TestCase):
    @precisionOverride({torch.double: 1e-8, torch.float: 1e-4,
                        torch.half: 1e-1, torch.cfloat: 1e-4, torch.cdouble: 1e-8})
    @dtypes(*floating_and_complex_types_and(torch.half))
    def test_addmm(self, device, dtype):
        m1 = torch.randn((10, 50), dtype=dtype, device=device)
        m2 = torch.randn((50, 25), dtype=dtype, device=device)
        M = torch.randn((10, 25), dtype=dtype, device=device)

        res1 = torch.addmm(M, m1, m2)
        res2 = torch.full_like(res1, math.nan)

        torch.addmm(M, m1, m2, out=res2)

        res3 = m1.cpu().numpy() @ m2.cpu().numpy()
        res3 += M.cpu().numpy()
        res3 = torch.from_numpy(res3).to(dtype)

        self.assertEqual(res1.cpu(), res2.cpu())
        self.assertEqual(res1.cpu(), res3.cpu())

    @dtypes(torch.half, torch.float, torch.double, torch.cfloat, torch.cdouble)
    def test_addmv(self, dtype):
        m1 = torch.randn([2, 3], dtype=dtype)
        m2 = torch.randn([3], dtype=dtype)
        x = torch.randn([2], dtype=dtype)

        res1 = torch.addmv(x, m1, m2)

        res2 = torch.full_like(res1, math.nan)

        torch.addmv(x, m1, m2, out=res2)

        res3 = m1.cpu().numpy() @ m2.cpu().numpy()
        res3 += x.cpu().numpy()
        res3 = torch.from_numpy(res3).to(dtype)

        self.assertEqual(res1.cpu(), res2.cpu())
        self.assertEqual(res1.cpu(), res3.cpu())

    @dtypes(torch.half, torch.float32, torch.float64, torch.int32, torch.int64, torch.cfloat, torch.cdouble)
    def test_mm(self, dtype):
        pass

    @dtypes(*floating_and_complex_types_and(torch.half))
    def test_bmm(self, device, dtype):
        b1 = torch.randn((3, 3, 4), device=device, dtype=dtype)
        b2 = torch.randn((3, 4, 5), device=device, dtype=dtype)

        res1 = torch.bmm(b1, b2)
        # res2 = torch.full((3, 3, 5), math.nan, dtype=dtype, devcie=device)
        res2 = torch.full((3, 3, 5), math.nan, dtype=dtype, device=device)
        torch.bmm(b1, b2, out=res2)
        expect = torch.from_numpy(b1.to(dtype).cpu().numpy() @ b2.to(dtype).cpu().numpy()).to(device)
        self.assertEqual(expect, res1.cpu())
        self.assertEqual(expect, res2.cpu())


    @dtypes(*floating_and_complex_types_and(torch.half))
    def test_addbmm(self, device, dtype):
        M = torch.zeros([3, 2], dtype=dtype, device=device)
        b1 = torch.randn([10, 3, 4], dtype=dtype, device=device)
        b2 = torch.randn([10, 4, 2], dtype=dtype, device=device)

        res1 = torch.addbmm(M, b1, b2)
        res2 = torch.full([3, 2], math.nan, dtype=dtype, device=device)
        torch.addbmm(M, b1, b2, out=res2)
        expect = torch.from_numpy(b1.to(dtype).cpu().numpy() @ b2.to(dtype).cpu().numpy()).to(device).sum(0)

        self.assertEqual(res1, res2)
        self.assertEqual(res1, expect)

    @dtypes(*floating_and_complex_types_and(torch.half))
    def test_baddbmm(self, device, dtype):
        M, N, O = 12, 8, 50
        num_batches = 10
        b1 = torch.randn((num_batches, M, N), dtype=dtype, device=device)
        b2 = torch.randn((num_batches, N, O), dtype=dtype, device=device)
        x = torch.zeros((num_batches, M, O), dtype=dtype, device=device)

        res1 = torch.baddbmm(x, b1, b2)
        res2 = torch.full((num_batches, M, O), math.nan, dtype=dtype, device=device)
        torch.baddbmm(x, b1, b2, out=res2)
        expect = torch.from_numpy(b1.to(dtype).cpu().numpy() @ b2.to(dtype).cpu().numpy()).to(device)

        self.assertEqual(res1, res2)
        self.assertEqual(res1, expect)

    def test_tensordot(self, device):
        a = torch.randn((3, 4, 5), device=device)
        b = torch.randn((4, 3, 2), device=device)
        c = torch.tensordot(a, b, dims=([1, 0], [0, 1])).cpu()
        cn = torch.from_numpy(np.tensordot(a.cpu().numpy(), b.cpu().numpy(), axes=([1, 0], [0, 1])))
        self.assertEqual(c, cn)

        cout = torch.zeros((5, 2), device=device)
        torch.tensordot(a, b, dims=([1, 0], [0, 1]), out=cout).cpu()
        self.assertEqual(c, cout)

        a = torch.randn(2, 3, 4, 5, device=device)
        b = torch.randn(4, 5, 6, 7, device=device)
        c = torch.tensordot(a, b, dims=2).cpu()
        cn = torch.from_numpy(np.tensordot(a.cpu().numpy(), b.cpu().numpy(), axes=2))

        self.assertEqual(c, cn)

        c = torch.tensordot(a, b).cpu()
        cn = torch.from_numpy(np.tensordot(a.cpu().numpy(), b.cpu().numpy()))
        self.assertEqual(c, cn)

        a = torch.tensordot(torch.tensor(0.), torch.tensor(0.), 0)
        an = torch.from_numpy(np.tensordot(np.zeros((), dtype=np.float32), np.zeros((), dtype=np.float32), 0))
        self.assertEqual(a, an)

instantiate_device_type_tests(TestBasicGEMM, globals(), only_for="cpu, xpu")


if __name__ == "__main__":
    run_tests()
