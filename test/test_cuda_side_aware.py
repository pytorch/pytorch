# Owner(s): ["module: cuda"]
"""Locality-interleaved allocator and side-aware elementwise schedule.

Checks behavior only (bitwise results and which path ran), never timing.
"""

import unittest

import torch
from torch.testing._internal.common_utils import run_tests, TestCase


def _supported():
    if not torch.cuda.is_available() or not hasattr(torch._C, "_cuda_localityInterleavedAllocator"):
        return False
    if torch.cuda.get_device_capability() != (10, 7):
        return False
    try:
        torch.cuda.memory.LocalityInterleavedAllocator()
        return True
    except RuntimeError:  # driver without locality domains, or not two domains
        return False


@unittest.skipUnless(_supported(), "needs an sm_107 device with two locality domains and CUDA >= 13.4")
class TestSideAware(TestCase):
    N = (64 << 20) // 2 + 12345  # >= 64 MiB of fp32, odd length so the last chunk is partial

    @classmethod
    def setUpClass(cls):
        cls.pool = torch.cuda.MemPool(torch.cuda.memory.LocalityInterleavedAllocator().allocator(), no_split=True)

    def setUp(self):
        self.prev = torch._C._cuda_get_side_aware()

    def tearDown(self):
        torch._C._cuda_set_side_aware(self.prev)

    def _launches(self, fn):
        before = torch._C._cuda_side_aware_launch_count()
        out = fn()
        torch.cuda.synchronize()
        return out, torch._C._cuda_side_aware_launch_count() - before

    def _check(self, fn, dtype, expect_side):
        gen = torch.Generator(device="cuda").manual_seed(0)
        with torch.cuda.use_mem_pool(self.pool):
            a = torch.randn(self.N, device="cuda", dtype=dtype, generator=gen)
            b = torch.randn(self.N, device="cuda", dtype=dtype, generator=gen)
            torch._C._cuda_set_side_aware(False)
            ref, n_off = self._launches(lambda: fn(a, b))
            torch._C._cuda_set_side_aware(True)
            # Per-functor calibration launches default, default, side, side first, so run 4 times and check
            # every result: each call must be bitwise identical whichever kernel ran.
            gots, n_on = self._launches(lambda: [fn(a, b) for _ in range(4)])
        self.assertEqual(n_off, 0)
        self.assertEqual(n_on > 0, expect_side)
        bits = {torch.float32: torch.int32, torch.bfloat16: torch.int16}[dtype]
        for got in gots:
            self.assertTrue(torch.equal(got.view(bits), ref.view(bits)))

    def test_pointwise_bitwise(self):
        ops = {
            "add": lambda a, b: a + b,
            "mul": lambda a, b: a * b,
            "neg": lambda a, b: -a,
            "exp": lambda a, b: a.exp(),
            "gelu": lambda a, b: torch.nn.functional.gelu(a),
            "scalar": lambda a, b: a * 2.0,
            "add_": lambda a, b: a.clone().add_(b),
        }
        for dtype in (torch.float32, torch.bfloat16):
            for name, fn in ops.items():
                with self.subTest(op=name, dtype=dtype):
                    self._check(fn, dtype, expect_side=True)

    def test_default_path_outside_arena(self):
        a = torch.randn(self.N, device="cuda")
        b = torch.randn(self.N, device="cuda")
        torch._C._cuda_set_side_aware(True)
        _, n = self._launches(lambda: a + b)
        self.assertEqual(n, 0)

    def test_default_path_small_and_misaligned(self):
        with torch.cuda.use_mem_pool(self.pool):
            small = torch.randn(1 << 20, device="cuda")
            big = torch.randn(self.N + 1, device="cuda")
        torch._C._cuda_set_side_aware(True)
        _, n_small = self._launches(lambda: small + small)
        _, n_offset = self._launches(lambda: big[1:] + big[:-1])  # 4 B offset: not vector aligned
        self.assertEqual(n_small, 0)
        self.assertEqual(n_offset, 0)

    def test_arena_alignment(self):
        with torch.cuda.use_mem_pool(self.pool):
            ts = [torch.empty(n, device="cuda") for n in (1, 12345, 3 << 20)]
        for t in ts:
            self.assertEqual(t.data_ptr() % (4 << 20), 0)


if __name__ == "__main__":
    run_tests()
