# Owner(s): ["module: inductor"]

import torch
from torch._inductor.test_case import run_tests, TestCase


class TestAllocFromPool(TestCase):
    def test_alloc_from_pool_in_bounds(self):
        base = torch.arange(8, dtype=torch.int64)
        view = torch.ops.inductor._alloc_from_pool(base, 0, torch.int64, [8], [1])
        self.assertEqual(view, base)

        second = torch.ops.inductor._alloc_from_pool(base, 8, torch.int64, [1], [1])
        self.assertEqual(second.item(), 1)

    def test_alloc_from_pool_rejects_oob_offset(self):
        base = torch.arange(8, dtype=torch.int64)
        # 8 * sizeof(int64) = 64-byte storage; offset 64 starts past the end.
        with self.assertRaisesRegex(RuntimeError, "out of bounds"):
            torch.ops.inductor._alloc_from_pool(base, 64, torch.int64, [1], [1])

    def test_alloc_from_pool_rejects_unaligned_offset(self):
        base = torch.arange(8, dtype=torch.int64)
        with self.assertRaisesRegex(RuntimeError, "multiple of dtype itemsize"):
            torch.ops.inductor._alloc_from_pool(base, 1, torch.int64, [1], [1])

    def test_alloc_from_pool_rejects_negative_offset(self):
        base = torch.arange(8, dtype=torch.int64)
        with self.assertRaisesRegex(RuntimeError, "non-negative"):
            torch.ops.inductor._alloc_from_pool(base, -8, torch.int64, [1], [1])


if __name__ == "__main__":
    run_tests()
