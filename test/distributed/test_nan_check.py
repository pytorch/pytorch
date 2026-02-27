# Owner(s): ["oncall: distributed"]

import sys
import unittest

import torch
import torch.distributed as dist


if not dist.is_available():
    print("distributed package not available, skipping tests", file=sys.stderr)
    sys.exit(0)

from torch.testing._internal.common_utils import run_tests, TestCase


HAS_ACCELERATOR = torch.accelerator.is_available()
ACCELERATOR = (
    torch.accelerator.current_accelerator() if HAS_ACCELERATOR else torch.device("cpu")
)


class TestNanCheck(TestCase):
    def _check_for_nan(self, tensor):
        torch.ops.c10d.check_for_nan(tensor)

    def test_no_nan_float32(self):
        self._check_for_nan(torch.randn(100))

    def test_no_nan_float64(self):
        self._check_for_nan(torch.randn(100, dtype=torch.float64))

    def test_no_nan_float16(self):
        self._check_for_nan(torch.randn(100, dtype=torch.float16))

    def test_no_nan_bfloat16(self):
        self._check_for_nan(torch.randn(100, dtype=torch.bfloat16))

    @unittest.skipUnless(HAS_ACCELERATOR, "no accelerator available")
    def test_no_nan_float32_accelerator(self):
        self._check_for_nan(torch.randn(100, device=ACCELERATOR))
        torch.accelerator.synchronize()

    @unittest.skipUnless(HAS_ACCELERATOR, "no accelerator available")
    def test_no_nan_float64_accelerator(self):
        self._check_for_nan(torch.randn(100, dtype=torch.float64, device=ACCELERATOR))
        torch.accelerator.synchronize()

    @unittest.skipUnless(HAS_ACCELERATOR, "no accelerator available")
    def test_no_nan_float16_accelerator(self):
        self._check_for_nan(torch.randn(100, dtype=torch.float16, device=ACCELERATOR))
        torch.accelerator.synchronize()

    @unittest.skipUnless(HAS_ACCELERATOR, "no accelerator available")
    def test_no_nan_bfloat16_accelerator(self):
        self._check_for_nan(torch.randn(100, dtype=torch.bfloat16, device=ACCELERATOR))
        torch.accelerator.synchronize()

    @unittest.skipUnless(HAS_ACCELERATOR, "no accelerator available")
    def test_skips_integer_tensors_accelerator(self):
        self._check_for_nan(
            torch.tensor([1, 2, 3], dtype=torch.int32, device=ACCELERATOR)
        )
        torch.accelerator.synchronize()

    @unittest.skipUnless(HAS_ACCELERATOR, "no accelerator available")
    def test_empty_tensor_accelerator(self):
        self._check_for_nan(torch.tensor([], dtype=torch.float32, device=ACCELERATOR))
        torch.accelerator.synchronize()

    @unittest.skipUnless(HAS_ACCELERATOR, "no accelerator available")
    def test_large_tensor_no_nan_accelerator(self):
        self._check_for_nan(torch.randn(100000, device=ACCELERATOR))
        torch.accelerator.synchronize()

    def test_nan_float32(self):
        tensor = torch.tensor([1.0, 2.0, float("nan"), 4.0])
        with self.assertRaisesRegex(RuntimeError, "NaN"):
            self._check_for_nan(tensor)

    def test_nan_float64(self):
        tensor = torch.tensor([1.0, float("nan")], dtype=torch.float64)
        with self.assertRaisesRegex(RuntimeError, "NaN"):
            self._check_for_nan(tensor)

    def test_nan_float16(self):
        tensor = torch.tensor([1.0, float("nan")], dtype=torch.float16)
        with self.assertRaisesRegex(RuntimeError, "NaN"):
            self._check_for_nan(tensor)

    def test_nan_bfloat16(self):
        tensor = torch.tensor([1.0, float("nan")], dtype=torch.bfloat16)
        with self.assertRaisesRegex(RuntimeError, "NaN"):
            self._check_for_nan(tensor)

    def test_nan_at_end(self):
        tensor = torch.ones(1000)
        tensor[-1] = float("nan")
        with self.assertRaisesRegex(RuntimeError, "NaN"):
            self._check_for_nan(tensor)

    def test_skips_integer_tensors(self):
        self._check_for_nan(torch.tensor([1, 2, 3], dtype=torch.int32))

    def test_empty_tensor(self):
        self._check_for_nan(torch.tensor([], dtype=torch.float32))

    def test_all_nan(self):
        with self.assertRaisesRegex(RuntimeError, "NaN"):
            self._check_for_nan(torch.full((10,), float("nan")))

    def test_large_tensor_no_nan(self):
        self._check_for_nan(torch.randn(100000))

    def test_large_tensor_with_nan(self):
        tensor = torch.randn(100000)
        tensor[99999] = float("nan")
        with self.assertRaisesRegex(RuntimeError, "NaN"):
            self._check_for_nan(tensor)


if __name__ == "__main__":
    run_tests()
