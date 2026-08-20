# Owner(s): ["module: inductor"]

import torch
from torch._inductor.test_case import run_tests, TestCase


class TestReinterpretTensor(TestCase):
    def test_reinterpret_tensor_in_bounds(self):
        base = torch.arange(8, dtype=torch.int64)
        view = torch.ops.inductor._reinterpret_tensor(base, [8], [1], 0)
        self.assertEqual(view, base)

        second = torch.ops.inductor._reinterpret_tensor(base, [1], [1], 1)
        self.assertEqual(second.item(), 1)

    def test_reinterpret_tensor_rejects_oob_offset(self):
        base = torch.arange(8, dtype=torch.int64)
        # storage_offset 0 + increment 8 starts past the last valid element.
        with self.assertRaisesRegex(RuntimeError, "out of bounds"):
            torch.ops.inductor._reinterpret_tensor(base, [1], [1], 8)

    def test_reinterpret_tensor_rejects_negative_storage_offset(self):
        base = torch.arange(8, dtype=torch.int64)
        with self.assertRaisesRegex(RuntimeError, "invalid storage offset"):
            torch.ops.inductor._reinterpret_tensor(base, [1], [1], -1)


if __name__ == "__main__":
    run_tests()
