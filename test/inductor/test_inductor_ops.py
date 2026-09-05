# Owner(s): ["module: inductor"]

import torch
from torch._inductor.test_case import run_tests, TestCase


def _reinterpret_debug_bounds_enabled() -> bool:
    """True when _reinterpret_tensor was built without NDEBUG."""
    base = torch.arange(1, dtype=torch.int64)
    try:
        torch.ops.inductor._reinterpret_tensor(base, [1], [1], 1)
    except RuntimeError as e:
        return "_reinterpret_tensor" in str(e)
    return False


class TestInductorOps(TestCase):
    def test_reinterpret_tensor_in_bounds(self):
        base = torch.arange(8, dtype=torch.int64)
        view = torch.ops.inductor._reinterpret_tensor(base, [8], [1], 0)
        self.assertEqual(view, base)

        second = torch.ops.inductor._reinterpret_tensor(base, [1], [1], 1)
        self.assertEqual(second.item(), 1)

    def test_reinterpret_tensor_rejects_oob_offset_debug_only(self):
        if not _reinterpret_debug_bounds_enabled():
            self.skipTest("bounds checks are debug-build only")
        base = torch.arange(8, dtype=torch.int64)
        # storage_offset 0 + increment 8 starts past the last valid element.
        with self.assertRaisesRegex(RuntimeError, "_reinterpret_tensor.*out of bounds"):
            torch.ops.inductor._reinterpret_tensor(base, [1], [1], 8)

    def test_reinterpret_tensor_allows_empty_storage_before_swap(self):
        # CUDAGraph / _swap_data_ptr_: storage is resized to 0, then data is
        # swapped in. Views may be reinterpreted while empty.
        x = torch.randn(4, 4)
        y = torch.randn(4, 4)
        y_data = y.clone()
        x_storage = x.untyped_storage()
        y_storage = y.untyped_storage()
        x_storage.resize_(0)
        view = torch.ops.inductor._reinterpret_tensor(x, [2, 4], [4, 1], 4)
        x_storage._swap_data_ptr_(y_storage)
        self.assertEqual(view, y_data[1:3])


if __name__ == "__main__":
    run_tests()
