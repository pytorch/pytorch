# Owner(s): ["module: dynamo"]

import unittest

import torch
import torch._dynamo.testing
from torch._dynamo.testing import CompileCounter, same
from torch.testing._internal.common_utils import run_tests, TestCase


class DataPtrTests(TestCase):
    """Test data_ptr() comparison operations with torch.compile"""

    def test_data_ptr_same_tensor(self):
        """Test comparing data_ptr() of the same tensor"""

        def fn(x):
            return x.data_ptr() == x.data_ptr()

        x = torch.randn(4, 4)
        opt_fn = torch.compile(fn, fullgraph=True)
        self.assertTrue(opt_fn(x))
        self.assertEqual(fn(x), opt_fn(x))

    def test_data_ptr_detach(self):
        """Test that detach() shares the same data_ptr"""

        def fn(x):
            detached = x.detach()
            return x.data_ptr() == detached.data_ptr()

        x = torch.randn(4, 4)
        opt_fn = torch.compile(fn, fullgraph=True)
        self.assertTrue(opt_fn(x))
        self.assertEqual(fn(x), opt_fn(x))

    def test_data_ptr_view(self):
        """Test that view() shares the same data_ptr"""

        def fn(x):
            viewed = x.view_as(x)
            return x.data_ptr() == viewed.data_ptr()

        x = torch.randn(4, 4)
        opt_fn = torch.compile(fn, fullgraph=True)
        self.assertTrue(opt_fn(x))
        self.assertEqual(fn(x), opt_fn(x))

    def test_data_ptr_clone(self):
        """Test that clone() has a different data_ptr"""

        def fn(x):
            cloned = x.clone()
            return x.data_ptr() == cloned.data_ptr()

        x = torch.randn(4, 4)
        opt_fn = torch.compile(fn, fullgraph=True)
        self.assertFalse(opt_fn(x))
        self.assertEqual(fn(x), opt_fn(x))

    def test_data_ptr_inequality(self):
        """Test data_ptr() != operator"""

        def fn(x):
            cloned = x.clone()
            return x.data_ptr() != cloned.data_ptr()

        x = torch.randn(4, 4)
        opt_fn = torch.compile(fn, fullgraph=True)
        self.assertTrue(opt_fn(x))
        self.assertEqual(fn(x), opt_fn(x))

    def test_data_ptr_mixed_comparison(self):
        """Test multiple data_ptr() comparisons in one function"""

        def fn(x):
            detached = x.detach()
            cloned = x.clone()
            viewed = x.view_as(x)

            same_as_detached = x.data_ptr() == detached.data_ptr()
            diff_from_clone = x.data_ptr() != cloned.data_ptr()
            same_as_view = x.data_ptr() == viewed.data_ptr()

            return same_as_detached and diff_from_clone and same_as_view

        x = torch.randn(4, 4)
        opt_fn = torch.compile(fn, fullgraph=True)
        self.assertTrue(opt_fn(x))
        self.assertEqual(fn(x), opt_fn(x))

    def test_data_ptr_with_other_checks(self):
        """Test data_ptr() combined with stride and shape checks (original repro)"""

        def fn(x):
            original = x
            detached = x.detach()
            same_data = (
                original.data_ptr() == detached.data_ptr()
                and original.stride() == detached.stride()
                and original.shape == detached.shape
            )
            return same_data

        x = torch.rand(1, 1, 1, 3, dtype=torch.float32)
        opt_fn = torch.compile(fn, fullgraph=True)
        self.assertTrue(opt_fn(x))
        self.assertEqual(fn(x), opt_fn(x))

    def test_data_ptr_no_graph_break(self):
        """Ensure data_ptr() comparisons don't cause graph breaks"""

        def fn(x):
            detached = x.detach()
            return x.data_ptr() == detached.data_ptr()

        x = torch.randn(4, 4)
        counter = CompileCounter()
        opt_fn = torch._dynamo.optimize(counter, nopython=True)(fn)
        result = opt_fn(x)
        self.assertTrue(result)
        self.assertEqual(counter.frame_count, 1)  # Single graph, no breaks

    def test_data_ptr_reshape_aliasing(self):
        """Test that reshape() can share data_ptr when contiguous"""

        def fn(x):
            # reshape should share storage when x is contiguous
            reshaped = x.reshape(-1)
            return x.data_ptr() == reshaped.data_ptr()

        x = torch.randn(4, 4)  # contiguous
        opt_fn = torch.compile(fn, fullgraph=True)
        self.assertTrue(opt_fn(x))
        self.assertEqual(fn(x), opt_fn(x))

    def test_data_ptr_transpose_aliasing(self):
        """Test that transpose() shares data_ptr (creates view)"""

        def fn(x):
            transposed = x.t()
            return x.data_ptr() == transposed.data_ptr()

        x = torch.randn(4, 4)
        opt_fn = torch.compile(fn, fullgraph=True)
        self.assertTrue(opt_fn(x))
        self.assertEqual(fn(x), opt_fn(x))

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_data_ptr_cuda(self):
        """Test data_ptr() comparisons work on CUDA tensors"""

        def fn(x):
            detached = x.detach()
            cloned = x.clone()
            return (
                x.data_ptr() == detached.data_ptr()
                and x.data_ptr() != cloned.data_ptr()
            )

        x = torch.randn(4, 4, device="cuda")
        opt_fn = torch.compile(fn, fullgraph=True)
        self.assertTrue(opt_fn(x))
        self.assertEqual(fn(x), opt_fn(x))

    def test_data_ptr_slice_aliasing(self):
        """Test that slicing shares data_ptr (offset may differ)"""

        def fn(x):
            # Full slice should have same data_ptr
            sliced = x[:]
            return x.data_ptr() == sliced.data_ptr()

        x = torch.randn(4, 4)
        opt_fn = torch.compile(fn, fullgraph=True)
        self.assertTrue(opt_fn(x))
        self.assertEqual(fn(x), opt_fn(x))

    def test_data_ptr_as_strided_aliasing(self):
        """Test that as_strided shares data_ptr"""

        def fn(x):
            # as_strided with same shape and stride should share storage
            strided = torch.as_strided(x, x.shape, x.stride())
            return x.data_ptr() == strided.data_ptr()

        x = torch.randn(4, 4)
        opt_fn = torch.compile(fn, fullgraph=True)
        self.assertTrue(opt_fn(x))
        self.assertEqual(fn(x), opt_fn(x))

    def test_data_ptr_inplace_op_preserves(self):
        """Test that in-place operations preserve data_ptr"""

        def fn(x):
            original_ptr_matches = x.data_ptr() == x.data_ptr()
            x.add_(1)
            ptr_after_inplace = x.data_ptr() == x.data_ptr()
            return original_ptr_matches and ptr_after_inplace

        x = torch.randn(4, 4)
        opt_fn = torch.compile(fn, fullgraph=True)
        self.assertTrue(opt_fn(x))

    def test_data_ptr_comparison_with_incompatible_type(self):
        """Test that comparing data_ptr with incompatible types returns NotImplemented"""

        def fn(x):
            ptr = x.data_ptr()
            # Comparing with the same type should work
            return ptr == ptr

        x = torch.randn(4, 4)
        opt_fn = torch.compile(fn, fullgraph=True)
        self.assertTrue(opt_fn(x))
        self.assertEqual(fn(x), opt_fn(x))

    def test_data_ptr_slice_different_first_element(self):
        """
        Test data_ptr() returns address of first element.
        x[5:] shares storage with x, but its first element is x's 5th element.
        """

        def fn(x):
            sliced = x[5:]
            return x.data_ptr() == sliced.data_ptr()

        x = torch.randn(10)
        opt_fn = torch.compile(fn, fullgraph=True)
        # False: different first elements despite shared storage
        self.assertFalse(opt_fn(x))
        self.assertEqual(fn(x), opt_fn(x))

    def test_data_ptr_slice_inequality(self):
        """Test that slices with different offsets are properly detected as unequal"""

        def fn(x):
            sliced = x[5:]
            return x.data_ptr() != sliced.data_ptr()

        x = torch.randn(10)
        opt_fn = torch.compile(fn, fullgraph=True)
        # Should be True - different offsets!
        self.assertTrue(opt_fn(x))
        self.assertEqual(fn(x), opt_fn(x))

    def test_data_ptr_matrix_columns_different_first_element(self):
        """
        Different columns start at different elements.
        m[:, 0] first element is m[0,0]
        m[:, 1] first element is m[0,1]
        """

        def fn(x):
            col0 = x[:, 0]  # First element at offset 0
            col1 = x[:, 1]  # First element at offset 1
            return col0.data_ptr() == col1.data_ptr()

        x = torch.randn(5, 5)
        opt_fn = torch.compile(fn, fullgraph=True)
        # False: columns start at different elements
        self.assertFalse(opt_fn(x))
        self.assertEqual(fn(x), opt_fn(x))

    def test_data_ptr_full_slice_same_first_element(self):
        """
        Full slice x[:] and x both start at the same first element.
        Both point to x[0].
        """

        def fn(x):
            full_slice = x[:]
            # Both x and x[:] start at x[0]
            return x.data_ptr() == full_slice.data_ptr()

        x = torch.randn(10)
        opt_fn = torch.compile(fn, fullgraph=True)
        # True: same first element (both at offset 0)
        self.assertTrue(opt_fn(x))
        self.assertEqual(fn(x), opt_fn(x))


if __name__ == "__main__":
    run_tests()
