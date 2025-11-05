# Owner(s): ["module: dynamo"]

import unittest

import torch
from torch._dynamo.test_case import run_tests, TestCase
from torch._dynamo.testing import CompileCounter


class DataPtrTests(TestCase):
    """Test data_ptr() comparison operations with torch.compile"""

    # ========== Basic Equality Tests ==========

    def test_data_ptr_same_tensor_eq(self):
        """Test x.data_ptr() == x.data_ptr() returns True"""
        def fn(x):
            return x.data_ptr() == x.data_ptr()

        x = torch.randn(4, 4)
        opt_fn = torch.compile(fn, fullgraph=True)
        self.assertTrue(opt_fn(x))
        self.assertEqual(fn(x), opt_fn(x))

    def test_data_ptr_same_tensor_ne(self):
        """Test x.data_ptr() != x.data_ptr() returns False"""
        def fn(x):
            return x.data_ptr() != x.data_ptr()

        x = torch.randn(4, 4)
        opt_fn = torch.compile(fn, fullgraph=True)
        self.assertFalse(opt_fn(x))
        self.assertEqual(fn(x), opt_fn(x))

    # ========== Aliasing Tests (views share storage) ==========

    def test_data_ptr_detach_eq(self):
        """Test detach() shares data_ptr (equality)"""
        def fn(x):
            return x.data_ptr() == x.detach().data_ptr()

        x = torch.randn(4, 4)
        opt_fn = torch.compile(fn, fullgraph=True)
        self.assertTrue(opt_fn(x))
        self.assertEqual(fn(x), opt_fn(x))

    def test_data_ptr_detach_eq_swapped(self):
        """Test detach() shares data_ptr (swapped operands)"""
        def fn(x):
            return x.detach().data_ptr() == x.data_ptr()

        x = torch.randn(4, 4)
        opt_fn = torch.compile(fn, fullgraph=True)
        self.assertTrue(opt_fn(x))
        self.assertEqual(fn(x), opt_fn(x))

    def test_data_ptr_detach_ne(self):
        """Test detach() shares data_ptr (inequality returns False)"""
        def fn(x):
            return x.data_ptr() != x.detach().data_ptr()

        x = torch.randn(4, 4)
        opt_fn = torch.compile(fn, fullgraph=True)
        self.assertFalse(opt_fn(x))
        self.assertEqual(fn(x), opt_fn(x))

    def test_data_ptr_view_eq(self):
        """Test view() shares data_ptr"""
        def fn(x):
            return x.data_ptr() == x.view_as(x).data_ptr()

        x = torch.randn(4, 4)
        opt_fn = torch.compile(fn, fullgraph=True)
        self.assertTrue(opt_fn(x))
        self.assertEqual(fn(x), opt_fn(x))

    def test_data_ptr_view_ne(self):
        """Test view() shares data_ptr (inequality returns False)"""
        def fn(x):
            return x.data_ptr() != x.view_as(x).data_ptr()

        x = torch.randn(4, 4)
        opt_fn = torch.compile(fn, fullgraph=True)
        self.assertFalse(opt_fn(x))
        self.assertEqual(fn(x), opt_fn(x))

    def test_data_ptr_reshape_eq(self):
        """Test reshape() shares data_ptr when contiguous"""
        def fn(x):
            return x.data_ptr() == x.reshape(-1).data_ptr()

        x = torch.randn(4, 4)
        opt_fn = torch.compile(fn, fullgraph=True)
        self.assertTrue(opt_fn(x))
        self.assertEqual(fn(x), opt_fn(x))

    def test_data_ptr_transpose_eq(self):
        """Test transpose() shares data_ptr (creates view)"""
        def fn(x):
            return x.data_ptr() == x.t().data_ptr()

        x = torch.randn(4, 4)
        opt_fn = torch.compile(fn, fullgraph=True)
        self.assertTrue(opt_fn(x))
        self.assertEqual(fn(x), opt_fn(x))

    # ========== Clone Tests (different storage) ==========

    def test_data_ptr_clone_eq(self):
        """Test clone() has different data_ptr (equality returns False)"""
        def fn(x):
            return x.data_ptr() == x.clone().data_ptr()

        x = torch.randn(4, 4)
        opt_fn = torch.compile(fn, fullgraph=True)
        self.assertFalse(opt_fn(x))
        self.assertEqual(fn(x), opt_fn(x))

    def test_data_ptr_clone_ne(self):
        """Test clone() has different data_ptr (inequality returns True)"""
        def fn(x):
            return x.data_ptr() != x.clone().data_ptr()

        x = torch.randn(4, 4)
        opt_fn = torch.compile(fn, fullgraph=True)
        self.assertTrue(opt_fn(x))
        self.assertEqual(fn(x), opt_fn(x))

    def test_data_ptr_different_tensors_eq(self):
        """Test different tensors have different data_ptr (equality)"""
        def fn(x, y):
            return x.data_ptr() == y.data_ptr()

        x = torch.randn(4, 4)
        y = torch.randn(4, 4)
        opt_fn = torch.compile(fn, fullgraph=True)
        self.assertFalse(opt_fn(x, y))
        self.assertEqual(fn(x, y), opt_fn(x, y))

    def test_data_ptr_different_tensors_ne(self):
        """Test different tensors have different data_ptr (inequality)"""
        def fn(x, y):
            return x.data_ptr() != y.data_ptr()

        x = torch.randn(4, 4)
        y = torch.randn(4, 4)
        opt_fn = torch.compile(fn, fullgraph=True)
        self.assertTrue(opt_fn(x, y))
        self.assertEqual(fn(x, y), opt_fn(x, y))

    # ========== Int Parameter Comparisons ==========

    def test_data_ptr_eq_int_parameter(self):
        """Test data_ptr() == with int parameter (matching)"""
        def fn(x, ptr_value):
            return x.data_ptr() == ptr_value

        x = torch.randn(4, 4)
        actual_ptr = x.data_ptr()

        opt_fn = torch.compile(fn, fullgraph=True)
        self.assertTrue(opt_fn(x, actual_ptr))
        self.assertEqual(fn(x, actual_ptr), opt_fn(x, actual_ptr))

    def test_data_ptr_eq_int_parameter_swapped(self):
        """Test data_ptr() == with int parameter (swapped operands)"""
        def fn(x, ptr_value):
            return ptr_value == x.data_ptr()

        x = torch.randn(4, 4)
        actual_ptr = x.data_ptr()

        opt_fn = torch.compile(fn, fullgraph=True)
        self.assertTrue(opt_fn(x, actual_ptr))
        self.assertEqual(fn(x, actual_ptr), opt_fn(x, actual_ptr))

    def test_data_ptr_ne_int_parameter_matching(self):
        """Test data_ptr() != with int parameter (matching pointer)"""
        def fn(x, ptr_value):
            return x.data_ptr() != ptr_value

        x = torch.randn(4, 4)
        actual_ptr = x.data_ptr()

        opt_fn = torch.compile(fn, fullgraph=True)
        self.assertFalse(opt_fn(x, actual_ptr))
        self.assertEqual(fn(x, actual_ptr), opt_fn(x, actual_ptr))

    def test_data_ptr_ne_int_parameter_different(self):
        """Test data_ptr() != with int parameter (different value)"""
        def fn(x, ptr_value):
            return x.data_ptr() != ptr_value

        x = torch.randn(4, 4)
        wrong_ptr = x.data_ptr() + 1000

        opt_fn = torch.compile(fn, fullgraph=True)
        self.assertTrue(opt_fn(x, wrong_ptr))
        self.assertEqual(fn(x, wrong_ptr), opt_fn(x, wrong_ptr))

    def test_data_ptr_ne_int_parameter_swapped(self):
        """Test data_ptr() != with int parameter (swapped operands)"""
        def fn(x, ptr_value):
            return ptr_value != x.data_ptr()

        x = torch.randn(4, 4)
        wrong_ptr = x.data_ptr() + 1000

        opt_fn = torch.compile(fn, fullgraph=True)
        self.assertTrue(opt_fn(x, wrong_ptr))
        self.assertEqual(fn(x, wrong_ptr), opt_fn(x, wrong_ptr))

    def test_data_ptr_eq_int_constant(self):
        """Test data_ptr() == with int constant"""
        def fn(x):
            return x.data_ptr() == 99999

        x = torch.randn(10)
        opt_fn = torch.compile(fn, fullgraph=True)
        self.assertFalse(opt_fn(x))
        self.assertEqual(fn(x), opt_fn(x))

    def test_data_ptr_method_as_argument(self):
        """Test passing torch.Tensor.data_ptr method as function argument"""
        def fn(x, data_ptr_method):
            return x.data_ptr() == data_ptr_method(x)

        x = torch.randn(4, 4)
        ptr_method = torch.Tensor.data_ptr

        opt_fn = torch.compile(fn, fullgraph=True)
        self.assertTrue(opt_fn(x, ptr_method))
        self.assertEqual(fn(x, ptr_method), opt_fn(x, ptr_method))

    def test_data_ptr_method_as_argument_ne(self):
        """Test passing torch.Tensor.data_ptr method as function argument with !="""
        def fn(x, y, data_ptr_method):
            return data_ptr_method(x) != data_ptr_method(y)

        x = torch.randn(4, 4)
        y = torch.randn(4, 4)
        ptr_method = torch.Tensor.data_ptr

        opt_fn = torch.compile(fn, fullgraph=True)
        self.assertTrue(opt_fn(x, y, ptr_method))
        self.assertEqual(fn(x, y, ptr_method), opt_fn(x, y, ptr_method))

    # ========== Slice/Offset Tests ==========

    def test_data_ptr_full_slice_eq(self):
        """Test full slice x[:] shares data_ptr with x"""
        def fn(x):
            return x.data_ptr() == x[:].data_ptr()

        x = torch.randn(10)
        opt_fn = torch.compile(fn, fullgraph=True)
        self.assertTrue(opt_fn(x))
        self.assertEqual(fn(x), opt_fn(x))

    def test_data_ptr_partial_slice_ne(self):
        """Test partial slice x[5:] has different data_ptr (different first element)"""
        def fn(x):
            return x.data_ptr() != x[5:].data_ptr()

        x = torch.randn(10)
        opt_fn = torch.compile(fn, fullgraph=True)
        self.assertTrue(opt_fn(x))
        self.assertEqual(fn(x), opt_fn(x))

    def test_data_ptr_matrix_columns_ne(self):
        """Test different columns have different data_ptr (different first element)"""
        def fn(x):
            return x[:, 0].data_ptr() != x[:, 1].data_ptr()

        x = torch.randn(5, 5)
        opt_fn = torch.compile(fn, fullgraph=True)
        self.assertTrue(opt_fn(x))
        self.assertEqual(fn(x), opt_fn(x))

    # ========== Empty Tensor Tests ==========

    def test_data_ptr_empty_tensors_eq(self):
        """Test two empty tensors both have data_ptr() == 0"""
        def fn(x, y):
            return x.data_ptr() == y.data_ptr()

        empty1 = torch.tensor((), dtype=torch.float32)
        empty2 = torch.tensor((), dtype=torch.float32)
        opt_fn = torch.compile(fn, fullgraph=True)
        self.assertTrue(opt_fn(empty1, empty2))
        self.assertEqual(fn(empty1, empty2), opt_fn(empty1, empty2))

    def test_data_ptr_empty_vs_nonempty_ne(self):
        """Test empty vs non-empty tensor have different data_ptr"""
        def fn(x, y):
            return x.data_ptr() != y.data_ptr()

        empty = torch.tensor((), dtype=torch.float32)
        nonempty = torch.randn(10)
        opt_fn = torch.compile(fn, fullgraph=True)
        self.assertTrue(opt_fn(empty, nonempty))
        self.assertEqual(fn(empty, nonempty), opt_fn(empty, nonempty))

    # ========== Mixed Operations ==========

    def test_data_ptr_mixed_eq_ne_operators(self):
        """Test mixing == and != operators in same function"""
        def fn(x):
            detached = x.detach()
            cloned = x.clone()
            viewed = x.view_as(x)

            # These should be True
            same_eq_detached = x.data_ptr() == detached.data_ptr()
            same_eq_view = x.data_ptr() == viewed.data_ptr()
            diff_ne_clone = x.data_ptr() != cloned.data_ptr()

            # These should be False
            same_ne_detached = x.data_ptr() != detached.data_ptr()
            same_ne_view = x.data_ptr() != viewed.data_ptr()
            diff_eq_clone = x.data_ptr() == cloned.data_ptr()

            return (
                same_eq_detached
                and same_eq_view
                and diff_ne_clone
                and not same_ne_detached
                and not same_ne_view
                and not diff_eq_clone
            )

        x = torch.randn(4, 4)
        opt_fn = torch.compile(fn, fullgraph=True)
        self.assertTrue(opt_fn(x))
        self.assertEqual(fn(x), opt_fn(x))

    def test_data_ptr_chained_comparisons(self):
        """Test storing data_ptr() results and chaining comparisons"""
        def fn(x):
            a = x.data_ptr()
            b = x.detach().data_ptr()
            c = x.clone().data_ptr()

            return (a == b) and (a != c) and (b != c)

        x = torch.randn(4, 4)
        opt_fn = torch.compile(fn, fullgraph=True)
        self.assertTrue(opt_fn(x))
        self.assertEqual(fn(x), opt_fn(x))

    def test_data_ptr_with_other_checks(self):
        """Test data_ptr() combined with stride and shape checks"""
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

    # ========== Graph Break Tests ==========

    def test_data_ptr_no_graph_break(self):
        """Ensure data_ptr() comparisons don't cause graph breaks"""
        def fn(x):
            return x.data_ptr() == x.detach().data_ptr()

        x = torch.randn(4, 4)
        counter = CompileCounter()
        opt_fn = torch._dynamo.optimize(counter, nopython=True)(fn)
        result = opt_fn(x)
        self.assertTrue(result)
        self.assertEqual(counter.frame_count, 1)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_data_ptr_cuda(self):
        """Test data_ptr() comparisons work on CUDA tensors"""
        def fn(x):
            return (
                x.data_ptr() == x.detach().data_ptr()
                and x.data_ptr() != x.clone().data_ptr()
            )

        x = torch.randn(4, 4, device="cuda")
        opt_fn = torch.compile(fn, fullgraph=True)
        self.assertTrue(opt_fn(x))
        self.assertEqual(fn(x), opt_fn(x))


if __name__ == "__main__":
    run_tests()
