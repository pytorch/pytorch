# Owner(s): ["module: dynamo"]

import torch
import torch._dynamo.test_case
from torch.testing._internal.common_utils import run_tests


class TestGroupTensorsByDeviceAndDtype(torch._dynamo.test_case.TestCase):
    """Tests for the group_tensors_by_device_and_dtype polyfill."""

    def test_polyfill_matches_cpp_single_list(self):
        """Test that polyfill matches C++ implementation for a single list."""
        from torch._dynamo.polyfills import group_tensors_by_device_and_dtype

        list1 = [
            torch.randn(4, dtype=torch.float32),
            torch.randn(4, dtype=torch.float64),
            torch.randn(4, dtype=torch.float32),
            torch.randn(4, dtype=torch.float64),
        ]
        tensorlistlist = [list1]

        cpp_result = torch._C._group_tensors_by_device_and_dtype(tensorlistlist, True)
        polyfill_result = group_tensors_by_device_and_dtype(
            tensorlistlist, with_indices=True
        )

        self.assertEqual(set(cpp_result), set(polyfill_result))
        for key in cpp_result:
            cpp_lists, cpp_indices = cpp_result[key]
            polyfill_lists, polyfill_indices = polyfill_result[key]
            self.assertEqual(len(cpp_lists), len(polyfill_lists))
            for cpp_l, polyfill_l in zip(cpp_lists, polyfill_lists):
                self.assertEqual(len(cpp_l), len(polyfill_l))
                for cpp_t, polyfill_t in zip(cpp_l, polyfill_l):
                    if cpp_t is None:
                        self.assertIsNone(polyfill_t)
                    else:
                        self.assertTrue(torch.equal(cpp_t, polyfill_t))
            self.assertEqual(cpp_indices, polyfill_indices)

    def test_polyfill_matches_cpp_multiple_lists(self):
        """Test that polyfill matches C++ implementation for multiple lists."""
        from torch._dynamo.polyfills import group_tensors_by_device_and_dtype

        list1 = [
            torch.randn(4, dtype=torch.float32),
            torch.randn(4, dtype=torch.float64),
            torch.randn(4, dtype=torch.float32),
        ]
        list2 = [torch.rand_like(t) for t in list1]
        list3 = [torch.rand_like(t) for t in list1]
        tensorlistlist = [list1, list2, list3]

        cpp_result = torch._C._group_tensors_by_device_and_dtype(tensorlistlist, True)
        polyfill_result = group_tensors_by_device_and_dtype(
            tensorlistlist, with_indices=True
        )

        self.assertEqual(set(cpp_result), set(polyfill_result))
        for key in cpp_result:
            cpp_lists, cpp_indices = cpp_result[key]
            polyfill_lists, polyfill_indices = polyfill_result[key]
            self.assertEqual(len(cpp_lists), len(polyfill_lists))
            for cpp_l, polyfill_l in zip(cpp_lists, polyfill_lists):
                self.assertEqual(len(cpp_l), len(polyfill_l))
                for cpp_t, polyfill_t in zip(cpp_l, polyfill_l):
                    if cpp_t is None:
                        self.assertIsNone(polyfill_t)
                    else:
                        self.assertTrue(torch.equal(cpp_t, polyfill_t))
            self.assertEqual(cpp_indices, polyfill_indices)

    def test_polyfill_matches_cpp_with_nones(self):
        """Test that polyfill matches C++ implementation with None values."""
        from torch._dynamo.polyfills import group_tensors_by_device_and_dtype

        list1 = [
            torch.randn(4, dtype=torch.float32),
            torch.randn(4, dtype=torch.float64),
            torch.randn(4, dtype=torch.float32),
        ]
        list2 = [None for _ in list1]
        list3 = [torch.rand_like(t) for t in list1]
        tensorlistlist = [list1, list2, list3]

        cpp_result = torch._C._group_tensors_by_device_and_dtype(tensorlistlist, True)
        polyfill_result = group_tensors_by_device_and_dtype(
            tensorlistlist, with_indices=True
        )

        self.assertEqual(set(cpp_result), set(polyfill_result))
        for key in cpp_result:
            cpp_lists, cpp_indices = cpp_result[key]
            polyfill_lists, polyfill_indices = polyfill_result[key]
            self.assertEqual(len(cpp_lists), len(polyfill_lists))
            for cpp_l, polyfill_l in zip(cpp_lists, polyfill_lists):
                self.assertEqual(len(cpp_l), len(polyfill_l))
                for cpp_t, polyfill_t in zip(cpp_l, polyfill_l):
                    if cpp_t is None:
                        self.assertIsNone(polyfill_t)
                    else:
                        self.assertTrue(torch.equal(cpp_t, polyfill_t))
            self.assertEqual(cpp_indices, polyfill_indices)

    def test_polyfill_matches_cpp_without_indices(self):
        """Test that polyfill matches C++ implementation without indices."""
        from torch._dynamo.polyfills import group_tensors_by_device_and_dtype

        list1 = [
            torch.randn(4, dtype=torch.float32),
            torch.randn(4, dtype=torch.float64),
        ]
        list2 = [torch.rand_like(t) for t in list1]
        tensorlistlist = [list1, list2]

        cpp_result = torch._C._group_tensors_by_device_and_dtype(tensorlistlist, False)
        polyfill_result = group_tensors_by_device_and_dtype(
            tensorlistlist, with_indices=False
        )

        self.assertEqual(set(cpp_result), set(polyfill_result))
        for key in cpp_result:
            cpp_lists, cpp_indices = cpp_result[key]
            polyfill_lists, polyfill_indices = polyfill_result[key]
            self.assertEqual(len(cpp_lists), len(polyfill_lists))
            for cpp_l, polyfill_l in zip(cpp_lists, polyfill_lists):
                self.assertEqual(len(cpp_l), len(polyfill_l))
                for cpp_t, polyfill_t in zip(cpp_l, polyfill_l):
                    if cpp_t is None:
                        self.assertIsNone(polyfill_t)
                    else:
                        self.assertTrue(torch.equal(cpp_t, polyfill_t))
            # When with_indices=False, indices should be empty
            self.assertEqual(cpp_indices, polyfill_indices)
            self.assertEqual(polyfill_indices, [])

    def test_polyfill_empty_input(self):
        """Test that polyfill handles empty input correctly."""
        from torch._dynamo.polyfills import group_tensors_by_device_and_dtype

        result = group_tensors_by_device_and_dtype([], with_indices=True)
        self.assertEqual(result, {})

        result = group_tensors_by_device_and_dtype([[]], with_indices=True)
        self.assertEqual(result, {})

    def test_polyfill_groups_correctly(self):
        """Test that polyfill groups tensors correctly by device and dtype."""
        from torch._dynamo.polyfills import group_tensors_by_device_and_dtype

        t_f32_0 = torch.randn(4, dtype=torch.float32)
        t_f32_1 = torch.randn(4, dtype=torch.float32)
        t_f64_0 = torch.randn(4, dtype=torch.float64)
        t_f64_1 = torch.randn(4, dtype=torch.float64)

        list1 = [t_f32_0, t_f64_0, t_f32_1, t_f64_1]
        tensorlistlist = [list1]

        result = group_tensors_by_device_and_dtype(tensorlistlist, with_indices=True)

        # Should have two groups: float32 and float64
        self.assertEqual(len(result), 2)

        device = torch.device("cpu")
        f32_key = (device, torch.float32)
        f64_key = (device, torch.float64)

        self.assertIn(f32_key, result)
        self.assertIn(f64_key, result)

        f32_lists, f32_indices = result[f32_key]
        f64_lists, f64_indices = result[f64_key]

        # Check float32 group
        self.assertEqual(len(f32_lists[0]), 2)
        self.assertTrue(torch.equal(f32_lists[0][0], t_f32_0))
        self.assertTrue(torch.equal(f32_lists[0][1], t_f32_1))
        self.assertEqual(f32_indices, [0, 2])

        # Check float64 group
        self.assertEqual(len(f64_lists[0]), 2)
        self.assertTrue(torch.equal(f64_lists[0][0], t_f64_0))
        self.assertTrue(torch.equal(f64_lists[0][1], t_f64_1))
        self.assertEqual(f64_indices, [1, 3])


if __name__ == "__main__":
    run_tests()
