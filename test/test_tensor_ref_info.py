# Owner(s): ["module: unknown"]

import torch
from torch.testing._internal.common_utils import run_tests, skipIfTorchDynamo, TestCase
from torch.utils._tensor_ref_info import is_safe_to_inplace, tensor_ref_info, TensorRefInfo


@skipIfTorchDynamo("Refcount introspection is not meaningful under Dynamo")
class TestTensorRefInfo(TestCase):
    def test_basic_counts(self):
        t = torch.randn(10)
        info = tensor_ref_info(t)
        self.assertIsInstance(info, TensorRefInfo)
        self.assertEqual(info.tensor_use_count, 1)
        self.assertGreaterEqual(info.tensor_weak_use_count, 1)
        self.assertEqual(info.storage_use_count, 1)
        self.assertGreaterEqual(info.storage_weak_use_count, 1)
        self.assertFalse(info.is_view)
        self.assertFalse(info.is_cow)
        self.assertEqual(info.storage_offset, 0)
        self.assertEqual(info.data_ptr, t.data_ptr())

    def test_view_increases_storage_use_count(self):
        t = torch.randn(10)
        base_count = t._storage_use_count()
        v = t.view(2, 5)
        self.assertEqual(t._storage_use_count(), base_count + 1)
        info = tensor_ref_info(v)
        self.assertTrue(info.is_view)
        del v
        self.assertEqual(t._storage_use_count(), base_count)

    def test_slice_is_view_with_offset(self):
        t = torch.randn(10)
        v = t[3:]
        info = tensor_ref_info(v)
        self.assertTrue(info.is_view)
        self.assertGreater(info.storage_offset, 0)
        self.assertEqual(info.data_ptr, t.data_ptr() + 3 * t.element_size())

    def test_clone_does_not_share_storage(self):
        t = torch.randn(10)
        c = t.clone()
        self.assertEqual(t._storage_use_count(), 1)
        self.assertEqual(c._storage_use_count(), 1)
        self.assertNotEqual(t.data_ptr(), c.data_ptr())

    def test_cow_detection(self):
        t = torch.randn(10)
        c = t._lazy_clone()
        self.assertTrue(torch._C._is_cow_tensor(t))
        self.assertTrue(torch._C._is_cow_tensor(c))
        info = tensor_ref_info(t)
        self.assertTrue(info.is_cow)

    def test_safe_to_inplace_standalone(self):
        t = torch.randn(10)
        self.assertTrue(is_safe_to_inplace(t))

    def test_not_safe_with_view(self):
        t = torch.randn(10)
        v = t.view(2, 5)  # noqa: F841
        self.assertFalse(is_safe_to_inplace(t))

    def test_not_safe_cow(self):
        t = torch.randn(10)
        c = t._lazy_clone()  # noqa: F841
        self.assertFalse(is_safe_to_inplace(t))

    def test_safe_after_view_deleted(self):
        t = torch.randn(10)
        v = t.view(2, 5)
        self.assertFalse(is_safe_to_inplace(t))
        del v
        self.assertTrue(is_safe_to_inplace(t))

    def test_multiple_views(self):
        t = torch.randn(10)
        v1 = t[:]  # noqa: F841
        v2 = t.view(2, 5)  # noqa: F841
        self.assertEqual(t._storage_use_count(), 3)
        self.assertFalse(is_safe_to_inplace(t))

    def test_weak_use_count_method(self):
        t = torch.randn(10)
        self.assertGreaterEqual(t._weak_use_count(), 1)
        self.assertGreaterEqual(t._storage_weak_use_count(), 1)


if __name__ == "__main__":
    run_tests()
