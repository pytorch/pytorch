# Owner(s): ["module: inductor"]

"""
Test suite for Issue #155690: Storage offset preservation in clone_preserve_strides

This test validates the fix for the silent correctness bug where clone_preserve_strides
was not preserving storage_offset, causing incorrect behavior in torch.compile.
"""

import torch
import torch._inductor.utils as inductor_utils
from torch._inductor.test_case import run_tests, TestCase


class TestStorageOffsetFix(TestCase):
    """Test storage offset preservation in clone_preserve_strides function."""

    def test_clone_preserve_strides_basic_storage_offset(self):
        """Test basic storage offset preservation."""
        # Create tensor with storage offset
        base = torch.randn(1000, dtype=torch.float32)
        offset_tensor = base[10:510]  # storage_offset = 10
        
        # Test our fixed function
        cloned = inductor_utils.clone_preserve_strides(offset_tensor)
        
        # Verify storage offset is preserved
        self.assertEqual(offset_tensor.storage_offset(), cloned.storage_offset())
        
        # Verify values are preserved
        self.assertTrue(torch.allclose(offset_tensor, cloned))
        
        # Verify shapes and strides match
        self.assertEqual(offset_tensor.size(), cloned.size())
        self.assertEqual(offset_tensor.stride(), cloned.stride())

    def test_clone_preserve_strides_zero_offset(self):
        """Test that zero offset case still works (fast path)."""
        # Create tensor with zero storage offset
        tensor = torch.randn(500, dtype=torch.float32)
        
        # Test our fixed function
        cloned = inductor_utils.clone_preserve_strides(tensor)
        
        # Verify storage offset is preserved (should be 0)
        self.assertEqual(tensor.storage_offset(), cloned.storage_offset())
        self.assertEqual(cloned.storage_offset(), 0)
        
        # Verify values are preserved
        self.assertTrue(torch.allclose(tensor, cloned))

    def test_clone_preserve_strides_multiple_dtypes(self):
        """Test storage offset preservation with different data types."""
        dtypes = [
            torch.float16, torch.float32, torch.float64,
            torch.int32, torch.int64, torch.uint8, torch.bool
        ]

        for dtype in dtypes:
            with self.subTest(dtype=dtype):
                if dtype == torch.bool:
                    base = torch.randint(0, 2, (1000,), dtype=dtype)
                elif dtype.is_floating_point:
                    base = torch.randn(1000, dtype=dtype)
                elif dtype == torch.uint8:
                    base = torch.randint(0, 255, (1000,), dtype=dtype)  # Fixed range for uint8
                else:
                    base = torch.randint(-100, 100, (1000,), dtype=dtype)
                
                offset_tensor = base[5:505]  # storage_offset = 5
                cloned = inductor_utils.clone_preserve_strides(offset_tensor)
                
                # Verify storage offset preservation
                self.assertEqual(offset_tensor.storage_offset(), cloned.storage_offset())
                
                # Verify values preservation
                self.assertTrue(torch.equal(offset_tensor, cloned))

    def test_clone_preserve_strides_extreme_offsets(self):
        """Test with very large storage offsets."""
        base = torch.randn(50000, dtype=torch.float32)
        
        # Test various large offsets
        offsets = [1000, 10000, 25000]
        
        for offset in offsets:
            with self.subTest(offset=offset):
                offset_tensor = base[offset:offset+1000]
                cloned = inductor_utils.clone_preserve_strides(offset_tensor)
                
                self.assertEqual(offset_tensor.storage_offset(), cloned.storage_offset())
                self.assertTrue(torch.allclose(offset_tensor, cloned))

    def test_clone_preserve_strides_empty_tensor(self):
        """Test with empty tensors."""
        empty = torch.empty(0, dtype=torch.float32)
        cloned = inductor_utils.clone_preserve_strides(empty)
        
        self.assertEqual(empty.storage_offset(), cloned.storage_offset())
        self.assertEqual(empty.size(), cloned.size())

    def test_clone_preserve_strides_single_element(self):
        """Test with single element tensors."""
        single = torch.tensor([42.0])
        cloned = inductor_utils.clone_preserve_strides(single)
        
        self.assertEqual(single.storage_offset(), cloned.storage_offset())
        self.assertTrue(torch.equal(single, cloned))

    def test_clone_preserve_strides_aliasing_behavior(self):
        """Test that aliasing behavior is preserved correctly."""
        base = torch.randn(10000, dtype=torch.float32)
        
        # Create overlapping views
        view1 = base[100:5100]  # storage_offset = 100
        view2 = base[2000:7000]  # storage_offset = 2000, overlaps with view1
        
        # Clone both views
        cloned1 = inductor_utils.clone_preserve_strides(view1)
        cloned2 = inductor_utils.clone_preserve_strides(view2)
        
        # Verify storage offsets preserved
        self.assertEqual(view1.storage_offset(), cloned1.storage_offset())
        self.assertEqual(view2.storage_offset(), cloned2.storage_offset())
        
        # Verify values preserved
        self.assertTrue(torch.allclose(view1, cloned1))
        self.assertTrue(torch.allclose(view2, cloned2))
        
        # Verify they are independent (no aliasing in cloned versions)
        original_cloned1_value = cloned1[0].item()
        cloned1[0] = 999.0
        # cloned2 should not be affected
        self.assertNotEqual(cloned2[0].item(), 999.0)
        # Restore for cleanup
        cloned1[0] = original_cloned1_value

    def test_torch_compile_integration(self):
        """Test that the fix works with torch.compile."""
        def simple_operation(x, y):
            return x + y
        
        # Create tensors with storage offsets
        base1 = torch.randn(5000, dtype=torch.float32)
        base2 = torch.randn(5000, dtype=torch.float32)
        
        tensor1 = base1[10:1010]  # storage_offset = 10
        tensor2 = base2[20:1020]  # storage_offset = 20
        
        # Test eager mode
        eager_result = simple_operation(tensor1, tensor2)
        
        # Test compiled mode
        compiled_fn = torch.compile(simple_operation, backend="inductor")
        compiled_result = compiled_fn(tensor1, tensor2)
        
        # Results should match
        self.assertTrue(torch.allclose(eager_result, compiled_result, rtol=1e-5))

    def test_memory_corruption_prevention(self):
        """Test that our fix prevents memory corruption."""
        # Create a pattern that might expose memory issues
        base = torch.randn(5000, dtype=torch.float32)
        base.fill_(42.0)
        
        # Create offset view
        offset_view = base[1000:4000]
        
        # Clone it
        cloned = inductor_utils.clone_preserve_strides(offset_view)
        
        # Modify original
        base.fill_(99.0)
        
        # Cloned should still have original values
        self.assertTrue(torch.allclose(cloned, torch.full_like(cloned, 42.0)))
        
        # Check storage offset preserved
        self.assertEqual(offset_view.storage_offset(), cloned.storage_offset())


if __name__ == "__main__":
    run_tests()
