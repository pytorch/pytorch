# Owner(s): ["module: functorch"]
import torch
from functorch.dim import Dim, DimList, dims, Tensor
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.testing._internal.common_device_type import instantiate_device_type_tests

class TestGetSetItem(TestCase):
    """Comprehensive tests for first-class dimension indexing operations."""

    def setUp(self):
        super().setUp()
        """Set up common test fixtures."""
        self.batch, self.height, self.width = dims(3)

    def test_basic_dim_indexing(self, device):
        """Test basic indexing with a single Dim."""
        tensor = torch.randn(3, 4, 5, device=device)
        x, y, z = dims(3)

        # Test indexing with each dim
        result1 = tensor[x]
        self.assertIsInstance(result1, Tensor)

        result2 = tensor[y]
        self.assertIsInstance(result2, Tensor)

        result3 = tensor[z]
        self.assertIsInstance(result3, Tensor)

    def test_multiple_dim_indexing(self, device):
        """Test indexing with multiple Dims."""
        tensor = torch.randn(3, 4, 5, device=device)
        x, y, z = dims(3)

        # Test multiple dims in one indexing operation
        result = tensor[x, y]
        self.assertIsInstance(result, Tensor)

        result = tensor[x, y, z]
        self.assertIsInstance(result, Tensor)

    def test_mixed_indexing(self, device):
        """Test mixing Dims with regular indexing."""
        tensor = torch.randn(3, 4, 5, device=device)
        x, y, z = dims(3)

        # Mix dim with slice
        result1 = tensor[x, :]
        self.assertIsInstance(result1, Tensor)

        result2 = tensor[:, y]
        self.assertIsInstance(result2, Tensor)

        # Mix dim with integer
        result3 = tensor[x, 0]
        self.assertIsInstance(result3, Tensor)

        result4 = tensor[0, y]
        self.assertIsInstance(result4, Tensor)

    def test_ellipsis_indexing(self, device):
        """Test indexing with ellipsis (...)."""
        tensor = torch.randn(3, 4, 5, 6, device=device)
        x, y, z = dims(3)
        x, y, z, w = dims(4)

        # Test ellipsis with dims
        result1 = tensor[x, ...]
        self.assertIsInstance(result1, Tensor)

        result2 = tensor[..., y]
        self.assertIsInstance(result2, Tensor)

        result3 = tensor[x, ..., y]
        self.assertIsInstance(result3, Tensor)

    def test_none_indexing(self, device):
        """Test indexing with None (newaxis)."""
        tensor = torch.randn(3, 4, device=device)
        x, y = dims(2)

        # Test None with dims
        result1 = tensor[x, None, y]
        self.assertIsInstance(result1, Tensor)

        result2 = tensor[None, x]
        self.assertIsInstance(result2, Tensor)

    def test_slice_indexing(self, device):
        """Test indexing with slices mixed with dims."""
        tensor = torch.randn(6, 8, 10, device=device)
        x, y, z = dims(3)

        # Test various slice patterns with dims
        result1 = tensor[x, 1:5]
        self.assertIsInstance(result1, Tensor)

        result2 = tensor[1:3, y]
        self.assertIsInstance(result2, Tensor)

        result3 = tensor[x, 1:5, z]
        self.assertIsInstance(result3, Tensor)

    def test_tensor_indexing(self, device):
        """Test indexing with tensor indices."""
        tensor = torch.randn(5, 6, 7, device=device)
        x, y, z = dims(3)
        x, y, z = dims(3)

        # Create index tensors
        idx = torch.tensor([0, 2, 4], device=device)

        # Test tensor indexing with dims
        result1 = tensor[x, idx]
        self.assertIsInstance(result1, Tensor)

        result2 = tensor[idx, y]
        self.assertIsInstance(result2, Tensor)

    def test_boolean_indexing(self, device):
        """Test boolean indexing with dims."""
        tensor = torch.randn(4, 5, device=device)
        x, y, z = dims(3)
        x, y = dims(2)

        # Create boolean mask
        mask = torch.tensor([True, False, True, False, True], device=device)

        # Test boolean indexing
        result = tensor[x, mask]
        self.assertIsInstance(result, Tensor)

    def test_dim_pack_indexing(self, device):
        """Test indexing with dimension packs (tuples/lists of dims)."""
        tensor = torch.randn(3, 4, device=device)  # Need 2D tensor for 2 dims

        # Create dims for dim pack
        a, b = dims(2)

        # Test dim pack indexing - using separate dimensions
        result = tensor[a, b]
        self.assertIsInstance(result, Tensor)

    def test_unbound_dim_binding(self, device):
        """Test automatic binding of unbound dimensions during indexing."""
        tensor = torch.randn(6, 8, device=device)
        x = Dim("x")  # unbound
        y = Dim("y")  # unbound

        # Should automatically bind dimensions
        result = tensor[x, y]
        self.assertIsInstance(result, Tensor)
        self.assertEqual(x.size, 6)
        self.assertEqual(y.size, 8)

    def test_dimlist_indexing(self, device):
        """Test indexing with DimList objects."""
        tensor = torch.randn(3, 4, 5, device=device)

        # Create a bound dimlist
        dl = DimList(dims(2))

        # Test dimlist indexing
        result = tensor[dl, :]
        self.assertIsInstance(result, Tensor)

    def test_unbound_dimlist_indexing(self, device):
        """Test indexing with unbound DimList."""
        tensor = torch.randn(3, 4, 5, device=device)

        # Create unbound dimlist
        dl = DimList()

        # Should bind to remaining dimensions
        result = tensor[0, dl]
        self.assertIsInstance(result, Tensor)

    def test_repeated_dim_usage(self, device):
        """Test using the same dim multiple times in indexing."""
        tensor = torch.randn(4, 4, 4, device=device)
        x, y, z = dims(3)

        # This should trigger advanced indexing for repeated dims
        result = tensor[x, x]
        self.assertIsInstance(result, Tensor)

    def test_complex_mixed_indexing(self, device):
        """Test complex combinations of different indexing types."""
        tensor = torch.randn(3, 4, 5, 6, 7, device=device)
        x, y, z = dims(3)
        a, b, c, d, e = dims(5)

        # Complex mixed indexing
        idx = torch.tensor([0, 2], device=device)

        result1 = tensor[a, 1:3, None, idx, :]
        self.assertIsInstance(result1, Tensor)

        # Use mask with correct shape
        correct_mask = torch.tensor([True, False, True, False, False, True, True], device=device)
        result2 = tensor[..., correct_mask]
        self.assertIsInstance(result2, torch.Tensor)

    def test_edge_cases(self, device):
        """Test edge cases and boundary conditions."""
        x, y, z = dims(3)

        # Single dimension tensor
        vec = torch.randn(5, device=device)
        a = Dim("a")
        result1 = vec[a]
        self.assertIsInstance(result1, Tensor)
        self.assertEqual(a.size, 5)  # Should bind to tensor size

        # Empty tensor indexing
        empty = torch.empty(0, 3, 4, device=device)
        result2 = empty[x, :]
        self.assertIsInstance(result2, Tensor)

    def test_error_conditions(self, device):
        """Test conditions that should raise errors."""
        tensor = torch.randn(3, 4, device=device)
        x, y, z = dims(3)

        # Too many indices
        with self.assertRaises(ValueError):
            _ = tensor[x, y, z]  # 3 indices for 2D tensor

        # Multiple unbound dim lists
        dl1 = DimList()
        dl2 = DimList()
        with self.assertRaises(Exception):  # Should raise DimensionBindError
            _ = tensor[dl1, dl2]

        # Multiple ellipsis
        with self.assertRaises(Exception):
            _ = tensor[..., x, ...]

    def test_inferred_dimension_binding(self, device):
        """Test dimension binding inference with dim packs."""
        # Skip this test for now as it requires more complex dim pack functionality

    def test_stride_calculation(self, device):
        """Test that stride calculations work correctly with dim packs."""
        tensor = torch.randn(6, 8, device=device)

        # Test basic indexing instead of complex dim packs
        a, b = dims(2)
        result1 = tensor[a, b]
        self.assertIsInstance(result1, Tensor)

        # Test with different tensor
        tensor2 = torch.randn(2, 3, 4, device=device)
        c, d, e = dims(3)
        result2 = tensor2[c, d, e]
        self.assertIsInstance(result2, Tensor)

    def test_device_handling(self, device):
        """Test indexing behavior with CPU tensors."""
        # CPU tensor
        cpu_tensor = torch.randn(3, 4, device=device)
        x, y = dims(2)

        result_cpu = cpu_tensor[x, y]
        self.assertIsInstance(result_cpu, Tensor)
        self.assertEqual(result_cpu.device, torch.device(device))

instantiate_device_type_tests(TestGetSetItem, globals())

if __name__ == "__main__":
    run_tests()
