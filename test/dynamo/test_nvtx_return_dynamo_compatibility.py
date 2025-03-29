# Owner(s): ["module: nvtx"]
import torch
import torch._dynamo.test_case
from torch.cuda.nvtx import (
    range_push, range_pop, range_start, range_end, mark,
    enable_tensor_returns, disable_tensor_returns, _tensor_compatible
)


class NVTXTensorReturnsTests(torch._dynamo.test_case.TestCase):
    def setUp(self):
        # Make sure we start with tensor returns disabled
        disable_tensor_returns()

    def tearDown(self):
        # Make sure we end with tensor returns disabled
        disable_tensor_returns()

    def test_default_behavior(self):
        """Test that the default behavior returns original values."""
        # Test default behavior without tensor returns
        self.assertIsInstance(range_push("test"), int)
        self.assertIsInstance(range_pop(), int) 
        self.assertIsInstance(range_start("test"), int)
        self.assertIsNone(range_end(12345))
        self.assertIsNone(mark("test event"))
        
        # Run twice to ensure consistency
        self.assertIsInstance(range_push("test"), int)
        self.assertIsInstance(range_pop(), int)

    def test_tensor_returns_enabled(self):
        """Test behavior when tensor returns are enabled."""
        # Enable tensor returns
        enable_tensor_returns()
        
        try:
            # Functions that return values should return tensors
            self.assertIsInstance(range_push("test"), torch.Tensor)
            self.assertIsInstance(range_pop(), torch.Tensor)
            self.assertIsInstance(range_start("test"), torch.Tensor)
            
            # Important correction: range_end and mark return None even with tensor returns enabled
            # This is because _tensor_compatible() returns None when the original function returns None
            # and no default_val was provided
            self.assertIsNone(range_end(12345))
            self.assertIsNone(mark("test event"))
        finally:
            disable_tensor_returns()

    def test_switching_modes(self):
        """Test switching between tensor returns modes."""
        # Test switching between modes
        self.assertIsInstance(range_push("test"), int)  # Default mode
        
        enable_tensor_returns()
        self.assertIsInstance(range_push("test"), torch.Tensor)
        
        disable_tensor_returns()
        self.assertIsInstance(range_push("test"), int)  # Back to default mode

    def test_behavior_with_default_values(self):
        """Test a custom function with default_val to understand the decorator behavior."""
        # Create a test function with a default value
        @_tensor_compatible(default_val=42)
        def test_func_with_default():
            return None
            
        # Create a test function without a default value
        @_tensor_compatible()
        def test_func_without_default():
            return None
            
        # Test with tensor returns enabled
        enable_tensor_returns()
        try:
            # With default_val, None should be converted to tensor(default_val)
            self.assertIsInstance(test_func_with_default(), torch.Tensor)
            self.assertEqual(test_func_with_default().item(), 42)
            
            # Without default_val, None should stay None
            self.assertIsNone(test_func_without_default())
        finally:
            disable_tensor_returns()


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests
    
    run_tests()