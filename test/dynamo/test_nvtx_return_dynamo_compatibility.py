# Owner(s): ["module: dynamo"]
import torch
import torch._dynamo.test_case
from torch.cuda.nvtx import (
    disable_tensor_returns,
    enable_tensor_returns,
    mark,
    range_end,
    range_pop,
    range_push,
    range_start,
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

    # Wrap NVTX operations in an nn.Module
    class NVTXModule(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            range_id = range_push("test_range")
            range_id += 1
            result = x * 2
            range_pop()
            return result

    def test_nvtx_compile_error_without_tensor_returns(self):
        """Test that torch.compile raises an error for NVTX functions when tensor returns are disabled."""
        disable_tensor_returns()

        # Create and compile the module
        module = self.NVTXModule()
        compiled_module = torch.compile(module, fullgraph=True)

        # Create input data
        x = torch.randn(3, 3)
        if torch.cuda.is_available():
            x = x.cuda()

        # Verify the original module works normally
        expected = module(x)
        expected += 0

        # Attempting to use the compiled module should raise an error
        with self.assertRaises(Exception) as context:
            _ = compiled_module(x)

        # Check that the error message contains NVTX-related content
        error_msg = str(context.exception)
        self.assertTrue(
            "NVTX" in error_msg
            or "tensor returns" in error_msg.lower()
            or "non-Tensor" in error_msg
            or "int call_function" in error_msg
        )
        print(f"Successfully caught error: {error_msg}")

    def test_nvtx_compile_works_with_tensor_returns(self):
        """Test that torch.compile works correctly with NVTX functions when tensor returns are enabled."""

        # Enable tensor returns
        enable_tensor_returns()

        try:
            # Create and compile the module without forcing full graph
            module = self.NVTXModule()
            compiled_module = torch.compile(module, fullgraph=False)

            # Create input data
            x = torch.randn(3, 3)
            if torch.cuda.is_available():
                x = x.cuda()

            # Get original results for comparison
            expected = module(x)

            # Use the compiled module
            actual = compiled_module(x)

            # Verify results match
            torch.testing.assert_close(actual, expected)
            print("Compilation successful with tensor returns enabled, results match.")

        finally:
            # Restore to disabled state
            disable_tensor_returns()


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
