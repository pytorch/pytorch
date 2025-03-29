import unittest
from unittest.mock import patch, MagicMock

import torch

# Import the functions to test
from torch.cuda.nvtx import (
    range_push, range_pop, range_start, range_end, mark,
    enable_tensor_returns, disable_tensor_returns
)


class TestNVTXTensorReturns(unittest.TestCase):
    """Test the tensor return functionality of NVTX functions."""

    def setUp(self) -> None:
        """Set up the test environment."""
        # Make sure we start with tensor returns disabled
        disable_tensor_returns()

    def tearDown(self) -> None:
        """Clean up after tests."""
        # Make sure we end with tensor returns disabled
        disable_tensor_returns()

    @patch('torch.cuda.nvtx._nvtx')
    def test_default_behavior(self, mock_nvtx: MagicMock) -> None:
        """Test that the default behavior returns original values."""
        # Set up mock returns
        mock_nvtx.rangePushA.return_value = 1
        mock_nvtx.rangePop.return_value = 0
        mock_nvtx.rangeStartA.return_value = 12345
        mock_nvtx.markA.return_value = None

        # Test default behavior
        self.assertEqual(range_push("test"), 1)
        self.assertEqual(range_pop(), 0)
        self.assertEqual(range_start("test"), 12345)
        self.assertIsNone(range_end(12345))
        self.assertIsNone(mark("test event"))

    @patch('torch.cuda.nvtx._nvtx')
    def test_tensor_returns_enabled(self, mock_nvtx: MagicMock) -> None:
        """Test that all functions return tensors when tensor returns are enabled."""
        # Set up mock returns
        mock_nvtx.rangePushA.return_value = 1
        mock_nvtx.rangePop.return_value = 0
        mock_nvtx.rangeStartA.return_value = 12345
        mock_nvtx.markA.return_value = None

        # Enable tensor returns
        enable_tensor_returns()

        # Test that all functions return tensors
        self.assertIsInstance(range_push("test"), torch.Tensor)
        self.assertEqual(range_push("test").item(), 1)

        self.assertIsInstance(range_pop(), torch.Tensor)
        self.assertEqual(range_pop().item(), 0)

        self.assertIsInstance(range_start("test"), torch.Tensor)
        self.assertEqual(range_start("test").item(), 12345)

        self.assertIsInstance(range_end(12345), torch.Tensor)
        self.assertEqual(range_end(12345).item(), 0)  # Returns zero tensor

        self.assertIsInstance(mark("test event"), torch.Tensor)
        self.assertEqual(mark("test event").item(), 0)  # Returns zero tensor

    @patch('torch.cuda.nvtx._nvtx')
    def test_switching_modes(self, mock_nvtx: MagicMock) -> None:
        """Test switching between tensor returns modes."""
        # Set up mock returns
        mock_nvtx.rangePushA.return_value = 5

        # Test switching between modes
        self.assertEqual(range_push("test"), 5)  # Default mode

        enable_tensor_returns()
        push_result = range_push("test")
        self.assertIsInstance(push_result, torch.Tensor)
        self.assertEqual(push_result.item(), 5)

        disable_tensor_returns()
        self.assertEqual(range_push("test"), 5)  # Back to default mode

    @patch('torch.cuda.nvtx._nvtx')
    def test_complex_return_types(self, mock_nvtx: MagicMock) -> None:
        """Test handling of complex return types that can't be directly converted to tensors."""
        # Set up a complex return value that can't be directly converted to tensor
        complex_obj = {"key": "value"}
        mock_nvtx.deviceRangeStart.return_value = complex_obj


if __name__ == "__main__":
    unittest.main()
