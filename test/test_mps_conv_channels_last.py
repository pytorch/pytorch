"""
Test for MPS convolution backward memory format preservation.

This tests the fix for the issue where convolution_backward on MPS
returns grad_input with contiguous strides when grad_output is contiguous
but the original input was channels_last.

This breaks torch.compile with inductor backend which asserts expected strides.

Related issues: #142344, #144570
"""

import torch
import unittest


class TestMPSConvBackwardMemoryFormat(unittest.TestCase):
    """Test that MPS conv backward preserves input memory format."""

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS not available")
    def test_conv_backward_preserves_channels_last(self):
        """
        Test that convolution_backward returns grad_input with channels_last
        strides when the original input was channels_last, regardless of
        grad_output's memory format.
        """
        # grad_output: contiguous strides (this is what inductor may produce)
        grad_out = torch.randn(8, 3072, 8, 8, device='mps', dtype=torch.float16)
        self.assertEqual(grad_out.stride(), (196608, 64, 8, 1))  # contiguous

        # input: channels_last strides
        x = torch.randn(8, 768, 8, 8, device='mps', dtype=torch.float16)
        x = x.to(memory_format=torch.channels_last)
        self.assertEqual(x.stride(), (49152, 1, 6144, 768))  # channels_last

        # weight: contiguous (OIHW)
        w = torch.randn(3072, 768, 3, 3, device='mps', dtype=torch.float16)

        # Call convolution_backward directly
        result = torch.ops.aten.convolution_backward.default(
            grad_out, x, w,
            [0],        # bias_sizes (no bias)
            [1, 1],     # stride
            [1, 1],     # padding
            [1, 1],     # dilation
            False,      # transposed
            [0, 0],     # output_padding
            1,          # groups
            [True, True, False]  # output_mask
        )

        grad_input = result[0]

        # grad_input should preserve input's channels_last format
        expected_strides = (49152, 1, 6144, 768)  # channels_last
        self.assertEqual(
            grad_input.stride(),
            expected_strides,
            f"grad_input has wrong strides: {grad_input.stride()}, expected {expected_strides}"
        )

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS not available")
    def test_conv_backward_contiguous_input_stays_contiguous(self):
        """
        Test that when input is contiguous, grad_input is also contiguous
        (no regression from the fix).
        """
        grad_out = torch.randn(8, 3072, 8, 8, device='mps', dtype=torch.float16)
        x = torch.randn(8, 768, 8, 8, device='mps', dtype=torch.float16)  # contiguous
        w = torch.randn(3072, 768, 3, 3, device='mps', dtype=torch.float16)

        result = torch.ops.aten.convolution_backward.default(
            grad_out, x, w,
            [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]
        )

        grad_input = result[0]

        # grad_input should be contiguous when input was contiguous
        expected_strides = (49152, 64, 8, 1)  # contiguous
        self.assertEqual(
            grad_input.stride(),
            expected_strides,
            f"grad_input has wrong strides: {grad_input.stride()}, expected {expected_strides}"
        )

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS not available")
    def test_conv_backward_channels_last_grad_out_and_input(self):
        """
        Test when both grad_output and input are channels_last.
        """
        grad_out = torch.randn(8, 3072, 8, 8, device='mps', dtype=torch.float16)
        grad_out = grad_out.to(memory_format=torch.channels_last)
        x = torch.randn(8, 768, 8, 8, device='mps', dtype=torch.float16)
        x = x.to(memory_format=torch.channels_last)
        w = torch.randn(3072, 768, 3, 3, device='mps', dtype=torch.float16)

        result = torch.ops.aten.convolution_backward.default(
            grad_out, x, w,
            [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]
        )

        grad_input = result[0]

        # grad_input should be channels_last
        expected_strides = (49152, 1, 6144, 768)  # channels_last
        self.assertEqual(
            grad_input.stride(),
            expected_strides,
            f"grad_input has wrong strides: {grad_input.stride()}, expected {expected_strides}"
        )

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS not available")
    def test_conv2d_backward_via_autograd(self):
        """
        Test the fix via normal autograd (nn.Conv2d).
        """
        torch.manual_seed(42)

        # Create channels_last input
        x = torch.randn(4, 64, 32, 32, device='mps', dtype=torch.float32, requires_grad=True)
        x = x.to(memory_format=torch.channels_last)

        # Conv layer
        conv = torch.nn.Conv2d(64, 128, 3, padding=1, device='mps', dtype=torch.float32)

        # Forward
        y = conv(x)

        # Create contiguous grad_output (simulating what inductor might do)
        grad_out = torch.randn_like(y).contiguous()

        # Backward
        y.backward(grad_out)

        # x.grad should preserve channels_last format
        self.assertTrue(
            x.grad.is_contiguous(memory_format=torch.channels_last),
            f"x.grad should be channels_last, got strides {x.grad.stride()}"
        )


if __name__ == '__main__':
    unittest.main()
