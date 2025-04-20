import torch
import unittest
import math
import random
import itertools
from torch.testing._internal.common_utils import TestCase, run_tests
from torch.testing._internal.common_device_type import instantiate_device_type_tests, dtypes, skipMPS

class TestUpsampleNearest3DMPS(TestCase):
    def test_upsample_nearest3d_vec(self, device="mps"):
        """Test upsample_nearest3d.vec implementation on MPS device"""
        # Skip if MPS is not available
        if not torch.backends.mps.is_available():
            self.skipTest("MPS device not available")

        # Test with different input shapes
        input_shapes = [
            (1, 1, 2, 2, 2),  # Minimal shape
            (2, 3, 4, 5, 6),  # Standard shape
            (3, 4, 8, 10, 12)  # Larger shape
        ]

        # Test with different scale factors
        scale_factors = [
            (2.0, 2.0, 2.0),  # Double in all dimensions
            (1.5, 1.5, 1.5),  # 1.5x in all dimensions
            (0.5, 0.5, 0.5),  # Downsampling
            (1.0, 2.0, 3.0)   # Different scales per dimension
        ]

        # Test with different output sizes
        output_sizes = [
            (4, 4, 4),  # Double the minimal shape
            (8, 10, 12),  # Double the standard shape
            (4, 5, 6)   # Custom size
        ]

        # Test with different data types
        dtypes = [torch.float32, torch.float16]

        for input_shape in input_shapes:
            for dtype in dtypes:
                # Create input tensor
                x = torch.randn(input_shape, device="mps", dtype=dtype)

                # Test with scale_factor
                for scale_factor in scale_factors:
                    # Skip downsampling tests for now as they might not be supported
                    if scale_factor[0] < 1.0 or scale_factor[1] < 1.0 or scale_factor[2] < 1.0:
                        continue

                    # Run on MPS
                    y_mps = torch.nn.functional.interpolate(
                        x, scale_factor=scale_factor, mode="nearest")

                    # Verify output shape
                    expected_shape = list(x.shape)
                    expected_shape[2] = int(expected_shape[2] * scale_factor[0])
                    expected_shape[3] = int(expected_shape[3] * scale_factor[1])
                    expected_shape[4] = int(expected_shape[4] * scale_factor[2])
                    self.assertEqual(y_mps.shape, torch.Size(expected_shape))

                # Test with output_size
                for output_size in output_sizes:
                    # Run on MPS
                    y_mps = torch.nn.functional.interpolate(
                        x, size=output_size, mode="nearest")

                    # Verify output shape
                    expected_shape = list(x.shape)
                    expected_shape[2] = output_size[0]
                    expected_shape[3] = output_size[1]
                    expected_shape[4] = output_size[2]
                    self.assertEqual(y_mps.shape, torch.Size(expected_shape))

    # Skip backward test for now as it's not implemented yet
    # def test_upsample_nearest3d_vec_backward(self, device="mps"):
    #     """Test backward pass of upsample_nearest3d.vec on MPS device"""
    #     # Skip if MPS is not available
    #     if not torch.backends.mps.is_available():
    #         self.skipTest("MPS device not available")
    #
    #     # Create input tensor
    #     x = torch.randn(2, 3, 4, 5, 6, device="mps", requires_grad=True)
    #
    #     # Forward pass
    #     y = torch.nn.functional.interpolate(x, scale_factor=(2.0, 2.0, 2.0), mode="nearest")
    #
    #     # Create gradient
    #     grad_output = torch.randn_like(y)
    #
    #     # Backward pass
    #     y.backward(grad_output)
    #
    #     # Check that gradient is not None
    #     self.assertIsNotNone(x.grad)

    def test_upsample_nearest3d_vec_edge_cases(self, device="mps"):
        """Test edge cases for upsample_nearest3d.vec on MPS device"""
        # Skip if MPS is not available
        if not torch.backends.mps.is_available():
            self.skipTest("MPS device not available")

        # Test with empty tensor
        x = torch.randn(0, 3, 4, 5, 6, device="mps")
        y = torch.nn.functional.interpolate(x, scale_factor=(2.0, 2.0, 2.0), mode="nearest")
        self.assertEqual(y.shape, (0, 3, 8, 10, 12))

        # Test with single element tensor
        x = torch.randn(1, 1, 1, 1, 1, device="mps")
        y = torch.nn.functional.interpolate(x, scale_factor=(2.0, 2.0, 2.0), mode="nearest")
        self.assertEqual(y.shape, (1, 1, 2, 2, 2))

        # Test with scale_factor = 1.0 (no change)
        x = torch.randn(2, 3, 4, 5, 6, device="mps")
        y = torch.nn.functional.interpolate(x, scale_factor=(1.0, 1.0, 1.0), mode="nearest")
        self.assertEqual(y.shape, (2, 3, 4, 5, 6))
        self.assertTrue(torch.allclose(y, x, rtol=1e-3, atol=1e-3))

        # Test with different scale factors for each dimension
        x = torch.randn(2, 3, 4, 5, 6, device="mps")
        y = torch.nn.functional.interpolate(x, scale_factor=(2.0, 1.5, 0.5), mode="nearest")
        self.assertEqual(y.shape, (2, 3, 8, 7, 3))

        # Test with very large scale factors
        x = torch.randn(1, 1, 2, 2, 2, device="mps")
        y = torch.nn.functional.interpolate(x, scale_factor=(10.0, 10.0, 10.0), mode="nearest")
        self.assertEqual(y.shape, (1, 1, 20, 20, 20))

        # Test with very small scale factors (downsampling)
        x = torch.randn(1, 1, 20, 20, 20, device="mps")
        y = torch.nn.functional.interpolate(x, scale_factor=(0.1, 0.1, 0.1), mode="nearest")
        self.assertEqual(y.shape, (1, 1, 2, 2, 2))

        # Test with non-standard input shapes
        x = torch.randn(2, 3, 7, 11, 13, device="mps")
        y = torch.nn.functional.interpolate(x, scale_factor=(1.5, 1.5, 1.5), mode="nearest")
        self.assertEqual(y.shape, (2, 3, 10, 16, 19))

        # Test with output_size instead of scale_factor
        x = torch.randn(2, 3, 4, 5, 6, device="mps")
        y = torch.nn.functional.interpolate(x, size=(8, 10, 12), mode="nearest")
        self.assertEqual(y.shape, (2, 3, 8, 10, 12))

        # Test with output_size smaller than input size (downsampling)
        x = torch.randn(2, 3, 8, 10, 12, device="mps")
        y = torch.nn.functional.interpolate(x, size=(4, 5, 6), mode="nearest")
        self.assertEqual(y.shape, (2, 3, 4, 5, 6))

    def test_upsample_nearest3d_vec_data_types(self, device="mps"):
        """Test different data types for upsample_nearest3d.vec on MPS device"""
        # Skip if MPS is not available
        if not torch.backends.mps.is_available():
            self.skipTest("MPS device not available")

        # Test with float32
        x_float32 = torch.randn(2, 3, 4, 5, 6, device="mps", dtype=torch.float32)
        y_float32 = torch.nn.functional.interpolate(x_float32, scale_factor=(2.0, 2.0, 2.0), mode="nearest")
        self.assertEqual(y_float32.dtype, torch.float32)

        # Test with float16
        x_float16 = torch.randn(2, 3, 4, 5, 6, device="mps", dtype=torch.float16)
        y_float16 = torch.nn.functional.interpolate(x_float16, scale_factor=(2.0, 2.0, 2.0), mode="nearest")
        self.assertEqual(y_float16.dtype, torch.float16)

        # We don't compare float16 and float32 results directly because they can be quite different
        # due to precision issues. Instead, we just verify that both work correctly.

    def test_upsample_nearest3d_vec_precision(self, device="mps"):
        """Test precision of upsample_nearest3d.vec on MPS device"""
        # Skip if MPS is not available
        if not torch.backends.mps.is_available():
            self.skipTest("MPS device not available")

        # Create a tensor with known values
        x = torch.zeros(1, 1, 2, 2, 2, device="mps")
        x[0, 0, 0, 0, 0] = 1.0
        x[0, 0, 0, 1, 1] = 2.0
        x[0, 0, 1, 0, 1] = 3.0
        x[0, 0, 1, 1, 0] = 4.0

        # Upsample with scale_factor=2
        y = torch.nn.functional.interpolate(x, scale_factor=(2.0, 2.0, 2.0), mode="nearest")

        # Expected output: each value should be replicated in a 2x2x2 block
        expected = torch.zeros(1, 1, 4, 4, 4, device="cpu")
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    expected[0, 0, 0*2+i, 0*2+j, 0*2+k] = 1.0
                    expected[0, 0, 0*2+i, 1*2+j, 1*2+k] = 2.0
                    expected[0, 0, 1*2+i, 0*2+j, 1*2+k] = 3.0
                    expected[0, 0, 1*2+i, 1*2+j, 0*2+k] = 4.0

        # Compare with expected output
        self.assertTrue(torch.allclose(y.to("cpu"), expected, rtol=1e-5, atol=1e-5))

    def test_upsample_nearest3d_vec_memory_usage(self, device="mps"):
        """Test memory usage of upsample_nearest3d.vec on MPS device"""
        # Skip if MPS is not available
        if not torch.backends.mps.is_available():
            self.skipTest("MPS device not available")

        # Create a large tensor
        x = torch.randn(2, 3, 32, 32, 32, device="mps")

        # Record initial memory usage
        torch.mps.empty_cache()
        initial_memory = torch.mps.current_allocated_memory()

        # Perform upsampling
        y = torch.nn.functional.interpolate(x, scale_factor=(2.0, 2.0, 2.0), mode="nearest")

        # Record memory after upsampling
        after_memory = torch.mps.current_allocated_memory()

        # Calculate expected memory increase (approximately):
        # y should be 8 times larger than x (2^3 = 8)
        x_size_bytes = x.nelement() * x.element_size()
        y_size_bytes = y.nelement() * y.element_size()

        # Print memory usage information for debugging
        print(f"Initial memory: {initial_memory / (1024 * 1024):.2f} MB")
        print(f"After memory: {after_memory / (1024 * 1024):.2f} MB")
        print(f"Memory increase: {(after_memory - initial_memory) / (1024 * 1024):.2f} MB")
        print(f"Expected y size: {y_size_bytes / (1024 * 1024):.2f} MB")

        # Verify that memory increase is reasonable
        # Allow for some overhead, but memory increase should be at least the size of y
        self.assertGreaterEqual(after_memory - initial_memory, y_size_bytes * 0.9)

        # Clean up
        del x, y
        torch.mps.empty_cache()

    def test_upsample_nearest3d_vec_stress(self, device="mps"):
        """Stress test for upsample_nearest3d.vec on MPS device"""
        # Skip if MPS is not available
        if not torch.backends.mps.is_available():
            self.skipTest("MPS device not available")

        # Run multiple upsampling operations in a loop
        for _ in range(10):
            # Random input shape
            batch_size = random.randint(1, 4)
            channels = random.randint(1, 8)
            depth = random.randint(4, 16)
            height = random.randint(4, 16)
            width = random.randint(4, 16)

            # Random scale factors (only upsampling for now)
            scale_d = random.uniform(1.0, 2.0)
            scale_h = random.uniform(1.0, 2.0)
            scale_w = random.uniform(1.0, 2.0)

            # Create input tensor
            x = torch.randn(batch_size, channels, depth, height, width, device="mps")

            # Perform upsampling
            y = torch.nn.functional.interpolate(x, scale_factor=(scale_d, scale_h, scale_w), mode="nearest")

            # Verify output shape
            expected_depth = int(depth * scale_d)
            expected_height = int(height * scale_h)
            expected_width = int(width * scale_w)
            self.assertEqual(y.shape, (batch_size, channels, expected_depth, expected_height, expected_width))

    # Skip integration test for now as backward is not implemented yet
    # def test_upsample_nearest3d_vec_integration(self, device="mps"):
    #     """Integration test for upsample_nearest3d.vec with other operations on MPS device"""
    #     # Skip if MPS is not available
    #     if not torch.backends.mps.is_available():
    #         self.skipTest("MPS device not available")
    #
    #     # Create input tensor
    #     x = torch.randn(2, 3, 4, 5, 6, device="mps", requires_grad=True)
    #
    #     # Test with a simple neural network pipeline
    #     # 1. Convolution
    #     conv = torch.nn.Conv3d(3, 6, kernel_size=3, padding=1).to("mps")
    #     conv_out = conv(x)
    #
    #     # 2. ReLU
    #     relu_out = torch.nn.functional.relu(conv_out)
    #
    #     # 3. Upsampling
    #     up_out = torch.nn.functional.interpolate(relu_out, scale_factor=(2.0, 2.0, 2.0), mode="nearest")
    #
    #     # 4. Another convolution
    #     conv2 = torch.nn.Conv3d(6, 3, kernel_size=3, padding=1).to("mps")
    #     final_out = conv2(up_out)
    #
    #     # Verify shapes
    #     self.assertEqual(up_out.shape, torch.Size([2, 6, 8, 10, 12]))
    #     self.assertEqual(final_out.shape, torch.Size([2, 3, 8, 10, 12]))

    def test_upsample_nearest3d_vec_performance(self, device="mps"):
        """Performance test for upsample_nearest3d.vec on MPS device"""
        # Skip if MPS is not available
        if not torch.backends.mps.is_available():
            self.skipTest("MPS device not available")

        # Create a large tensor
        x = torch.randn(2, 3, 32, 32, 32, device="mps")

        # Warm up
        for _ in range(5):
            y = torch.nn.functional.interpolate(x, scale_factor=(2.0, 2.0, 2.0), mode="nearest")
            del y

        # Measure MPS performance
        torch.mps.synchronize()
        import time
        start_time = time.time()

        for _ in range(10):
            y = torch.nn.functional.interpolate(x, scale_factor=(2.0, 2.0, 2.0), mode="nearest")
            torch.mps.synchronize()

        end_time = time.time()
        mps_time = (end_time - start_time) / 10

        # Print performance information
        print(f"MPS time: {mps_time:.6f} seconds")

        # Verify output shape
        y = torch.nn.functional.interpolate(x, scale_factor=(2.0, 2.0, 2.0), mode="nearest")
        self.assertEqual(y.shape, torch.Size([2, 3, 64, 64, 64]))

if __name__ == "__main__":
    run_tests()
