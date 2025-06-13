# Owner(s): ["module: unknown"]
import unittest
import torch
import torch.nn.functional as F
from torch.testing._internal.common_utils import TestCase, run_tests
from hypothesis import given, strategies as st, settings


@unittest.skipIf(
    not torch._C.has_zendnn, "ZenDNN is not available in this PyTorch build"
)
class TestZenDNNLinear(TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        # Check if bfloat16 is supported on the current device
        self.bf16_supported = torch._C._cpu._is_avx512_bf16_supported()

    def _test_zendnn_linear(self, input, weight, bias, atol, rtol):
        # Run reference implementation using torch.nn.functional.linear
        expected = F.linear(input, weight, bias)

        # Run ZenDNN implementation
        if bias is not None:
            result = torch.ops.aten.zendnn_linear(input=input, weight=weight, bias=bias)
        else:
            result = torch.ops.aten.zendnn_linear(input=input, weight=weight)

        # Compare results
        torch.testing.assert_close(result, expected, rtol=rtol, atol=atol)

    @given(
        batch_size=st.integers(1, 32),
        in_features=st.integers(2, 256),
        out_features=st.integers(2, 256),
        has_bias=st.booleans(),
        use_bf16=st.booleans(),
    )
    @settings(deadline=None)
    def test_zendnn_linear_2d_input(
        self, batch_size, in_features, out_features, has_bias, use_bf16
    ):
        if use_bf16 and not self.bf16_supported:
            # Skip test if bf16 is requested but not supported
            self.skipTest("BFloat16 not supported on this device")
        dtype = torch.bfloat16 if use_bf16 else torch.float32

        # Create input tensor
        input = torch.randn(batch_size, in_features, device=self.device, dtype=dtype)

        # Create weight tensor
        weight = torch.randn(out_features, in_features, device=self.device, dtype=dtype)

        # Create bias tensor (optional)
        bias = (
            torch.randn(out_features, device=self.device, dtype=dtype)
            if has_bias
            else None
        )
        rtol = 1e-2 if use_bf16 else 1e-4  # Relax tolerances for BF16
        atol = 1e-2 if use_bf16 else 1e-4
        self._test_zendnn_linear(input, weight, bias, atol, rtol)

    @given(
        batch_size=st.integers(1, 16),
        seq_len=st.integers(1, 32),
        in_features=st.integers(2, 128),
        out_features=st.integers(2, 128),
        has_bias=st.booleans(),
        use_bf16=st.booleans(),
    )
    @settings(deadline=None)
    def test_zendnn_linear_3d_input(
        self, batch_size, seq_len, in_features, out_features, has_bias, use_bf16
    ):
        if use_bf16 and not self.bf16_supported:
            # Skip test if bf16 is requested but not supported
            self.skipTest("BFloat16 not supported on this device")

        dtype = torch.bfloat16 if use_bf16 else torch.float32

        # Create input tensor
        input = torch.randn(
            batch_size, seq_len, in_features, device=self.device, dtype=dtype
        )

        # Create weight tensor
        weight = torch.randn(out_features, in_features, device=self.device, dtype=dtype)

        # Create bias tensor (optional)
        bias = (
            torch.randn(out_features, device=self.device, dtype=dtype)
            if has_bias
            else None
        )

        rtol = 1e-2 if use_bf16 else 1e-4  # Relax tolerances for BF16
        atol = 1e-2 if use_bf16 else 1e-4
        self._test_zendnn_linear(input, weight, bias, atol, rtol)

    @given(
        dims=st.integers(4, 5),
        batch_dim=st.integers(1, 8),
        in_features=st.integers(2, 64),
        out_features=st.integers(2, 64),
        has_bias=st.booleans(),
        use_bf16=st.booleans(),
    )
    @settings(deadline=None)
    def test_zendnn_linear_nd_input(
        self, dims, batch_dim, in_features, out_features, has_bias, use_bf16
    ):
        if use_bf16 and not self.bf16_supported:
            # Skip test if bf16 is requested but not supported
            self.skipTest("BFloat16 not supported on this device")

        dtype = torch.bfloat16 if use_bf16 else torch.float32

        # Create shape with multiple batch dimensions
        shape = [batch_dim] * (dims - 1) + [in_features]

        # Create input tensor
        input = torch.randn(*shape, device=self.device, dtype=dtype)

        # Create weight tensor
        weight = torch.randn(out_features, in_features, device=self.device, dtype=dtype)

        # Create bias tensor (optional)
        bias = (
            torch.randn(out_features, device=self.device, dtype=dtype)
            if has_bias
            else None
        )

        rtol = 1e-2 if use_bf16 else 1e-4  # Relax tolerances for BF16
        atol = 1e-2 if use_bf16 else 1e-4
        self._test_zendnn_linear(input, weight, bias, atol, rtol)

    @given(
        batch_size=st.integers(1, 32),
        in_features=st.integers(2, 128),
        out_features=st.integers(2, 128),
        use_bf16=st.booleans(),
    )
    @settings(deadline=None)
    def test_zendnn_linear_keyword_args(
        self, batch_size, in_features, out_features, use_bf16
    ):
        if use_bf16 and not self.bf16_supported:
            # Skip test if bf16 is requested but not supported
            self.skipTest("BFloat16 not supported on this device")

        dtype = torch.bfloat16 if use_bf16 else torch.float32

        # Create tensors
        input = torch.randn(batch_size, in_features, device=self.device, dtype=dtype)
        weight = torch.randn(out_features, in_features, device=self.device, dtype=dtype)
        bias = torch.randn(out_features, device=self.device, dtype=dtype)

        # Run with positional arguments
        result1 = torch.ops.aten.zendnn_linear(input, weight, bias)

        # Run with keyword arguments
        result2 = torch.ops.aten.zendnn_linear(input=input, weight=weight, bias=bias)

        # Compare results
        rtol = 1e-2 if use_bf16 else 1e-4  # Relax tolerances for BF16
        atol = 1e-2 if use_bf16 else 1e-4
        torch.testing.assert_close(result1, result2, rtol=rtol, atol=atol)

    def test_zendnn_linear_exception_weight_dim(self):
        # Test invalid weight dimension
        input = torch.randn(10, 20)
        weight = torch.randn(30, 20, 5)  # Should be 2D

        with self.assertRaises(RuntimeError):
            torch.ops.aten.zendnn_linear(input, weight)

    def test_zendnn_linear_exception_bias_dim(self):
        # Test invalid bias dimension
        input = torch.randn(10, 20)
        weight = torch.randn(30, 20)
        bias = torch.randn(30, 5)  # Should be 1D

        with self.assertRaises(RuntimeError):
            torch.ops.aten.zendnn_linear(input, weight, bias)

    def test_zendnn_linear_exception_feature_mismatch(self):
        # Test mismatch in feature dimensions
        input = torch.randn(10, 20)
        weight = torch.randn(30, 25)  # Should be (30, 20)

        with self.assertRaises(RuntimeError):
            torch.ops.aten.zendnn_linear(input, weight)

    def test_zendnn_linear_exception_bias_size(self):
        # Test mismatch in bias size
        input = torch.randn(10, 20)
        weight = torch.randn(30, 20)
        bias = torch.randn(35)  # Should be size 30

        with self.assertRaises(RuntimeError):
            torch.ops.aten.zendnn_linear(input, weight, bias)

    def test_zendnn_linear_dtype_mismatch(self):
        # Test dtype mismatch between input tensors
        input = torch.randn(10, 20, dtype=torch.float32)
        weight = torch.randn(30, 20, dtype=torch.float64)  # Different dtype

        with self.assertRaises(RuntimeError):
            torch.ops.aten.zendnn_linear(input, weight)

    def test_zendnn_linear_bf16(self):
        # Skip if BF16 is not supported
        if not self.bf16_supported:
            self.skipTest("BFloat16 not supported on this device")

        # Create BF16 tensors
        input = torch.randn(10, 20, dtype=torch.bfloat16)
        weight = torch.randn(30, 20, dtype=torch.bfloat16)
        bias = torch.randn(30, dtype=torch.bfloat16)

        # Verify both implementations produce similar results
        expected = F.linear(input, weight, bias)
        result = torch.ops.aten.zendnn_linear(input, weight, bias)

        torch.testing.assert_close(result, expected, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    run_tests()
