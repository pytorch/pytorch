# Owner(s): ["module: unknown"]
import unittest

from hypothesis import given, settings, strategies as st

import torch
import torch.nn.functional as F
from torch.testing._internal.common_utils import run_tests, TestCase


@unittest.skipIf(
    not torch._C.has_zendnn, "ZenDNN is not available in this PyTorch build"
)
class TestZenDNNLinear(TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        # Check if bfloat16 is supported on the current device
        self.bf16_supported = torch._C._cpu._is_avx512_bf16_supported()

    def _test_zendnn_linear_with_weight_prepack(self, input, weight, bias, atol, rtol):
        # Run reference implementation using torch.nn.functional.linear
        expected = F.linear(input, weight, bias)

        # Prepack the weight tensor
        weight_prepacked = torch.ops.aten.zendnn_weight_prepack_for_linear(weight)

        # Run ZenDNN implementation
        if bias is not None:
            result = torch.ops.aten.zendnn_linear(
                input=input,
                weight=weight_prepacked,
                bias=bias,
                is_weight_prepacked=True,
            )
        else:
            result = torch.ops.aten.zendnn_linear(
                input=input, weight=weight_prepacked, is_weight_prepacked=True
            )

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
    def test_zendnn_linear_with_weight_prepack_n_2d_input(
        self,
        batch_size,
        in_features,
        out_features,
        has_bias,
        use_bf16,
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
        rtol = 1e-2 if use_bf16 else 1e-3  # Relax tolerances for weight prepacking
        atol = 1e-2 if use_bf16 else 1e-3
        self._test_zendnn_linear_with_weight_prepack(input, weight, bias, atol, rtol)

    @given(
        batch_size=st.integers(1, 16),
        seq_len=st.integers(1, 32),
        in_features=st.integers(2, 128),
        out_features=st.integers(2, 128),
        has_bias=st.booleans(),
        use_bf16=st.booleans(),
    )
    @settings(deadline=None)
    def test_zendnn_linear_with_weight_prepack_n_3d_input(
        self,
        batch_size,
        seq_len,
        in_features,
        out_features,
        has_bias,
        use_bf16,
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

        rtol = 1e-2 if use_bf16 else 1e-3  # Relax tolerances for weight prepacking
        atol = 1e-2 if use_bf16 else 1e-3
        self._test_zendnn_linear_with_weight_prepack(input, weight, bias, atol, rtol)

    @given(
        dims=st.integers(4, 5),
        batch_dim=st.integers(1, 8),
        in_features=st.integers(2, 64),
        out_features=st.integers(2, 64),
        has_bias=st.booleans(),
        use_bf16=st.booleans(),
    )
    @settings(deadline=None)
    def test_zendnn_linear_with_weight_prepack_n_nd_input(
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

        rtol = 1e-2 if use_bf16 else 1e-3  # Relax tolerances for weight prepacking
        atol = 1e-2 if use_bf16 else 1e-3
        self._test_zendnn_linear_with_weight_prepack(input, weight, bias, atol, rtol)


if __name__ == "__main__":
    run_tests()
