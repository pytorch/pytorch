# Owner(s): ["module: unknown"]
import os
import unittest
from typing import Optional

from hypothesis import given, settings, strategies as st

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch._inductor.fx_passes.zendnn_utils import counters
from torch.testing._internal.common_utils import run_tests, TestCase


class CustomLinearModel(nn.Module):
    def __init__(self, weight: Tensor, bias: Optional[Tensor]):
        super().__init__()
        self.linear = nn.Linear(weight.size(1), weight.size(0), bias is not None)
        self.linear.weight.data = weight.clone()
        if bias is not None:
            self.linear.bias.data = bias.clone()

    def forward(self, input: Tensor) -> Tensor:
        return self.linear(input)


@unittest.skipIf(
    not (
        torch._C.has_zendnn  # type: ignore[attr-defined]
        and (torch._C._cpu._is_amd_cpu() or torch._inductor.config.enable_zendnn)
    ),
    "ZenDNN is not available in this PyTorch build",
)
class TestZenDNNLinear(TestCase):
    def setUp(self) -> None:
        self.device = torch.device("cpu")
        # Check if bfloat16 is supported on the current device
        self.bf16_supported = torch._C._cpu._is_avx512_bf16_supported()

    def _test_zendnn_linear(
        self,
        input: Tensor,
        weight: Tensor,
        bias: Optional[Tensor],
        atol: float,
        rtol: float,
        weight_prepacked: bool,
    ) -> None:
        # Run reference implementation using torch.nn.functional.linear
        expected = F.linear(input, weight, bias)
        if weight_prepacked:
            # Prepack the weight tensor for ZenDNN
            weight = torch.ops.aten.zendnn_weight_prepack_for_linear(weight)

        # Run ZenDNN implementation
        if bias is not None:
            result = torch.ops.aten.zendnn_linear(
                input=input,
                weight=weight,
                bias=bias,
                is_weight_prepacked=weight_prepacked,
            )
        else:
            result = torch.ops.aten.zendnn_linear(
                input=input, weight=weight, is_weight_prepacked=weight_prepacked
            )

        # Compare results
        torch.testing.assert_close(
            result, expected, rtol=rtol, atol=atol, equal_nan=True
        )

    @given(  # type: ignore[misc]
        batch_size=st.integers(1, 32),
        in_features=st.integers(2, 256),
        out_features=st.integers(2, 256),
        has_bias=st.booleans(),
        use_bf16=st.booleans(),
        weight_prepacked=st.booleans(),
    )
    @settings(deadline=None)  # type: ignore[misc]
    def test_zendnn_linear_2d_input(
        self,
        batch_size: int,
        in_features: int,
        out_features: int,
        has_bias: bool,
        use_bf16: bool,
        weight_prepacked: bool,
    ) -> None:
        if use_bf16 and not self.bf16_supported:
            # Skip test if bf16 is requested but not supported
            self.skipTest("BFloat16 not supported on this device")
        dtype = torch.bfloat16 if use_bf16 else torch.float32

        # Create input tensor
        input = torch.randn(
            batch_size,
            in_features,
            dtype=dtype,
            requires_grad=False,
            device=self.device,
        )

        # Create weight tensor
        weight = torch.randn(
            out_features,
            in_features,
            dtype=dtype,
            requires_grad=False,
            device=self.device,
        )

        # Create bias tensor (optional)
        bias = (
            torch.randn(
                out_features, dtype=dtype, requires_grad=False, device=self.device
            )
            if has_bias
            else None
        )
        rtol = 1e-2 if use_bf16 else 1e-4  # Relax tolerances for BF16
        atol = 1e-2 if use_bf16 else 1e-4
        self._test_zendnn_linear(input, weight, bias, atol, rtol, weight_prepacked)

    @given(  # type: ignore[misc]
        batch_size=st.integers(1, 16),
        seq_len=st.integers(1, 32),
        in_features=st.integers(2, 128),
        out_features=st.integers(2, 128),
        has_bias=st.booleans(),
        use_bf16=st.booleans(),
        weight_prepacked=st.booleans(),
    )
    @settings(deadline=None)  # type: ignore[misc]
    def test_zendnn_linear_3d_input(
        self,
        batch_size: int,
        seq_len: int,
        in_features: int,
        out_features: int,
        has_bias: bool,
        use_bf16: bool,
        weight_prepacked: bool,
    ) -> None:
        if use_bf16 and not self.bf16_supported:
            # Skip test if bf16 is requested but not supported
            self.skipTest("BFloat16 not supported on this device")

        dtype = torch.bfloat16 if use_bf16 else torch.float32

        # Create input tensor
        input = torch.randn(
            batch_size,
            seq_len,
            in_features,
            dtype=dtype,
            requires_grad=False,
            device=self.device,
        )

        # Create weight tensor
        weight = torch.randn(
            out_features,
            in_features,
            dtype=dtype,
            requires_grad=False,
            device=self.device,
        )

        # Create bias tensor (optional)
        bias = (
            torch.randn(
                out_features, dtype=dtype, requires_grad=False, device=self.device
            )
            if has_bias
            else None
        )

        rtol = 1e-2 if use_bf16 else 1e-4  # Relax tolerances for BF16
        atol = 1e-2 if use_bf16 else 1e-4
        self._test_zendnn_linear(input, weight, bias, atol, rtol, weight_prepacked)

    @given(  # type: ignore[misc]
        dims=st.integers(4, 5),
        batch_dim=st.integers(1, 8),
        in_features=st.integers(2, 64),
        out_features=st.integers(2, 64),
        has_bias=st.booleans(),
        use_bf16=st.booleans(),
        weight_prepacked=st.booleans(),
    )
    @settings(deadline=None)  # type: ignore[misc]
    def test_zendnn_linear_nd_input(
        self,
        dims: int,
        batch_dim: int,
        in_features: int,
        out_features: int,
        has_bias: bool,
        use_bf16: bool,
        weight_prepacked: bool,
    ) -> None:
        if use_bf16 and not self.bf16_supported:
            # Skip test if bf16 is requested but not supported
            self.skipTest("BFloat16 not supported on this device")

        dtype = torch.bfloat16 if use_bf16 else torch.float32

        # Create shape with multiple batch dimensions
        shape = [batch_dim] * (dims - 1) + [in_features]

        # Create input tensor
        input = torch.randn(
            *shape, dtype=dtype, requires_grad=False, device=self.device
        )

        # Create weight tensor
        weight = torch.randn(
            out_features,
            in_features,
            dtype=dtype,
            requires_grad=False,
            device=self.device,
        )

        # Create bias tensor (optional)
        bias = (
            torch.randn(
                out_features, dtype=dtype, requires_grad=False, device=self.device
            )
            if has_bias
            else None
        )

        rtol = 1e-2 if use_bf16 else 1e-4  # Relax tolerances for BF16
        atol = 1e-2 if use_bf16 else 1e-4
        self._test_zendnn_linear(input, weight, bias, atol, rtol, weight_prepacked)

    @given(  # type: ignore[misc]
        batch_size=st.integers(1, 32),
        in_features=st.integers(2, 128),
        out_features=st.integers(2, 128),
        use_bf16=st.booleans(),
    )
    @settings(deadline=None)  # type: ignore[misc]
    def test_zendnn_linear_keyword_args(
        self, batch_size: int, in_features: int, out_features: int, use_bf16: bool
    ) -> None:
        if use_bf16 and not self.bf16_supported:
            # Skip test if bf16 is requested but not supported
            self.skipTest("BFloat16 not supported on this device")

        dtype = torch.bfloat16 if use_bf16 else torch.float32

        # Create tensors
        input = torch.randn(
            batch_size,
            in_features,
            dtype=dtype,
            requires_grad=False,
            device=self.device,
        )
        weight = torch.randn(
            out_features,
            in_features,
            dtype=dtype,
            requires_grad=False,
            device=self.device,
        )
        bias = torch.randn(
            out_features, dtype=dtype, requires_grad=False, device=self.device
        )

        # Run with positional arguments
        result1 = torch.ops.aten.zendnn_linear(input, weight, bias)

        # Run with keyword arguments
        result2 = torch.ops.aten.zendnn_linear(input=input, weight=weight, bias=bias)

        # Compare results
        rtol = 1e-2 if use_bf16 else 1e-4  # Relax tolerances for BF16
        atol = 1e-2 if use_bf16 else 1e-4
        torch.testing.assert_close(
            result1, result2, rtol=rtol, atol=atol, equal_nan=True
        )

    def test_zendnn_linear_exception_weight_dim(self) -> None:
        # Test invalid weight dimension
        input = torch.randn(10, 20, requires_grad=False, device=self.device)
        weight = torch.randn(
            30, 20, 5, requires_grad=False, device=self.device
        )  # Should be 2D

        with self.assertRaises(RuntimeError):
            torch.ops.aten.zendnn_linear(input, weight)

    def test_zendnn_linear_exception_bias_dim(self) -> None:
        # Test invalid bias dimension
        input = torch.randn(10, 20, requires_grad=False, device=self.device)
        weight = torch.randn(30, 20, requires_grad=False, device=self.device)
        bias = torch.randn(
            30, 5, requires_grad=False, device=self.device
        )  # Should be 1D

        with self.assertRaises(RuntimeError):
            torch.ops.aten.zendnn_linear(input, weight, bias)

    def test_zendnn_linear_exception_feature_mismatch(self) -> None:
        # Test mismatch in feature dimensions
        input = torch.randn(10, 20, requires_grad=False, device=self.device)
        weight = torch.randn(
            30, 25, requires_grad=False, device=self.device
        )  # Should be (30, 20)

        with self.assertRaises(RuntimeError):
            torch.ops.aten.zendnn_linear(input, weight)

    def test_zendnn_linear_exception_bias_size(self) -> None:
        # Test mismatch in bias size
        input = torch.randn(10, 20, requires_grad=False, device=self.device)
        weight = torch.randn(30, 20, requires_grad=False, device=self.device)
        bias = torch.randn(
            35, requires_grad=False, device=self.device
        )  # Should be size 30

        with self.assertRaises(RuntimeError):
            torch.ops.aten.zendnn_linear(input, weight, bias)

    def test_zendnn_linear_dtype_mismatch(self) -> None:
        # Test dtype mismatch between input tensors
        input = torch.randn(
            10, 20, dtype=torch.float32, requires_grad=False, device=self.device
        )
        weight = torch.randn(
            30, 20, dtype=torch.float64, requires_grad=False, device=self.device
        )  # Different dtype

        with self.assertRaises(RuntimeError):
            torch.ops.aten.zendnn_linear(input, weight)

    def test_zendnn_linear_bf16(self) -> None:
        # Skip if BF16 is not supported
        if not self.bf16_supported:
            self.skipTest("BFloat16 not supported on this device")

        # Create BF16 tensors
        input = torch.randn(
            10, 20, dtype=torch.bfloat16, requires_grad=False, device=self.device
        )
        weight = torch.randn(
            30, 20, dtype=torch.bfloat16, requires_grad=False, device=self.device
        )
        bias = torch.randn(
            30, dtype=torch.bfloat16, requires_grad=False, device=self.device
        )

        # Verify both implementations produce similar results
        expected = F.linear(input, weight, bias)
        result = torch.ops.aten.zendnn_linear(input, weight, bias)

        torch.testing.assert_close(
            result, expected, rtol=1e-2, atol=1e-2, equal_nan=True
        )


@unittest.skipIf(
    not (
        torch._C.has_zendnn  # type: ignore[attr-defined]
        and (torch._C._cpu._is_amd_cpu() or torch._inductor.config.enable_zendnn)
    ),
    "ZenDNN is not available in this PyTorch build",
)
class TestCompiledLinear(TestCase):
    def setUp(self) -> None:
        self.device = torch.device("cpu")
        self.previous_freezing = torch._inductor.config.freezing
        torch._inductor.config.freezing = True
        self.previous_weight_prepack = torch._inductor.config.cpp.weight_prepack
        self.previous_enable_zendnn = torch._inductor.config.enable_zendnn
        # Check if bfloat16 is supported on the current device
        self.bf16_supported = torch._C._cpu._is_avx512_bf16_supported()

    def tearDown(self) -> None:
        torch._inductor.config.cpp.weight_prepack = self.previous_weight_prepack
        torch._inductor.config.freezing = self.previous_freezing
        torch._inductor.config.enable_zendnn = self.previous_enable_zendnn

    def _test_compiled_linear(
        self,
        input: Tensor,
        weight: Tensor,
        bias: Optional[Tensor],
        atol: float,
        rtol: float,
        weight_prepacked: bool,
        dynamic_shape: bool,
    ) -> None:
        # Reset counters
        counters.clear()
        # Run reference implementation using torch.nn.functional.linear
        torch._dynamo.reset()
        torch._inductor.config.enable_zendnn = False
        linear_model = CustomLinearModel(weight, bias).to(input.dtype)
        compiled_model = torch.compile(
            linear_model, backend="inductor", dynamic=dynamic_shape
        )
        with torch.no_grad():
            expected = compiled_model(input)
        torch._inductor.config.enable_zendnn = True

        # Run ZenDNN implementation
        torch._dynamo.reset()
        torch._inductor.config.cpp.weight_prepack = weight_prepacked
        zendnn_linear_model = CustomLinearModel(weight, bias).to(input.dtype)
        compiled_model = torch.compile(
            zendnn_linear_model, backend="inductor", dynamic=dynamic_shape
        )

        with torch.no_grad():
            expected = linear_model(input)
            self.assertEqual(counters["zendnn"]["zendnn_linear"], 0)
            self.assertEqual(counters["zendnn"]["zendnn_weight_prepack_for_linear"], 0)
            result = compiled_model(input)
            self.assertEqual(counters["zendnn"]["zendnn_linear"], 1)
            self.assertEqual(
                counters["zendnn"]["zendnn_weight_prepack_for_linear"],
                1 if weight_prepacked else 0,
            )
        # Compare results
        torch.testing.assert_close(
            result, expected, rtol=rtol, atol=atol, equal_nan=True
        )

    @given(  # type: ignore[misc]
        batch_size=st.integers(1, 32),
        in_features=st.integers(2, 256),
        out_features=st.integers(2, 256),
        has_bias=st.booleans(),
        use_bf16=st.booleans(),
        weight_prepacked=st.booleans(),
        dynamic_shape=st.booleans(),
    )
    @settings(deadline=None)  # type: ignore[misc]
    def test_compiled_linear_2d_input(
        self,
        batch_size: int,
        in_features: int,
        out_features: int,
        has_bias: bool,
        use_bf16: bool,
        weight_prepacked: bool,
        dynamic_shape: bool,
    ) -> None:
        if use_bf16 and not self.bf16_supported:
            # Skip test if bf16 is requested but not supported
            self.skipTest("BFloat16 not supported on this device")
        dtype = torch.bfloat16 if use_bf16 else torch.float32

        # Create input tensor
        input = torch.randn(
            batch_size,  # m
            in_features,  # n
            dtype=dtype,
            requires_grad=False,
            device=self.device,
        )

        # Create weight tensor
        weight = torch.randn(
            out_features,  # k
            in_features,  # n
            dtype=dtype,
            requires_grad=False,
            device=self.device,
        )

        # Create bias tensor (optional)
        bias = (
            torch.randn(
                out_features, dtype=dtype, requires_grad=False, device=self.device
            )
            if has_bias
            else None
        )
        rtol = 1e-2 if use_bf16 else 1e-4  # Relax tolerances for BF16
        atol = 1e-2 if use_bf16 else 1e-4
        self._test_compiled_linear(
            input, weight, bias, atol, rtol, weight_prepacked, dynamic_shape
        )

    @given(  # type: ignore[misc]
        batch_size=st.integers(1, 16),
        seq_len=st.integers(1, 32),
        in_features=st.integers(2, 128),
        out_features=st.integers(2, 128),
        has_bias=st.booleans(),
        use_bf16=st.booleans(),
        weight_prepacked=st.booleans(),
        dynamic_shape=st.booleans(),
    )
    @settings(deadline=None)  # type: ignore[misc]
    def test_compiled_linear_3d_input(
        self,
        batch_size: int,
        seq_len: int,
        in_features: int,
        out_features: int,
        has_bias: bool,
        use_bf16: bool,
        weight_prepacked: bool,
        dynamic_shape: bool,
    ) -> None:
        weight_prepacked = True
        if use_bf16 and not self.bf16_supported:
            # Skip test if bf16 is requested but not supported
            self.skipTest("BFloat16 not supported on this device")

        dtype = torch.bfloat16 if use_bf16 else torch.float32

        # Create input tensor
        input = torch.randn(
            batch_size,
            seq_len,
            in_features,
            dtype=dtype,
            requires_grad=False,
            device=self.device,
        )

        # Create weight tensor
        weight = torch.randn(
            out_features,
            in_features,
            dtype=dtype,
            requires_grad=False,
            device=self.device,
        )

        # Create bias tensor (optional)
        bias = (
            torch.randn(
                out_features, dtype=dtype, requires_grad=False, device=self.device
            )
            if has_bias
            else None
        )

        rtol = 1e-2 if use_bf16 else 1e-4  # Relax tolerances for BF16
        atol = 1e-2 if use_bf16 else 1e-4
        self._test_compiled_linear(
            input, weight, bias, atol, rtol, weight_prepacked, dynamic_shape
        )

    @given(  # type: ignore[misc]
        dims=st.integers(4, 5),
        batch_dim=st.integers(1, 8),
        in_features=st.integers(2, 64),
        out_features=st.integers(2, 64),
        has_bias=st.booleans(),
        use_bf16=st.booleans(),
        weight_prepacked=st.booleans(),
        dynamic_shape=st.booleans(),
    )
    @settings(deadline=None)  # type: ignore[misc]
    def test_compiled_linear_nd_input(
        self,
        dims: int,
        batch_dim: int,
        in_features: int,
        out_features: int,
        has_bias: bool,
        use_bf16: bool,
        weight_prepacked: bool,
        dynamic_shape: bool,
    ) -> None:
        if use_bf16 and not self.bf16_supported:
            # Skip test if bf16 is requested but not supported
            self.skipTest("BFloat16 not supported on this device")

        dtype = torch.bfloat16 if use_bf16 else torch.float32

        # Create shape with multiple batch dimensions
        shape = [batch_dim] * (dims - 1) + [in_features]

        # Create input tensor
        input = torch.randn(
            *shape, dtype=dtype, requires_grad=False, device=self.device
        )

        # Create weight tensor
        weight = torch.randn(
            out_features,
            in_features,
            dtype=dtype,
            requires_grad=False,
            device=self.device,
        )

        # Create bias tensor (optional)
        bias = (
            torch.randn(
                out_features, dtype=dtype, requires_grad=False, device=self.device
            )
            if has_bias
            else None
        )

        rtol = 1e-2 if use_bf16 else 1e-4  # Relax tolerances for BF16
        atol = 1e-2 if use_bf16 else 1e-4
        self._test_compiled_linear(
            input, weight, bias, atol, rtol, weight_prepacked, dynamic_shape
        )


@unittest.skipIf(
    not (
        torch._C.has_zendnn  # type: ignore[attr-defined]
        and (torch._C._cpu._is_amd_cpu() or torch._inductor.config.enable_zendnn)
    ),
    "ZenDNN is not available in this PyTorch build",
)
class TestExportedLinear(TestCase):
    def setUp(self) -> None:
        self.device = torch.device("cpu")
        import tempfile

        self.tmpdir = tempfile.mkdtemp()
        # Check if bfloat16 is supported on the current device
        self.previous_weight_prepack = torch._inductor.config.cpp.weight_prepack
        self.previous_enable_zendnn = torch._inductor.config.enable_zendnn
        self.bf16_supported = torch._C._cpu._is_avx512_bf16_supported()

    def tearDown(self) -> None:
        import shutil

        torch._inductor.config.cpp.weight_prepack = self.previous_weight_prepack
        torch._inductor.config.enable_zendnn = self.previous_enable_zendnn
        shutil.rmtree(self.tmpdir)

    def _test_exported_linear(
        self,
        input: Tensor,
        weight: Tensor,
        bias: Optional[Tensor],
        atol: float,
        rtol: float,
        weight_prepacked: bool,
        dynamic_shape: bool,
    ) -> None:
        # Reset counters
        counters.clear()

        # Run reference implementation using torch.nn.functional.linear#
        torch._dynamo.reset()
        torch._inductor.config.enable_zendnn = False
        linear_model = CustomLinearModel(weight, bias).to(input.dtype)
        exported = torch.export.export(
            linear_model, (input,)
        )  # Dynamic is not currently supported
        output_path = torch._inductor.aoti_compile_and_package(
            exported,
            package_path=os.path.join(self.tmpdir, "model.pt2"),
        )
        exported_model = torch._inductor.aoti_load_package(output_path)
        with torch.no_grad():
            expected = exported_model(input)
        torch._inductor.config.enable_zendnn = True

        # Run ZenDNN implementation
        torch._dynamo.reset()
        torch._inductor.config.cpp.weight_prepack = weight_prepacked
        zendnn_linear_model = CustomLinearModel(weight, bias).to(input.dtype)
        self.assertEqual(counters["zendnn"]["zendnn_linear"], 0)
        self.assertEqual(counters["zendnn"]["zendnn_weight_prepack_for_linear"], 0)
        # Create exported version
        exported = torch.export.export(
            zendnn_linear_model, (input,)
        )  # Dynamic is not currently supported
        output_path = torch._inductor.aoti_compile_and_package(
            exported,
            package_path=os.path.join(self.tmpdir, "model_zendnn.pt2"),
        )
        self.assertEqual(counters["zendnn"]["zendnn_linear"], 1)
        self.assertEqual(
            counters["zendnn"]["zendnn_weight_prepack_for_linear"],
            1 if weight_prepacked else 0,
        )
        counters.clear()
        exported_model = torch._inductor.aoti_load_package(output_path)

        with torch.no_grad():
            result = exported_model(input)
        # Compare results
        self.assertEqual(counters["zendnn"]["zendnn_linear"], 0)
        self.assertEqual(counters["zendnn"]["zendnn_weight_prepack_for_linear"], 0)
        torch.testing.assert_close(
            result, expected, rtol=rtol, atol=atol, equal_nan=True
        )

    @given(  # type: ignore[misc]
        batch_size=st.integers(1, 32),
        in_features=st.integers(2, 256),
        out_features=st.integers(2, 256),
        has_bias=st.booleans(),
        use_bf16=st.booleans(),
        weight_prepacked=st.booleans(),
        dynamic_shape=st.booleans(),
    )
    @settings(deadline=None)  # type: ignore[misc]
    def test_exported_linear_2d_input(
        self,
        batch_size: int,
        in_features: int,
        out_features: int,
        has_bias: bool,
        use_bf16: bool,
        weight_prepacked: bool,
        dynamic_shape: bool,
    ) -> None:
        if use_bf16 and not self.bf16_supported:
            # Skip test if bf16 is requested but not supported
            self.skipTest("BFloat16 not supported on this device")
        dtype = torch.bfloat16 if use_bf16 else torch.float32

        # Create input tensor
        input = torch.randn(
            batch_size,
            in_features,
            dtype=dtype,
            requires_grad=False,
            device=self.device,
        )

        # Create weight tensor
        weight = torch.randn(
            out_features,
            in_features,
            dtype=dtype,
            requires_grad=False,
            device=self.device,
        )

        # Create bias tensor (optional)
        bias = (
            torch.randn(
                out_features, dtype=dtype, requires_grad=False, device=self.device
            )
            if has_bias
            else None
        )
        rtol = 1e-2 if use_bf16 else 1e-4  # Relax tolerances for BF16
        atol = 1e-2 if use_bf16 else 1e-4
        self._test_exported_linear(
            input, weight, bias, atol, rtol, weight_prepacked, dynamic_shape
        )

    @given(  # type: ignore[misc]
        batch_size=st.integers(1, 16),
        seq_len=st.integers(1, 32),
        in_features=st.integers(2, 128),
        out_features=st.integers(2, 128),
        has_bias=st.booleans(),
        use_bf16=st.booleans(),
        weight_prepacked=st.booleans(),
        dynamic_shape=st.booleans(),
    )
    @settings(deadline=None)  # type: ignore[misc]
    def test_exported_linear_3d_input(
        self,
        batch_size: int,
        seq_len: int,
        in_features: int,
        out_features: int,
        has_bias: bool,
        use_bf16: bool,
        weight_prepacked: bool,
        dynamic_shape: bool,
    ) -> None:
        if use_bf16 and not self.bf16_supported:
            # Skip test if bf16 is requested but not supported
            self.skipTest("BFloat16 not supported on this device")

        dtype = torch.bfloat16 if use_bf16 else torch.float32

        # Create input tensor
        input = torch.randn(
            batch_size,
            seq_len,
            in_features,
            dtype=dtype,
            requires_grad=False,
            device=self.device,
        )

        # Create weight tensor
        weight = torch.randn(
            out_features,
            in_features,
            dtype=dtype,
            requires_grad=False,
            device=self.device,
        )

        # Create bias tensor (optional)
        bias = (
            torch.randn(
                out_features, dtype=dtype, requires_grad=False, device=self.device
            )
            if has_bias
            else None
        )

        rtol = 1e-2 if use_bf16 else 1e-4  # Relax tolerances for BF16
        atol = 1e-2 if use_bf16 else 1e-4
        self._test_exported_linear(
            input, weight, bias, atol, rtol, weight_prepacked, dynamic_shape
        )

    @given(  # type: ignore[misc]
        dims=st.integers(4, 5),
        batch_dim=st.integers(1, 8),
        in_features=st.integers(2, 64),
        out_features=st.integers(2, 64),
        has_bias=st.booleans(),
        use_bf16=st.booleans(),
        weight_prepacked=st.booleans(),
        dynamic_shape=st.booleans(),
    )
    @settings(deadline=None)  # type: ignore[misc]
    def test_exported_linear_nd_input(
        self,
        dims: int,
        batch_dim: int,
        in_features: int,
        out_features: int,
        has_bias: bool,
        use_bf16: bool,
        weight_prepacked: bool,
        dynamic_shape: bool,
    ) -> None:
        if use_bf16 and not self.bf16_supported:
            # Skip test if bf16 is requested but not supported
            self.skipTest("BFloat16 not supported on this device")

        dtype = torch.bfloat16 if use_bf16 else torch.float32

        # Create shape with multiple batch dimensions
        shape = [batch_dim] * (dims - 1) + [in_features]

        # Create input tensor
        input = torch.randn(
            *shape, dtype=dtype, requires_grad=False, device=self.device
        )

        # Create weight tensor
        weight = torch.randn(
            out_features,
            in_features,
            dtype=dtype,
            requires_grad=False,
            device=self.device,
        )

        # Create bias tensor (optional)
        bias = (
            torch.randn(
                out_features, dtype=dtype, requires_grad=False, device=self.device
            )
            if has_bias
            else None
        )

        rtol = 1e-2 if use_bf16 else 1e-4  # Relax tolerances for BF16
        atol = 1e-2 if use_bf16 else 1e-4
        self._test_exported_linear(
            input, weight, bias, atol, rtol, weight_prepacked, dynamic_shape
        )


if __name__ == "__main__":
    run_tests()
