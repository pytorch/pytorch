# Owner(s): ["module: unknown"]
import copy
import os
import unittest

from hypothesis import given, settings, strategies as st

import torch
import torch._inductor.config
import torch.nn as nn
import torch.nn.functional as F
from torch._inductor.fx_passes.zendnn_utils import counters
from torch.testing._internal.common_utils import run_tests, TestCase


FUSION_TO_ACTIVATION_FUNC_MAP = {
    "relu": F.relu,
    "silu": F.silu,
    "sigmoid": torch.sigmoid,
    "tanh": torch.tanh,
    # "gelu_erf": F.gelu, # TODO: Gelu_ERF tests currently fails due to issue with blis.
    "gelu_tanh": lambda x: F.gelu(x, approximate="tanh"),
    "add": None,  # For 'add', we'll handle it separately
    "mul": None,  # For 'mul', we'll handle it separately
}


class CustomLinearFusionModel(nn.Module):
    def __init__(self, weight, bias, fusion_op, second_input=None):
        super().__init__()
        self.linear = nn.Linear(weight.size(1), weight.size(0), bias is not None)
        self.linear.weight.data = weight.clone()
        if bias is not None:
            self.linear.bias.data = bias.clone()
        self.fusion_op = fusion_op
        self.second_input = second_input

    def forward(self, x):
        x = self.linear(x)

        if self.fusion_op == "relu":
            return F.relu(x)
        elif self.fusion_op == "silu":
            return F.silu(x)
        elif self.fusion_op == "sigmoid":
            return torch.sigmoid(x)
        elif self.fusion_op == "tanh":
            return torch.tanh(x)
        elif self.fusion_op == "gelu_erf":
            return F.gelu(x)
        elif self.fusion_op == "gelu_tanh":
            return F.gelu(x, approximate="tanh")
        elif self.fusion_op == "add" and self.second_input is not None:
            return x + self.second_input
        elif self.fusion_op == "mul" and self.second_input is not None:
            return x * self.second_input
        else:
            return x


@unittest.skipIf(
    not (
        torch._C.has_zendnn
        and (torch._C._cpu._is_amd_cpu() or torch._inductor.config.enable_zendnn)
    ),
    "ZenDNN is not available in this PyTorch build",
)
class TestZenDNNLinearFusion(TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        # Check if bfloat16 is supported on the current device
        self.bf16_supported = torch._C._cpu._is_avx512_bf16_supported()
        self.previous_enable_zendnn = torch._inductor.config.enable_zendnn
        self.fusion_ops = list(FUSION_TO_ACTIVATION_FUNC_MAP.keys())

    def tearDown(self):
        torch._inductor.config.enable_zendnn = self.previous_enable_zendnn

    def _get_activation_fn(self, fusion_op):
        """Return the corresponding PyTorch activation function for the fusion op"""
        return FUSION_TO_ACTIVATION_FUNC_MAP[fusion_op]

    def _test_zendnn_linear_fusion(
        self,
        input,
        weight,
        bias,
        fusion_op,
        second_input=None,
        atol=1e-4,
        rtol=1e-4,
        weight_prepacked=False,
    ):
        # Reset counters
        counters.clear()
        # Run reference implementation
        torch._dynamo.reset()
        torch._inductor.config.enable_zendnn = False
        linear_model = CustomLinearFusionModel(
            weight, bias, fusion_op, second_input
        ).to(input.dtype)
        compiled_model = torch.compile(linear_model, backend="inductor")
        with torch.no_grad():
            expected = compiled_model(input)
        torch._inductor.config.enable_zendnn = True
        if weight_prepacked:
            weight = torch.ops.aten.zendnn_weight_prepack_for_linear(weight)

        # Run ZenDNN implementation
        if fusion_op in ["add", "mul"] and second_input is not None:
            result = torch.ops.aten.zendnn_linear_unary_binary(
                input=input,
                weight=weight,
                binary_input=second_input,
                bias=bias,
                is_weight_prepacked=weight_prepacked,
                post_op_1="",
                post_op_2=fusion_op,
                zendnn_op_name="linear_" + fusion_op,
            )
        else:
            result = torch.ops.aten.zendnn_linear(
                input=input,
                weight=weight,
                bias=bias,
                is_weight_prepacked=weight_prepacked,
                post_op=fusion_op,
                zendnn_op_name="linear_" + fusion_op,
            )

        # Compare results
        torch.testing.assert_close(
            result, expected, rtol=rtol, atol=atol, equal_nan=True
        )

    @given(
        batch_size=st.integers(1, 32),
        in_features=st.integers(2, 256),
        out_features=st.integers(2, 256),
        has_bias=st.booleans(),
        use_bf16=st.booleans(),
        fusion_op_idx=st.integers(
            0, len(FUSION_TO_ACTIVATION_FUNC_MAP) - 1
        ),  # Index into self.fusion_ops
        weight_prepacked=st.booleans(),  # Whether to use prepacked weights
    )
    @settings(deadline=None)
    def test_zendnn_linear_fusion_2d_input(
        self,
        batch_size,
        in_features,
        out_features,
        has_bias,
        use_bf16,
        fusion_op_idx,
        weight_prepacked,
    ):
        if use_bf16 and not self.bf16_supported:
            # Skip test if bf16 is requested but not supported
            self.skipTest("BFloat16 not supported on this device")

        dtype = torch.bfloat16 if use_bf16 else torch.float32
        fusion_op = self.fusion_ops[fusion_op_idx]

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

        # Create second input for binary ops (add, mul)
        second_input = None
        if fusion_op in ["add", "mul"]:
            second_input = torch.randn(
                batch_size,
                out_features,
                dtype=dtype,
                requires_grad=False,
                device=self.device,
            )

        rtol = 1e-2 if use_bf16 else 1e-4  # Relax tolerances for BF16
        atol = 1e-2 if use_bf16 else 1e-4

        self._test_zendnn_linear_fusion(
            input, weight, bias, fusion_op, second_input, atol, rtol, weight_prepacked
        )

    @given(
        batch_size=st.integers(1, 16),
        seq_len=st.integers(1, 32),
        in_features=st.integers(2, 128),
        out_features=st.integers(2, 128),
        has_bias=st.booleans(),
        use_bf16=st.booleans(),
        fusion_op_idx=st.integers(
            0, len(FUSION_TO_ACTIVATION_FUNC_MAP) - 1
        ),  # Index into self.fusion_ops
        weight_prepacked=st.booleans(),  # Whether to use prepacked weights
    )
    @settings(deadline=None)
    def test_zendnn_linear_fusion_3d_input(
        self,
        batch_size,
        seq_len,
        in_features,
        out_features,
        has_bias,
        use_bf16,
        fusion_op_idx,
        weight_prepacked,
    ):
        if use_bf16 and not self.bf16_supported:
            # Skip test if bf16 is requested but not supported
            self.skipTest("BFloat16 not supported on this device")

        dtype = torch.bfloat16 if use_bf16 else torch.float32
        fusion_op = self.fusion_ops[fusion_op_idx]

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

        # Create second input for binary ops (add, mul)
        second_input = None
        if fusion_op in ["add", "mul"]:
            second_input = torch.randn(
                batch_size,
                seq_len,
                out_features,
                dtype=dtype,
                requires_grad=False,
                device=self.device,
            )

        rtol = 1e-2 if use_bf16 else 1e-4  # Relax tolerances for BF16
        atol = 1e-2 if use_bf16 else 1e-4

        self._test_zendnn_linear_fusion(
            input, weight, bias, fusion_op, second_input, atol, rtol, weight_prepacked
        )

    @given(
        dims=st.integers(4, 5),
        batch_dim=st.integers(1, 8),
        in_features=st.integers(2, 64),
        out_features=st.integers(2, 64),
        has_bias=st.booleans(),
        use_bf16=st.booleans(),
        fusion_op_idx=st.integers(
            0, len(FUSION_TO_ACTIVATION_FUNC_MAP) - 1
        ),  # Index into self.fusion_ops
        weight_prepacked=st.booleans(),  # Whether to use prepacked weights
    )
    @settings(deadline=None)
    def test_zendnn_linear_fusion_nd_input(
        self,
        dims,
        batch_dim,
        in_features,
        out_features,
        has_bias,
        use_bf16,
        fusion_op_idx,
        weight_prepacked,
    ):
        if use_bf16 and not self.bf16_supported:
            # Skip test if bf16 is requested but not supported
            self.skipTest("BFloat16 not supported on this device")

        dtype = torch.bfloat16 if use_bf16 else torch.float32
        fusion_op = self.fusion_ops[fusion_op_idx]

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

        # Create second input for binary ops (add, mul)
        second_input = None
        if fusion_op in ["add", "mul"]:
            # Create shape for second input (same as output shape)
            output_shape = list(shape)
            output_shape[-1] = out_features
            second_input = torch.randn(
                *output_shape, dtype=dtype, requires_grad=False, device=self.device
            )

        rtol = 1e-2 if use_bf16 else 1e-4  # Relax tolerances for BF16
        atol = 1e-2 if use_bf16 else 1e-4

        self._test_zendnn_linear_fusion(
            input, weight, bias, fusion_op, second_input, atol, rtol, weight_prepacked
        )

    def test_zendnn_linear_fusion_invalid_op(self):
        # Test with invalid fusion op
        input = torch.randn(10, 20, requires_grad=False, device=self.device)
        weight = torch.randn(30, 20, requires_grad=False, device=self.device)
        bias = torch.randn(30, requires_grad=False, device=self.device)

        with self.assertRaises(RuntimeError):
            torch.ops.aten.zendnn_linear(
                input=input,
                weight=weight,
                bias=bias,
                is_weight_prepacked=False,
                post_op="invalid_op",
                zendnn_op_name="linear_invalid",
            )

    def test_zendnn_linear_fusion_dtype_mismatch(self):
        # Test dtype mismatch between input tensors
        input = torch.randn(
            10, 20, dtype=torch.float32, requires_grad=False, device=self.device
        )
        weight = torch.randn(
            30, 20, dtype=torch.float64, requires_grad=False, device=self.device
        )  # Different dtype

        with self.assertRaises(RuntimeError):
            torch.ops.aten.zendnn_linear(
                input=input,
                weight=weight,
                bias=None,
                is_weight_prepacked=False,
                post_op="relu",
                zendnn_op_name="linear_relu",
            )


@unittest.skipIf(
    not (
        torch._C.has_zendnn
        and (torch._C._cpu._is_amd_cpu() or torch._inductor.config.enable_zendnn)
    ),
    "ZenDNN is not available in this PyTorch build",
)
class TestCompiledLinearFusion(TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.previous_freezing = torch._inductor.config.freezing
        torch._inductor.config.freezing = True
        self.previous_weight_prepack = torch._inductor.config.cpp.weight_prepack
        self.previous_enable_zendnn = torch._inductor.config.enable_zendnn
        # Check if bfloat16 is supported on the current device
        self.bf16_supported = torch._C._cpu._is_avx512_bf16_supported()
        self.fusion_ops = list(FUSION_TO_ACTIVATION_FUNC_MAP.keys())

    def tearDown(self):
        torch._inductor.config.cpp.weight_prepack = self.previous_weight_prepack
        torch._inductor.config.freezing = self.previous_freezing
        torch._inductor.config.enable_zendnn = self.previous_enable_zendnn

    def _test_compiled_linear_fusion(
        self,
        input,
        weight,
        bias,
        fusion_op,
        second_input,
        atol,
        rtol,
        weight_prepacked,
        dynamic_shape,
    ):
        # Reset counters
        counters.clear()
        # Run reference implementation
        torch._dynamo.reset()
        torch._inductor.config.enable_zendnn = False
        linear_model = CustomLinearFusionModel(
            weight, bias, fusion_op, second_input
        ).to(input.dtype)
        compiled_model = torch.compile(
            linear_model, backend="inductor", dynamic=dynamic_shape
        )
        with torch.no_grad():
            expected = compiled_model(input)
        torch._inductor.config.enable_zendnn = True

        torch._inductor.config.cpp.weight_prepack = weight_prepacked
        torch._dynamo.reset()
        compiled_model = torch.compile(
            copy.deepcopy(linear_model), backend="inductor", dynamic=dynamic_shape
        )
        with torch.no_grad():
            self.assertEqual(counters["zendnn"]["zendnn_linear_" + fusion_op], 0)
            self.assertEqual(counters["zendnn"]["zendnn_weight_prepack_for_linear"], 0)
            result = compiled_model(input)
            self.assertEqual(counters["zendnn"]["zendnn_linear_" + fusion_op], 1)
            self.assertEqual(
                counters["zendnn"]["zendnn_weight_prepack_for_linear"],
                1 if weight_prepacked else 0,
            )

        # Compare results
        torch.testing.assert_close(
            result, expected, rtol=rtol, atol=atol, equal_nan=True
        )

    @given(
        batch_size=st.integers(1, 32),
        in_features=st.integers(2, 256),
        out_features=st.integers(2, 256),
        has_bias=st.booleans(),
        use_bf16=st.booleans(),
        fusion_op_idx=st.integers(
            0, len(FUSION_TO_ACTIVATION_FUNC_MAP) - 1
        ),  # Index into self.fusion_ops
        weight_prepacked=st.booleans(),  # Whether to use prepacked weights
        dynamic_shape=st.booleans(),  # Whether to use dynamic shape
    )
    @settings(deadline=None)
    def test_compiled_linear_fusion_2d_input(
        self,
        batch_size,
        in_features,
        out_features,
        has_bias,
        use_bf16,
        fusion_op_idx,
        weight_prepacked,
        dynamic_shape,
    ):
        if use_bf16 and not self.bf16_supported:
            # Skip test if bf16 is requested but not supported
            self.skipTest("BFloat16 not supported on this device")

        dtype = torch.bfloat16 if use_bf16 else torch.float32
        fusion_op = self.fusion_ops[fusion_op_idx]

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

        # Create second input for binary ops (add, mul)
        second_input = None
        if fusion_op in ["add", "mul"]:
            second_input = torch.randn(
                batch_size,
                out_features,
                dtype=dtype,
                requires_grad=False,
                device=self.device,
            )

        rtol = 1e-2 if use_bf16 else 1e-4  # Relax tolerances for BF16
        atol = 1e-2 if use_bf16 else 1e-4

        self._test_compiled_linear_fusion(
            input,
            weight,
            bias,
            fusion_op,
            second_input,
            atol,
            rtol,
            weight_prepacked,
            dynamic_shape,
        )

    @given(
        batch_size=st.integers(1, 16),
        seq_len=st.integers(1, 32),
        in_features=st.integers(2, 128),
        out_features=st.integers(2, 128),
        has_bias=st.booleans(),
        use_bf16=st.booleans(),
        fusion_op_idx=st.integers(
            0, len(FUSION_TO_ACTIVATION_FUNC_MAP) - 1
        ),  # Index into self.fusion_ops
        weight_prepacked=st.booleans(),  # Whether to use prepacked weights
        dynamic_shape=st.booleans(),  # Whether to use dynamic shape
    )
    @settings(deadline=None)
    def test_compiled_linear_fusion_3d_input(
        self,
        batch_size,
        seq_len,
        in_features,
        out_features,
        has_bias,
        use_bf16,
        fusion_op_idx,
        weight_prepacked,
        dynamic_shape,
    ):
        if use_bf16 and not self.bf16_supported:
            # Skip test if bf16 is requested but not supported
            self.skipTest("BFloat16 not supported on this device")

        dtype = torch.bfloat16 if use_bf16 else torch.float32
        fusion_op = self.fusion_ops[fusion_op_idx]

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

        # Create second input for binary ops (add, mul)
        second_input = None
        if fusion_op in ["add", "mul"]:
            second_input = torch.randn(
                batch_size,
                seq_len,
                out_features,
                dtype=dtype,
                requires_grad=False,
                device=self.device,
            )

        rtol = 1e-2 if use_bf16 else 1e-4  # Relax tolerances for BF16
        atol = 1e-2 if use_bf16 else 1e-4

        self._test_compiled_linear_fusion(
            input,
            weight,
            bias,
            fusion_op,
            second_input,
            atol,
            rtol,
            weight_prepacked,
            dynamic_shape,
        )

    @given(
        dims=st.integers(4, 5),
        batch_dim=st.integers(1, 8),
        in_features=st.integers(2, 64),
        out_features=st.integers(2, 64),
        has_bias=st.booleans(),
        use_bf16=st.booleans(),
        fusion_op_idx=st.integers(
            0, len(FUSION_TO_ACTIVATION_FUNC_MAP) - 1
        ),  # Index into self.fusion_ops
        weight_prepacked=st.booleans(),  # Whether to use prepacked weights
        dynamic_shape=st.booleans(),  # Whether to use dynamic shape
    )
    @settings(deadline=None)
    def test_compiled_linear_fusion_nd_input(
        self,
        dims,
        batch_dim,
        in_features,
        out_features,
        has_bias,
        use_bf16,
        fusion_op_idx,
        weight_prepacked,
        dynamic_shape,
    ):
        if use_bf16 and not self.bf16_supported:
            # Skip test if bf16 is requested but not supported
            self.skipTest("BFloat16 not supported on this device")

        dtype = torch.bfloat16 if use_bf16 else torch.float32
        fusion_op = self.fusion_ops[fusion_op_idx]

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

        # Create second input for binary ops (add, mul)
        second_input = None
        if fusion_op in ["add", "mul"]:
            # Create shape for second input (same as output shape)
            output_shape = list(shape)
            output_shape[-1] = out_features
            second_input = torch.randn(
                *output_shape, dtype=dtype, requires_grad=False, device=self.device
            )

        rtol = 1e-2 if use_bf16 else 1e-4  # Relax tolerances for BF16
        atol = 1e-2 if use_bf16 else 1e-4

        self._test_compiled_linear_fusion(
            input,
            weight,
            bias,
            fusion_op,
            second_input,
            atol,
            rtol,
            weight_prepacked,
            dynamic_shape,
        )


@unittest.skipIf(
    not (
        torch._C.has_zendnn
        and (torch._C._cpu._is_amd_cpu() or torch._inductor.config.enable_zendnn)
    ),
    "ZenDNN is not available in this PyTorch build",
)
class TestExportedLinearFusion(TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        import tempfile

        self.tmpdir = tempfile.mkdtemp()
        self.previous_weight_prepack = torch._inductor.config.cpp.weight_prepack
        # Check if bfloat16 is supported on the current device
        self.bf16_supported = torch._C._cpu._is_avx512_bf16_supported()
        self.fusion_ops = list(FUSION_TO_ACTIVATION_FUNC_MAP.keys())

    def tearDown(self):
        import shutil

        torch._inductor.config.cpp.weight_prepack = self.previous_weight_prepack
        shutil.rmtree(self.tmpdir)

    def _test_exported_linear_fusion(
        self,
        input,
        weight,
        bias,
        fusion_op,
        second_input,
        atol,
        rtol,
        weight_prepacked,
        dynamic_shape,
    ):
        # Reset counters
        counters.clear()

        # Run reference implementation
        torch._dynamo.reset()
        torch._inductor.config.enable_zendnn = False
        linear_model = CustomLinearFusionModel(
            weight, bias, fusion_op, second_input
        ).to(input.dtype)
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
        torch._inductor.config.cpp.weight_prepack = weight_prepacked
        torch._dynamo.reset()
        self.assertEqual(counters["zendnn"]["zendnn_linear_" + fusion_op], 0)
        self.assertEqual(counters["zendnn"]["zendnn_weight_prepack_for_linear"], 0)
        # Create exported version
        exported = torch.export.export(
            copy.deepcopy(linear_model), (input,)
        )  # dynamic shapes are currenlt not supported in export
        output_path = torch._inductor.aoti_compile_and_package(
            exported,
            package_path=os.path.join(self.tmpdir, "model_zendnn.pt2"),
        )
        self.assertEqual(counters["zendnn"]["zendnn_linear_" + fusion_op], 1)
        self.assertEqual(
            counters["zendnn"]["zendnn_weight_prepack_for_linear"],
            1 if weight_prepacked else 0,
        )
        counters.clear()

        exported_model = torch._inductor.aoti_load_package(output_path)

        # Forward pass with both models
        with torch.no_grad():
            result = exported_model(input)
        self.assertEqual(counters["zendnn"]["zendnn_linear_" + fusion_op], 0)
        self.assertEqual(counters["zendnn"]["zendnn_weight_prepack_for_linear"], 0)

        # Compare results
        torch.testing.assert_close(
            result, expected, rtol=rtol, atol=atol, equal_nan=True
        )

    @given(
        batch_size=st.integers(1, 32),
        in_features=st.integers(2, 256),
        out_features=st.integers(2, 256),
        has_bias=st.booleans(),
        use_bf16=st.booleans(),
        fusion_op_idx=st.integers(
            0, len(FUSION_TO_ACTIVATION_FUNC_MAP) - 1
        ),  # Index into self.fusion_ops
        weight_prepacked=st.booleans(),  # Whether to use prepacked weights
        dynamic_shape=st.booleans(),  # Whether to use dynamic shape
    )
    @settings(deadline=None)
    def test_exported_linear_fusion_2d_input(
        self,
        batch_size,
        in_features,
        out_features,
        has_bias,
        use_bf16,
        fusion_op_idx,
        weight_prepacked,
        dynamic_shape,
    ):
        if use_bf16 and not self.bf16_supported:
            # Skip test if bf16 is requested but not supported
            self.skipTest("BFloat16 not supported on this device")

        dtype = torch.bfloat16 if use_bf16 else torch.float32
        fusion_op = self.fusion_ops[fusion_op_idx]

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

        # Create second input for binary ops (add, mul)
        second_input = None
        if fusion_op in ["add", "mul"]:
            second_input = torch.randn(
                batch_size,
                out_features,
                dtype=dtype,
                requires_grad=False,
                device=self.device,
            )

        rtol = 1e-2 if use_bf16 else 1e-4  # Relax tolerances for BF16
        atol = 1e-2 if use_bf16 else 1e-4

        self._test_exported_linear_fusion(
            input,
            weight,
            bias,
            fusion_op,
            second_input,
            atol,
            rtol,
            weight_prepacked,
            dynamic_shape,
        )

    @given(
        batch_size=st.integers(1, 16),
        seq_len=st.integers(1, 32),
        in_features=st.integers(2, 128),
        out_features=st.integers(2, 128),
        has_bias=st.booleans(),
        use_bf16=st.booleans(),
        fusion_op_idx=st.integers(
            0, len(FUSION_TO_ACTIVATION_FUNC_MAP) - 1
        ),  # Index into self.fusion_ops
        weight_prepacked=st.booleans(),  # Whether to use prepacked weights
        dynamic_shape=st.booleans(),  # Whether to use dynamic shape
    )
    @settings(deadline=None)
    def test_exported_linear_fusion_3d_input(
        self,
        batch_size,
        seq_len,
        in_features,
        out_features,
        has_bias,
        use_bf16,
        fusion_op_idx,
        weight_prepacked,
        dynamic_shape,
    ):
        if use_bf16 and not self.bf16_supported:
            # Skip test if bf16 is requested but not supported
            self.skipTest("BFloat16 not supported on this device")

        dtype = torch.bfloat16 if use_bf16 else torch.float32
        fusion_op = self.fusion_ops[fusion_op_idx]

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

        # Create second input for binary ops (add, mul)
        second_input = None
        if fusion_op in ["add", "mul"]:
            second_input = torch.randn(
                batch_size,
                seq_len,
                out_features,
                dtype=dtype,
                requires_grad=False,
                device=self.device,
            )

        rtol = 1e-2 if use_bf16 else 1e-4  # Relax tolerances for BF16
        atol = 1e-2 if use_bf16 else 1e-4

        self._test_exported_linear_fusion(
            input,
            weight,
            bias,
            fusion_op,
            second_input,
            atol,
            rtol,
            weight_prepacked,
            dynamic_shape,
        )

    @given(
        dims=st.integers(4, 5),
        batch_dim=st.integers(1, 8),
        in_features=st.integers(2, 64),
        out_features=st.integers(2, 64),
        has_bias=st.booleans(),
        use_bf16=st.booleans(),
        fusion_op_idx=st.integers(
            0, len(FUSION_TO_ACTIVATION_FUNC_MAP) - 1
        ),  # Index into self.fusion_ops
        weight_prepacked=st.booleans(),  # Whether to use prepacked weights
        dynamic_shape=st.booleans(),  # Whether to use dynamic shape
    )
    @settings(deadline=None)
    def test_exported_linear_fusion_nd_input(
        self,
        dims,
        batch_dim,
        in_features,
        out_features,
        has_bias,
        use_bf16,
        fusion_op_idx,
        weight_prepacked,
        dynamic_shape,
    ):
        if use_bf16 and not self.bf16_supported:
            # Skip test if bf16 is requested but not supported
            self.skipTest("BFloat16 not supported on this device")

        dtype = torch.bfloat16 if use_bf16 else torch.float32
        fusion_op = self.fusion_ops[fusion_op_idx]

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

        # Create second input for binary ops (add, mul)
        second_input = None
        if fusion_op in ["add", "mul"]:
            # Create shape for second input (same as output shape)
            output_shape = list(shape)
            output_shape[-1] = out_features
            second_input = torch.randn(
                *output_shape, dtype=dtype, requires_grad=False, device=self.device
            )

        rtol = 1e-2 if use_bf16 else 1e-4  # Relax tolerances for BF16
        atol = 1e-2 if use_bf16 else 1e-4

        self._test_exported_linear_fusion(
            input,
            weight,
            bias,
            fusion_op,
            second_input,
            atol,
            rtol,
            weight_prepacked,
            dynamic_shape,
        )


if __name__ == "__main__":
    run_tests()
