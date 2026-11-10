# Owner(s): ["module: onnx"]
"""Native convolution, normalization, and neural-network translations."""

from __future__ import annotations

from unittest import mock

import numpy as np

import onnx
import onnx_ir as ir
import onnxruntime as ort
from test_native_factory_sequence_ops import _evaluate
from test_native_torchlib import no_script

import torch
from torch.onnx._internal._lazy_import import onnxscript_apis
from torch.onnx._internal.exporter import _registration
from torch.onnx._internal.exporter._torchlib import _torchlib_registry
from torch.onnx._internal.exporter._torchlib.ops import neural, nn, normalization
from torch.testing._internal import common_utils
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import HardwareClassification


class TestNativeNeuralOps(common_utils.TestCase):
    hw_classification = HardwareClassification.CPU

    @common_utils.parametrize("dimensions", [1, 2, 3])
    @common_utils.parametrize("complex_input", [False, True])
    @common_utils.parametrize("bias", [False, True])
    @common_utils.parametrize("batched", [False, True])
    def test_convolution(self, device, dimensions, complex_input, bias, batched):
        torch.manual_seed(0)
        dtype = torch.complex64 if complex_input else torch.float32
        x = torch.randn((2, 4, *([5] * dimensions)), dtype=dtype, device=device)
        w = torch.randn((4, 2, *([3] * dimensions)), dtype=dtype, device=device)
        b = torch.randn(4, dtype=dtype, device=device)
        if not batched:
            x = x[0]
        expected = getattr(torch.nn.functional, f"conv{dimensions}d")(
            x, w, b if bias else None, padding=1, groups=2
        )
        inputs = [x, w, *([b] if bias else [])]
        impl = getattr(
            normalization,
            f"aten_conv{dimensions}d" + ("_complex" if complex_input else ""),
        )
        if complex_input:
            inputs = [torch.view_as_real(value) for value in inputs]
            expected = torch.view_as_real(expected)
        actual = _evaluate(
            lambda a, w, *b: impl(a, w, b[0] if b else None, padding=1, groups=2),
            inputs,
            expected,
            runtime=True,
        )
        self.assertEqual(actual, expected, atol=3e-6, rtol=3e-5)

    @common_utils.parametrize("dimensions", [1, 2, 3])
    def test_transposed_convolution(self, device, dimensions):
        x = torch.randn(2, 3, *([4] * dimensions), device=device)
        w = torch.randn(3, 2, *([3] * dimensions), device=device)
        expected = getattr(torch.nn.functional, f"conv_transpose{dimensions}d")(
            x, w, stride=2, padding=1, output_padding=1
        )
        actual = _evaluate(
            lambda a, w: normalization.aten_convolution(
                a,
                w,
                stride=[2],
                padding=[1],
                transposed=True,
                output_padding=[1] * dimensions,
            ),
            [x, w],
            expected,
            runtime=True,
        )
        self.assertEqual(actual, expected)

    @common_utils.parametrize("training", [False, True])
    @common_utils.parametrize("affine", [False, True])
    @common_utils.parametrize("momentum", [0.0, 0.1, 1.0])
    @common_utils.parametrize("dtype", [torch.float32, torch.float64])
    def test_batch_norm_functional(self, device, training, affine, momentum, dtype):
        torch.manual_seed(0)
        x = torch.randn(2, 3, 4, 5, device=device, dtype=dtype)
        weight, bias = (
            torch.randn(3, device=device, dtype=dtype),
            torch.randn(3, device=device, dtype=dtype),
        )
        mean, var = (
            torch.randn(3, device=device, dtype=dtype),
            torch.rand(3, device=device, dtype=dtype) + 1,
        )
        expected = torch.ops.aten._native_batch_norm_legit_functional(
            x,
            weight if affine else None,
            bias if affine else None,
            mean,
            var,
            training,
            momentum,
            1e-5,
        )
        if not training:
            # Source torchlib uses CUDA's saved statistics in inference mode.
            expected = (
                expected[0],
                mean.float(),
                torch.rsqrt(var + 1e-5).float(),
                mean,
                var,
            )
        actual = _evaluate(
            lambda a,
            w,
            b,
            m,
            v: normalization.aten__native_batch_norm_legit_functional(
                a,
                w if affine else None,
                b if affine else None,
                m,
                v,
                training,
                momentum,
                1e-5,
            ),
            [x, weight, bias, mean, var],
            expected,
        )
        self.assertEqual(actual, list(expected), atol=2e-6, rtol=2e-5)

    def test_batch_norm_no_stats(self, device):
        x = torch.randn(2, 3, 4, device=device)
        expected = torch.ops.aten._native_batch_norm_legit.no_stats(
            x, None, None, True, 0.1, 1e-5
        )
        actual = _evaluate(
            lambda a: normalization.aten__native_batch_norm_no_stats(a, training=True),
            [x],
            expected,
        )
        self.assertEqual(actual, list(expected))

    @common_utils.parametrize(
        "impl",
        [
            lambda a: neural.aten_upsample_linear1d_vec(a, None, False, None),
            lambda a: neural.aten_upsample_bilinear2d_vec(a, None, False, None),
            lambda a: neural.aten_upsample_nearestnd_vec(a),
        ],
    )
    def test_resize_requires_output_shape(self, device, impl):
        x = torch.randn(2, 3, 4, device=device)
        with self.assertRaisesRegex(
            ValueError, "Supply one of them to determine the output shape"
        ):
            _evaluate(impl, [x], x)

    @common_utils.parametrize("stats", [False, True])
    def test_instance_norm(self, device, stats):
        x = torch.randn(2, 3, 4, 5, device=device)
        mean, var = torch.randn(3, device=device), torch.rand(3, device=device) + 1
        expected = torch.nn.functional.instance_norm(
            x, mean, var, use_input_stats=stats
        )
        actual = _evaluate(
            lambda a, m, v: normalization.aten_instance_norm(
                a, running_mean=m, running_var=v, use_input_stats=stats
            ),
            [x, mean, var],
            expected,
        )
        self.assertEqual(actual, expected)

    @common_utils.parametrize("affine", [False, True])
    def test_native_norms(self, device, affine):
        x = torch.randn(2, 4, 3, device=device)
        w, b = torch.randn(4, device=device), torch.randn(4, device=device)
        expected = torch.native_group_norm(
            x, w if affine else None, b if affine else None, 2, 4, 3, 2, 1e-5
        )
        actual = _evaluate(
            lambda a, w, b: normalization.aten_native_group_norm(
                a, w if affine else None, b if affine else None, group=2
            ),
            [x, w, b],
            expected,
        )
        self.assertEqual(actual, list(expected))
        w, b = torch.randn(4, 3, device=device), torch.randn(4, 3, device=device)
        expected = torch.native_layer_norm(
            x, [4, 3], w if affine else None, b if affine else None, 1e-5
        )
        actual = _evaluate(
            lambda a, w, b: normalization.aten_native_layer_norm(
                a, [4, 3], w if affine else None, b if affine else None
            ),
            [x, w, b],
            expected,
        )
        self.assertEqual(actual, list(expected))

    @common_utils.parametrize("mask", ["none", "bool", "float", "causal"])
    @common_utils.parametrize("dropout", [0.0, 1.0])
    def test_attention(self, device, mask, dropout):
        q = torch.randn(2, 4, 3, 8, device=device)
        k = torch.randn(2, 2, 5, 8, device=device)
        v = torch.randn_like(k)
        m = torch.rand(3, 5, device=device) > 0.5
        m[0] = False
        if mask == "float":
            m = torch.where(m, 0.0, -float("inf"))
        expected = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            m if mask in ("bool", "float") else None,
            dropout,
            mask == "causal",
            enable_gqa=True,
        )

        def attention(q, k, v, m):
            for value, heads in ((q, 4), (k, 2), (v, 2)):
                value.shape = ir.Shape([None, heads, None, None])
            return neural.aten_scaled_dot_product_attention(
                q,
                k,
                v,
                m if mask in ("bool", "float") else None,
                dropout,
                mask == "causal",
                enable_gqa=True,
            )

        actual = _evaluate(
            attention,
            [q, k, v, m],
            expected,
        )
        self.assertEqual(actual, expected, atol=1e-6, rtol=2e-5)

    @common_utils.parametrize("optimize", [False, True])
    def test_export(self, device, optimize):
        class Model(torch.nn.Module):
            def forward(self, x, weight):
                x = torch.nn.functional.conv2d(x, weight, padding=1)
                x = torch.nn.functional.group_norm(x, 2)
                return torch.nn.functional.max_pool2d(torch.nn.functional.gelu(x), 2)

        model = Model().eval()
        args = (
            torch.randn(2, 4, 6, 6, device=device),
            torch.randn(4, 4, 3, 3, device=device),
        )
        with (
            mock.patch.object(
                onnxscript_apis,
                "get_torchlib_ops",
                side_effect=AssertionError("legacy fallback"),
            ),
            no_script(),
        ):
            program = torch.onnx.export(
                model,
                args,
                dynamo=True,
                optimize=optimize,
                verbose=False,
                opset_version=18,
                dynamic_shapes=({0: torch.export.Dim("batch")}, None),
            )
        onnx.checker.check_model(program.model_proto, full_check=True)
        session = ort.InferenceSession(
            program.model_proto.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        for batch in (1, 3):
            inputs = (torch.randn(batch, 4, 6, 6, device=device), args[1])
            actual = session.run(
                None, {v.name: x.numpy() for v, x in zip(session.get_inputs(), inputs)}
            )[0]
            self.assertEqual(
                torch.from_numpy(np.asarray(actual)),
                model(*inputs),
                atol=2e-6,
                rtol=2e-5,
            )

    @common_utils.parametrize("opset", [18, 20, 21, 23])
    def test_registration(self, device, opset):
        registry = _registration.ONNXRegistry.from_torchlib(opset)
        entries = [
            meta
            for meta in _torchlib_registry.get_torchlib_ops()
            if meta.onnx_function.function.__module__
            in (neural.__name__, normalization.__name__)
        ]
        self.assertEqual(len(entries), 75)
        newer = {
            torch.ops.aten.gelu.default: (20, nn.aten_gelu_opset20),
            torch.ops.aten.group_norm.default: (21, nn.aten_group_norm),
            torch.ops.aten.scaled_dot_product_attention.default: (
                23,
                nn.aten_scaled_dot_product_attention_23,
            ),
        }
        for entry in entries:
            actual = [
                d
                for d in registry.get_decomps(entry.fx_target)
                if d.is_complex == entry.is_complex
            ]
            minimum, impl = newer.get(entry.fx_target, (999, None))
            self.assertIs(
                actual[0].onnx_function.function,
                impl if opset >= minimum else entry.onnx_function.function,
            )


instantiate_device_type_tests(TestNativeNeuralOps, globals(), only_for="cpu")


if __name__ == "__main__":
    common_utils.run_tests()
