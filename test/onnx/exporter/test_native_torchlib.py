# Owner(s): ["module: onnx"]
"""PyTorch-owned translations construct IR without ONNX Script evaluation."""

from __future__ import annotations

import contextlib
import operator
from unittest import mock

import numpy as np

import onnx
import onnx_ir as ir
import onnxscript
import onnxscript.evaluator
from onnx.reference import ReferenceEvaluator

import torch
from torch.onnx._internal.exporter import (
    _core,
    _custom_translation,
    _ir_builder,
    _registration,
    _schemas,
)
from torch.onnx._internal.exporter._torchlib import _torchlib_registry
from torch.onnx._internal.exporter._torchlib.ops import (
    core,
    hop,
    nn,
    quantized_decomposed,
    symbolic,
    symops,
)
from torch.testing._internal import common_utils
from torch.testing._internal.common_utils import HardwareClassification


@contextlib.contextmanager
def no_script():
    with (
        mock.patch.object(
            onnxscript, "script", side_effect=AssertionError("script called")
        ),
        mock.patch.object(
            onnxscript,
            "TracedOnnxFunction",
            side_effect=AssertionError("tracing called"),
        ),
        mock.patch.object(
            onnxscript.evaluator,
            "default_as",
            side_effect=AssertionError("evaluator called"),
        ),
    ):
        yield


@common_utils.instantiate_parametrized_tests
class TestNativeTorchlib(common_utils.TestCase):
    hw_classification = HardwareClassification.GENERIC

    def evaluate(self, function, tensors, *, opset=18, **kwargs):
        graph = ir.Graph([], [], nodes=[], name="native", opset_imports={"": opset})
        builder = _ir_builder.GraphBuilder(graph, opset=opset)
        values = [
            ir.Value(
                name=f"input_{i}",
                shape=ir.Shape(tensor.shape),
                type=ir.TensorType(ir.DataType.from_numpy(tensor.dtype)),
            )
            for i, tensor in enumerate(tensors)
        ]
        graph.inputs.extend(values)
        with no_script(), builder.activate():
            outputs = function(*values, **kwargs)
        if isinstance(outputs, ir.Value):
            outputs = (outputs,)
        graph.outputs.extend(outputs)
        model = ir.Model(graph, ir_version=10)
        model.functions.update(builder.functions)
        proto = ir.to_proto(model)
        # Schema inference fills intermediate shapes, not translation-time tensor casts.
        proto = onnx.shape_inference.infer_shapes(proto, strict_mode=True)
        onnx.checker.check_model(proto)
        return ReferenceEvaluator(proto).run(
            None, dict(zip((value.name for value in values), tensors))
        )

    def test_registration_is_native_and_preserves_metadata(self):
        with no_script():
            entries = _torchlib_registry.get_torchlib_ops()
        self.assertTrue(entries)
        for entry in entries:
            self.assertIsInstance(
                entry.onnx_function, _custom_translation.NativeTranslation
            )
            self.assertFalse(entry.is_custom)
            self.assertTrue(entry.skip_signature_inference)
            self.assertIs(entry.signature, entry.onnx_function.signature)
        abs_meta = next(entry for entry in entries if entry.fx_target is operator.abs)
        self.assertIsNone(abs_meta.signature)
        self.assertIs(abs_meta.onnx_function.function, core.aten_abs)

    def test_registration_does_not_infer_or_compile(self):
        def translation(x: ir.Value) -> ir.Value:
            return _ir_builder.current_builder().call("Neg", x)[0]

        with (
            no_script(),
            mock.patch.object(
                _schemas,
                "op_signature_from_function",
                side_effect=AssertionError("inference called"),
            ),
            mock.patch.object(_torchlib_registry, "_registry", []),
        ):
            self.assertIs(
                _torchlib_registry.onnx_impl(operator.neg)(translation), translation
            )
            self.assertIsNone(_torchlib_registry.get_torchlib_ops()[0].signature)

    @common_utils.parametrize("complex_input", [False, True])
    def test_native_export_does_not_evaluate_script(self, complex_input):
        class Model(torch.nn.Module):
            def forward(self, x):
                return x.abs()

        # External fallback registrations are still allowed to load their source library.
        _registration.ONNXRegistry.from_torchlib()
        x = torch.tensor([-3.0, 4.0])
        if complex_input:
            x = torch.complex(x, x.flip(0))
        with (
            no_script(),
            mock.patch.object(
                _core,
                "_call_legacy_translation",
                side_effect=AssertionError("legacy translation called"),
            ),
        ):
            program = torch.onnx.export(
                Model(), (x,), opset_version=20, optimize=False, verbose=False
            )
        input_data = torch.view_as_real(x) if complex_input else x
        result = ReferenceEvaluator(ir.to_proto(program.model)).run(
            None, {program.model.graph.inputs[0].name: input_data.numpy()}
        )[0]
        self.assertEqual(torch.from_numpy(result), x.abs())

    def test_symbolic_export_does_not_evaluate_script(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                y = torch.onnx.ops.symbolic(
                    "::Identity", (x,), dtype=x.dtype, shape=x.shape
                )
                return torch.onnx.ops.symbolic_multi_out(
                    "::Dropout",
                    (y,),
                    dtypes=(x.dtype, torch.bool),
                    shapes=(x.shape, x.shape),
                )

        _registration.ONNXRegistry.from_torchlib()
        x = torch.tensor([-3.0, 4.0])
        with (
            no_script(),
            mock.patch.object(
                _core,
                "_call_legacy_translation",
                side_effect=AssertionError("legacy translation called"),
            ),
        ):
            program = torch.onnx.export(
                Model(), (x,), opset_version=20, optimize=False, verbose=False
            )
        onnx.checker.check_model(ir.to_proto(program.model))
        self.assertEqual(
            [node.op_type for node in program.model.graph], ["Identity", "Dropout"]
        )
        self.assertEqual(len(program.model.graph.outputs), 2)

    @common_utils.parametrize("opset", [18, 20, 23])
    def test_abs_and_complex_abs(self, opset):
        x = np.array([-3.0, 0.0, 4.0], dtype=np.float32)
        self.assertEqual(self.evaluate(core.aten_abs, [x], opset=opset)[0], abs(x))
        complex_x = np.array([[-3.0, 4.0], [0.0, -2.0]], dtype=np.float32)
        self.assertEqual(
            self.evaluate(core.aten_abs_complex, [complex_x], opset=opset)[0],
            np.array([5.0, 2.0], dtype=np.float32),
        )

    def test_symbolic_scalar_and_sequence_operations(self):
        x = np.array(3, dtype=np.int64)
        y = np.array(7, dtype=np.int64)
        for function, expected in ((symops.sym_max, 7), (symops.sym_min, 3)):
            self.assertEqual(self.evaluate(function, [x, y])[0], np.array(expected))
        self.assertEqual(
            self.evaluate(symops.sym_float, [x])[0], np.array(3, dtype=np.float32)
        )
        self.assertEqual(
            self.evaluate(lambda a, b: symops.sym_sum([a, b, a]), [x, y])[0],
            np.array(13),
        )
        self.assertEqual(self.evaluate(lambda: symops.sym_sum([]), [])[0], np.array(0))
        self.assertEqual(self.evaluate(lambda a: symops.sym_sum([a]), [x])[0], x)
        self.assertEqual(
            self.evaluate(symops.sym_not, [np.array(True)])[0], np.array(False)
        )
        self.assertEqual(self.evaluate(symops.sym_ite, [np.array(True), x, y])[0], x)

    @common_utils.parametrize("opset", [20, 23])
    @common_utils.parametrize("approximate", ["none", "tanh"])
    def test_gelu(self, opset, approximate):
        x = np.linspace(-3, 3, 12, dtype=np.float32)
        result = self.evaluate(
            nn.aten_gelu_opset20, [x], opset=opset, approximate=approximate
        )[0]
        self.assertEqual(
            torch.from_numpy(result),
            torch.nn.functional.gelu(torch.from_numpy(x), approximate=approximate),
        )

    def test_group_norm(self):
        x = np.linspace(-3, 3, 48, dtype=np.float32).reshape(2, 4, 6)
        result = self.evaluate(nn.aten_group_norm, [x], opset=21, num_groups=2)[0]
        self.assertEqual(
            torch.from_numpy(result),
            torch.nn.functional.group_norm(torch.from_numpy(x), 2),
        )

    def test_rms_norm(self):
        x = np.linspace(-3, 3, 24, dtype=np.float32).reshape(2, 3, 4)
        result = self.evaluate(
            nn.aten_rms_norm, [x], opset=23, normalized_shape=[4], eps=1e-5
        )[0]
        self.assertEqual(
            torch.from_numpy(result),
            torch.nn.functional.rms_norm(torch.from_numpy(x), [4], eps=1e-5),
        )

    def test_attention(self):
        x = np.linspace(-1, 1, 24, dtype=np.float32).reshape(1, 2, 3, 4)
        result = self.evaluate(
            nn.aten_scaled_dot_product_attention_23, [x, x, x], opset=23
        )[0]
        self.assertEqual(
            torch.from_numpy(result),
            torch.nn.functional.scaled_dot_product_attention(
                *(torch.from_numpy(x) for _ in range(3))
            ),
        )

    @common_utils.parametrize("masked", [False, True])
    def test_attention_primitive_helpers(self, masked):
        x = np.linspace(-1, 1, 24, dtype=np.float32).reshape(1, 2, 3, 4)

        def attention(q, k, v):
            scale = nn._attention_scale(q)
            if masked:
                mask = nn._causal_attention_mask(q, k)
                return nn._aten_scaled_dot_product_attention_float_mask_onnx(
                    q, k, v, mask, scale, 0.0
                )
            return nn._aten_scaled_dot_product_attention_no_mask_onnx(
                q, k, v, scale, 0.0
            )

        result = self.evaluate(attention, [x, x, x], opset=23)[0]
        self.assertEqual(
            torch.from_numpy(result),
            torch.nn.functional.scaled_dot_product_attention(
                *(torch.from_numpy(x) for _ in range(3)), is_causal=masked
            ),
        )

    @common_utils.parametrize(
        "dtype,quant_min,quant_max",
        [
            (ir.DataType.INT8, -8, 7),
            (ir.DataType.UINT8, 0, 255),
            (ir.DataType.INT32, -(2**31), 2**31 - 1),
        ],
    )
    def test_quantization(self, dtype, quant_min, quant_max):
        x = np.array([-100.0, -0.75, 0.0, 1.25, 100.0], dtype=np.float32)
        result = self.evaluate(
            quantized_decomposed.quantized_decomposed_quantize_per_tensor,
            [x],
            scale=0.5,
            zero_point=1,
            quant_min=quant_min,
            quant_max=quant_max,
            dtype=dtype,
        )[0]
        expected = np.clip(np.round(x / 0.5) + 1, quant_min, quant_max).astype(
            dtype.numpy()
        )
        self.assertEqual(result, expected)

    def test_per_channel_quantization(self):
        x = np.array([[-1.0, 0.0, 1.0], [-4.0, 0.0, 4.0]], dtype=np.float32)
        scales = np.array([0.5, 2.0], dtype=np.float32)
        zeros = np.array([0, 1], dtype=np.int64)
        for dtype in (ir.DataType.INT8, ir.DataType.INT32):
            result = self.evaluate(
                quantized_decomposed.quantized_decomposed_quantize_per_channel,
                [x, scales, zeros],
                axis=0,
                quant_min=-128,
                quant_max=127,
                dtype=dtype,
            )[0]
            self.assertEqual(
                result,
                (np.round(x / scales[:, None]) + zeros[:, None]).astype(dtype.numpy()),
            )

    def test_symbolic_raw_metadata_and_optional_inputs(self):
        graph = ir.Graph([], [], nodes=[], name="symbolic")
        builder = _ir_builder.GraphBuilder(graph)
        x = ir.Value(name="x", type=ir.TensorType(ir.DataType.FLOAT))
        with no_script(), builder.activate():
            outputs = symbolic._call_symbolic_op(
                "Custom",
                "test.native",
                [x, None, x, None],
                {"count": 2},
                [ir.DataType.FLOAT, ir.DataType.INT64],
                3,
                {"key": "value"},
            )
        node = outputs[0].producer()
        self.assertEqual(list(node.inputs), [x, None, x])
        self.assertEqual(
            [out.dtype for out in outputs], [ir.DataType.FLOAT, ir.DataType.INT64]
        )
        self.assertEqual(node.metadata_props["key"], "value")
        self.assertEqual(graph.opset_imports["test.native"], 3)
        with no_script(), builder.activate():
            no_outputs = symbolic._call_symbolic_op(
                "Sink", "test.native", [x], {}, [], 3, {"sink": "true"}
            )
        self.assertEqual(no_outputs, ())
        self.assertEqual(next(reversed(graph)).metadata_props["sink"], "true")

    @common_utils.parametrize("opset", [18, 19, 20, 21, 23])
    def test_opset_fallback(self, opset):
        registry = _registration.ONNXRegistry.from_torchlib(opset)
        gelu = registry.get_decomps(torch.ops.aten.gelu.default)
        self.assertTrue(gelu)
        self.assertEqual(
            isinstance(gelu[0].onnx_function, _custom_translation.NativeTranslation),
            True,
        )
        self.assertTrue(registry.get_decomps(torch.ops.aten.add.Tensor))
        abs_entries = registry.get_decomps(torch.ops.aten.abs.default)
        self.assertEqual(len(abs_entries), 2)
        self.assertEqual({entry.is_complex for entry in abs_entries}, {False, True})
        self.assertTrue(
            all(
                isinstance(entry.onnx_function, _custom_translation.NativeTranslation)
                for entry in abs_entries
            )
        )

    def test_higher_order_cond_captures(self):
        def identity_function(name):
            value = ir.Value(
                name=f"{name}_in",
                type=ir.TensorType(ir.DataType.FLOAT),
                shape=ir.Shape([2]),
            )
            graph = ir.Graph([value], [], nodes=[], name=name)
            builder = _ir_builder.GraphBuilder(graph)
            graph.outputs.extend(builder.call("Identity", value))
            return ir.Function("test.function", name, graph=graph, attributes=[])

        true_func, false_func = identity_function("true"), identity_function("false")
        graph = ir.Graph([], [], nodes=[], name="cond")
        builder = _ir_builder.GraphBuilder(graph)
        x = ir.Value(
            name="x", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([2])
        )
        cond = ir.Value(
            name="cond", type=ir.TensorType(ir.DataType.BOOL), shape=ir.Shape([])
        )
        graph.inputs.extend([cond, x])
        with no_script(), builder.activate():
            outputs = hop.higher_order_cond(cond, true_func, false_func, [x])
        outputs[0].type, outputs[0].shape = x.type, x.shape
        graph.outputs.extend(outputs)
        graph.opset_imports["test.function"] = 1
        model = ir.Model(graph, ir_version=10, functions=[true_func, false_func])
        proto = ir.to_proto(model)
        onnx.checker.check_model(proto)
        self.assertEqual(
            ReferenceEvaluator(proto).run(
                None,
                {"cond": np.array(True), "x": np.array([1.0, 2.0], dtype=np.float32)},
            )[0],
            np.array([1.0, 2.0], dtype=np.float32),
        )


if __name__ == "__main__":
    common_utils.run_tests()
