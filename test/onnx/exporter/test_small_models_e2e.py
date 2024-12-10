# Owner(s): ["module: onnx"]
"""Unit tests for the onnx dynamo exporter."""

from __future__ import annotations

import numpy as np
import onnxscript
import torchvision

import torch
from torch._ops import OpOverload
from torch.onnx._internal.exporter import _testing as onnx_testing
from torch.testing._internal import common_utils


@common_utils.instantiate_parametrized_tests
class DynamoExporterTest(common_utils.TestCase):
    def test_insert_contiguous_between_transpose_and_view(self):
        class Model(torch.nn.Module):
            def forward(self, query, key, value):
                res = torch.nn.functional.scaled_dot_product_attention(
                    query, key, value
                )
                rest = res.transpose(0, 1)
                return rest.view(8, 32, 128 * 64)

        model = Model()

        query = torch.rand(32, 8, 128, 64, dtype=torch.float16)
        key = torch.rand(32, 8, 128, 64, dtype=torch.float16)
        value = torch.rand(32, 8, 128, 64, dtype=torch.float16)

        ep = torch.export.export(model, (query, key, value), strict=False)
        self.assertNotIn("call_method", str(ep.graph))

        onnx_program = torch.onnx.export(
            model, (query, key, value), dynamo=True, fallback=False
        )
        onnx_testing.assert_onnx_program(onnx_program, atol=1e-3, rtol=1)

    def test_constant_complex(self):
        class MulModule(torch.nn.Module):
            def forward(self, x):
                y = 2 + 3j
                return torch.ops.aten.mul(x, y)

        # Example usage with complex inputs
        x = torch.tensor(
            [[1.0 + 2.0j, 3.0 + 4.0j], [5.0 + 6.0j, 7.0 + 8.0j]], dtype=torch.complex64
        )

        onnx_program = torch.onnx.export(MulModule(), (x,), dynamo=True)
        onnx_testing.assert_onnx_program(onnx_program)

    def test_pow_does_not_trigger_type_promotion(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                return x**2.0

        x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float16)

        onnx_program = torch.onnx.export(Model(), (x,), dynamo=True)
        onnx_testing.assert_onnx_program(onnx_program)
        self.assertNotIn("Cast", [node.op_type for node in onnx_program.model.graph])

    def test_onnx_export_control_flow(self):
        class CondModel(torch.nn.Module):
            def forward(self, x):
                def true_fn(x):
                    return x + 1.0

                def false_fn(x):
                    return x - 42.0

                y = torch.cond(x.sum() > 0, true_fn, false_fn, [x])
                return y

        onnx_program = torch.onnx.export(
            CondModel(),
            (torch.tensor([1, 2]),),
            dynamo=True,
            fallback=False,
        )
        onnx_model = onnx_program.model
        self.assertIn("If", [node.op_type for node in onnx_model.graph])
        onnx_testing.assert_onnx_program(onnx_program)
        # Test different branches
        onnx_testing.assert_onnx_program(onnx_program, args=(torch.tensor([-1, -2]),))

    def test_onnx_export_nested_control_flow_and_nested_weights(self):
        class Submodule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # Nested weight
                self.weight = torch.nn.Parameter(torch.tensor([100.0]))

            def forward(self, x):
                def true_fn(x):
                    return x * self.weight

                def false_fn(x):
                    return x / self.weight

                y = torch.cond(x.sum() <= 0, true_fn, false_fn, [x])
                return y

        class CondModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.submodule = Submodule()
                self.weight = torch.nn.Parameter(torch.tensor([42.0]))

            def forward(self, x):
                def true_fn(x):
                    return self.submodule(x - self.weight)

                def false_fn(x):
                    return x - self.weight

                y = torch.cond(x.sum() > 0, true_fn, false_fn, [x])
                return y

        onnx_program = torch.onnx.export(
            CondModel(),
            (torch.tensor([1, 2]),),
            dynamo=True,
            fallback=False,
        )
        onnx_testing.assert_onnx_program(onnx_program)
        onnx_testing.assert_onnx_program(onnx_program, args=(torch.tensor([0, 0]),))
        onnx_testing.assert_onnx_program(onnx_program, args=(torch.tensor([43, 43]),))

    def test_onnx_export_torchvision_ops(self):
        class VisionModel(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, *x):
                out = torchvision.ops.nms(x[0], x[1], x[2])
                return out

        args = (
            torch.tensor([[0, 0, 1, 1], [0.5, 0.5, 1, 1]], dtype=torch.float),
            torch.tensor([0.1, 0.2]),
            0,
        )
        onnx_program = torch.onnx.export(VisionModel(), args, dynamo=True)
        onnx_testing.assert_onnx_program(onnx_program)

    def test_custom_op(self):
        def onnx_test_numpy_sin(x: torch.Tensor) -> torch.Tensor:
            assert x.device.type == "cpu"
            x_np = x.numpy()
            return torch.from_numpy(np.sin(x_np))


        class ModuleWithACustomOperator(torch.nn.Module):
            def forward(self, x):
                return onnx_test_numpy_sin(x)


        model = ModuleWithACustomOperator()
        x = torch.randn(1, 3)
        expected = model(x)

        def register(fct, fct_shape, namespace, fname):
            schema_str = torch.library.infer_schema(fct, mutates_args=())
            custom_def = torch.library.CustomOpDef(namespace, fname, schema_str, fct)
            custom_def.register_kernel("cpu")(fct)
            custom_def._abstract_fn = fct_shape

        register(onnx_test_numpy_sin, lambda x: torch.empty_like(x), "mylib", "onnx_test_numpy_sin")

        class ModuleWithACustomOperator(torch.nn.Module):
            def forward(self, x):
                return torch.ops.mylib.onnx_test_numpy_sin(x)

        model = ModuleWithACustomOperator()
        torch.testing.assert_allclose(model(x), expected)
        ep = torch.export.export(model, (x,))
        for node in ep.graph.nodes:
            assert not isinstance(
                node.target, OpOverload
            ) or node.target == torch.ops.mylib.onnx_test_numpy_sin.default

        op = onnxscript.opset18


        @onnxscript.script()
        def onnx_test_numpy_sin_to_onnx(x):
            return op.Sin(x)

        ep = torch.onnx.export(
            model,
            (x,),
            custom_translation_table={
                torch.ops.mylib.onnx_test_numpy_sin.default: onnx_test_numpy_sin_to_onnx
            },
            dynamo=True,

        )
        onnx_testing.assert_onnx_program(ep)

    # TODO(justinchuby): Test multi-output HOPs

if __name__ == "__main__":
    common_utils.run_tests()
