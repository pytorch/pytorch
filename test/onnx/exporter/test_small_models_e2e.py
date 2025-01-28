# Owner(s): ["module: onnx"]
"""Unit tests for the onnx dynamo exporter."""

from __future__ import annotations

import logging

import torchvision
import transformers

import torch
from torch.onnx._internal.exporter import _testing as onnx_testing
from torch.testing._internal import common_utils


@common_utils.instantiate_parametrized_tests
class DynamoExporterTest(common_utils.TestCase):
    def export(self, model, args=(), kwargs=None, **options) -> torch.onnx.ONNXProgram:
        onnx_program = torch.onnx.export(
            model, args, kwargs=kwargs, dynamo=True, fallback=False, **options
        )
        assert onnx_program is not None
        return onnx_program

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

        onnx_program = self.export(
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

        onnx_program = self.export(MulModule(), (x,), dynamo=True)
        onnx_testing.assert_onnx_program(onnx_program)

    def test_pow_does_not_trigger_type_promotion(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                return x**2.0

        x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float16)

        onnx_program = self.export(Model(), (x,), dynamo=True)
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

        onnx_program = self.export(
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

        onnx_program = self.export(
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
        onnx_program = self.export(VisionModel(), args, dynamo=True)
        onnx_testing.assert_onnx_program(onnx_program)

    # TODO(justinchuby): Test multi-output HOPs

    def test_empty(self):
        def func(x):
            return torch.empty(x.size(), dtype=torch.int64)

        # Since `torch.empty` returns tensor with uninitialized data, we cannot
        # test this under `test_fx_to_onnx_with_onnxruntime.py` with result comparison.
        _ = self.export(func, (torch.randn(1, 2),))

    def test_multiple_outputs_op_with_evaluator(self):
        class TopKModel(torch.nn.Module):
            def forward(self, x):
                values, _ = torch.topk(x, 3)
                return torch.sum(values)

        onnx_program = self.export(
            TopKModel(), (torch.arange(1.0, 6.0, requires_grad=True),)
        )
        onnx_testing.assert_onnx_program(onnx_program)

    def test_exported_program_torch_distributions_normal_Normal(self):
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                self.normal = torch.distributions.normal.Normal(0, 1)
                super().__init__()

            def forward(self, x):
                return self.normal.sample(x.shape)

        with torch.no_grad():
            exported_program = torch.export.export(
                Model(), args=(torch.randn(2),), strict=False
            )
            _ = self.export(exported_program)

    @common_utils.parametrize(
        "float8_type",
        [
            common_utils.subtest(
                torch.float8_e5m2,
                name="torch_float8_e5m2",
            ),
            common_utils.subtest(
                torch.float8_e5m2fnuz,
                name="torch_float8_e5m2fnuz",
            ),
            common_utils.subtest(
                torch.float8_e4m3fn,
                name="torch_float8_e4m3fn",
            ),
            common_utils.subtest(
                torch.float8_e4m3fnuz,
                name="torch_float8_e4m3fnuz",
            ),
        ],
    )
    def test_float8_support(self, float8_type):
        class Float8Module(torch.nn.Module):
            def forward(self, input: torch.Tensor):
                input = input.to(float8_type)
                return input + torch.tensor(1.0, dtype=float8_type)

        _ = self.export(Float8Module(), (torch.randn(1, 2),))

    def test_export_with_logging_logger(self):
        logger = logging.getLogger(__name__)

        class LoggingLoggerModule(torch.nn.Module):
            def forward(self, x):
                logger.log("abc")
                return x + 1

        onnx_program = self.export(LoggingLoggerModule(), (torch.tensor(1),))
        onnx_testing.assert_onnx_program(onnx_program)

    def test_export_with_hf_logging_logger(self):
        logger = transformers.utils.logging.get_logger(__name__)

        class HFLoggingLoggerModule(torch.nn.Module):
            def forward(self, x):
                logger.warning_once("abc")
                return x + 1

        onnx_program = self.export(HFLoggingLoggerModule(), (torch.tensor(1),))
        onnx_testing.assert_onnx_program(onnx_program)

    def test_export_with_print(self):
        class PrintModule(torch.nn.Module):
            def forward(self, x):
                print("abc")
                return x + 1

        onnx_program = self.export(PrintModule(), (torch.tensor(1),))
        onnx_testing.assert_onnx_program(onnx_program)


if __name__ == "__main__":
    common_utils.run_tests()
