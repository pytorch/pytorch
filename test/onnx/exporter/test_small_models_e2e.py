# Owner(s): ["module: onnx"]
"""Unit tests for the onnx dynamo exporter."""

from __future__ import annotations

import logging

import pytest
import transformers
from onnxscript import ir

import torch
from torch.onnx._internal.exporter import _testing as onnx_testing
from torch.testing._internal import common_utils
from torch.utils import _pytree as torch_pytree


class _WithExport:
    def export(self, model, args=(), kwargs=None, **options) -> torch.onnx.ONNXProgram:
        if isinstance(model, torch.nn.Module):
            model = model.eval()
        onnx_program = torch.onnx.export(
            model,
            args,
            kwargs=kwargs,
            dynamo=True,
            fallback=False,
            verbose=False,
            **options,
        )
        assert onnx_program is not None
        return onnx_program


@common_utils.instantiate_parametrized_tests
class DynamoExporterTest(common_utils.TestCase, _WithExport):
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

        onnx_program = self.export(model, (query, key, value))
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

        onnx_program = self.export(MulModule(), (x,))
        onnx_testing.assert_onnx_program(onnx_program)

    def test_pow_does_not_trigger_type_promotion(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                return x**2.0

        x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float16)

        onnx_program = self.export(Model(), (x,))
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

        onnx_program = self.export(CondModel(), (torch.tensor([1, 2]),))
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

        onnx_program = self.export(CondModel(), (torch.tensor([1, 2]),))
        onnx_testing.assert_onnx_program(onnx_program)
        onnx_testing.assert_onnx_program(onnx_program, args=(torch.tensor([0, 0]),))
        onnx_testing.assert_onnx_program(onnx_program, args=(torch.tensor([43, 43]),))

    def test_onnx_export_control_flow_multi_outputs(self):
        class CondModel(torch.nn.Module):
            def forward(self, x):
                z = torch.ones_like(x)

                def true_fn(x, z):
                    x = x + 1.0
                    z = z * 1.0
                    return x, z

                def false_fn(x, z):
                    x = x - 1.0
                    z = z * 0.0
                    return x, z

                x = torch.cond(x.sum() > 0, true_fn, false_fn, (x, z))
                return x, z

        onnx_program = self.export(CondModel(), (torch.tensor([1, 2]),))
        onnx_testing.assert_onnx_program(onnx_program)
        onnx_testing.assert_onnx_program(onnx_program, args=(torch.tensor([-1, -2]),))

    def test_empty(self):
        class EmptyModel(torch.nn.Module):
            def forward(self, x):
                return torch.empty(x.size(), dtype=torch.int64)

        # Since `torch.empty` returns tensor with uninitialized data, we cannot
        # test this under `test_fx_to_onnx_with_onnxruntime.py` with result comparison.
        _ = self.export(EmptyModel(), (torch.randn(1, 2),))

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
        "float8_type, onnx_type",
        [
            common_utils.subtest(
                (torch.float8_e5m2, ir.DataType.FLOAT8E5M2),
                name="torch_float8_e5m2",
            ),
            common_utils.subtest(
                (torch.float8_e5m2fnuz, ir.DataType.FLOAT8E5M2FNUZ),
                name="torch_float8_e5m2fnuz",
            ),
            common_utils.subtest(
                (torch.float8_e4m3fn, ir.DataType.FLOAT8E4M3FN),
                name="torch_float8_e4m3fn",
            ),
            common_utils.subtest(
                (torch.float8_e4m3fnuz, ir.DataType.FLOAT8E4M3FNUZ),
                name="torch_float8_e4m3fnuz",
            ),
        ],
    )
    def test_float8_support(self, float8_type: torch.dtype, onnx_type: ir.DataType):
        class Float8Module(torch.nn.Module):
            def forward(self, input: torch.Tensor):
                input = input.to(float8_type)
                return input

        onnx_program = self.export(Float8Module(), (torch.randn(1, 2),))
        self.assertEqual(onnx_program.model.graph.outputs[0].dtype, onnx_type)

    def test_float4_support(self):
        class Float4Module(torch.nn.Module):
            def forward(self):
                return torch.empty([1], dtype=torch.float4_e2m1fn_x2)

        onnx_program = self.export(Float4Module(), optimize=False)
        output = onnx_program.model.graph.outputs[0]
        self.assertEqual(output.dtype, ir.DataType.FLOAT4E2M1)
        # The shape is [*shape[:-1], shape[-1]*2] because ONNX stores the shape of the unpacked tensor
        self.assertEqual(output.shape.numpy(), [2])

    def test_bfloat16_support(self):
        class BfloatModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # Test parameters
                self.param = torch.nn.Parameter(torch.tensor(2.0, dtype=torch.bfloat16))

            def forward(self, x):
                # Test constant tensors are stored as bfloat16
                const = torch.tensor(1.0, dtype=torch.bfloat16)
                return x * const * self.param

        input = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)
        onnx_program = self.export(BfloatModel(), (input,), optimize=False)
        initializers = onnx_program.model.graph.initializers.values()
        self.assertEqual(len(initializers), 2)
        for initializer in initializers:
            self.assertEqual(initializer.dtype, ir.DataType.BFLOAT16)
        self.assertEqual(onnx_program.model.graph.inputs[0].dtype, ir.DataType.BFLOAT16)
        self.assertEqual(
            onnx_program.model.graph.outputs[0].dtype, ir.DataType.BFLOAT16
        )

    def test_export_with_logging_logger(self):
        logger = logging.getLogger(__name__)

        class LoggingLoggerModule(torch.nn.Module):
            def forward(self, x):
                logger.info("abc")
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

    def test_export_with_dynamic_input(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                return x + 1.0

        dim0 = torch.export.Dim("dim0")
        onnx_program = self.export(
            Model(),
            (torch.randn(2, 3, 4, dtype=torch.float),),
            dynamic_shapes=({0: dim0},),
        )

        onnx_testing.assert_onnx_program(
            onnx_program, args=(torch.randn(3, 3, 4, dtype=torch.float),)
        )

    def test_export_with_specialized_input_during_tracing(self):
        class Model(torch.nn.Module):
            def forward(self, x, y):
                return x + y

        dim0_x = torch.export.Dim("dim0_x", min=6)
        dynamic_shapes = {"x": {0: dim0_x}, "y": torch.export.Dim.STATIC}
        # specialized input y to 5 during tracing
        onnx_program = self.export(
            Model(),
            (
                torch.ones(7, 5),
                5,
            ),
            dynamic_shapes=dynamic_shapes,
        )

        onnx_testing.assert_onnx_program(onnx_program, args=(torch.ones(8, 5), 5))

    def test_export_with_none_arg_name_in_dynamic(self):
        class Model(torch.nn.Module):
            def forward(self, a, b):
                return a.sum() + b.sum()

        dim = torch.export.Dim("dim")
        onnx_program = self.export(
            Model(),
            (
                torch.randn(4, 4),
                torch.randn(4, 4),
            ),
            dynamic_shapes=(None, {0: dim}),
        )

        test_inputs = (
            torch.randn(4, 4),
            torch.randn(7, 4),
        )
        onnx_testing.assert_onnx_program(onnx_program, args=test_inputs)

    def test_export_with_non_arg_name_with_kwarg(self):
        class Model(torch.nn.Module):
            def forward(self, a, b, kw1, kw2):
                return a.sum() + b.sum() + kw1.sum() - kw2.sum()

        dim = torch.export.Dim("dim")
        dim_for_kw1 = torch.export.Dim("dim_for_kw1")
        onnx_program = self.export(
            Model(),
            (torch.randn(4, 4), torch.randn(4, 4)),
            {"kw2": torch.ones(4, 4), "kw1": torch.zeros(4, 4)},
            # We are specifying dynamism on the first kwarg even though user passed in
            # different order
            dynamic_shapes=(None, {0: dim}, {0: dim_for_kw1}, None),
        )

        # This should work even if the kwarg order are flipped.
        onnx_testing.assert_onnx_program(
            onnx_program,
            args=(torch.randn(4, 4), torch.randn(7, 4)),
            kwargs={"kw2": torch.ones(4, 4), "kw1": torch.zeros(9, 4)},
        )

    def test_export_with_input_lifting_buffers_mutation(self):
        for persistent in (True, False):

            class CustomModule(torch.nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.register_buffer(
                        "my_buffer", torch.tensor(4.0), persistent=persistent
                    )

                def forward(self, x, b):
                    output = x + b
                    (
                        self.my_buffer.add_(1.0) + 3.0
                    )  # Mutate buffer through in-place addition
                    return output

            dim = torch.export.Dim("dim")
            onnx_program = self.export(
                CustomModule(),
                (
                    torch.rand((3, 3), dtype=torch.float32),
                    torch.randn(3, 3),
                ),
                dynamic_shapes=({0: dim}, {0: dim}),
            )

            onnx_testing.assert_onnx_program(
                onnx_program,
                args=(torch.rand((4, 3), dtype=torch.float32), torch.randn(4, 3)),
            )

    def test_export_with_non_arg_name_with_container_type(self):
        class Model(torch.nn.Module):
            def forward(self, a, b):
                return a[0].sum() + a[1].sum() + b.sum()

        count = 0

        def dynamify_inp(x):
            # Mark the second input a[1] dynamic
            nonlocal count
            if count == 1:
                dim = torch.export.Dim("dim", min=3)
                count += 1
                return {0: dim}
            count += 1
            return None

        dynamic_shapes = torch_pytree.tree_map(
            dynamify_inp,
            (
                (torch.randn(4, 4), torch.randn(4, 4)),
                torch.randn(4, 4),
            ),
        )
        onnx_program = self.export(
            Model(),
            (
                (torch.randn(4, 4), torch.randn(4, 4)),
                torch.randn(4, 4),
            ),
            dynamic_shapes=dynamic_shapes,
        )

        # NOTE: Careful with the input format. The input format should be
        # consistent with how the model is exported.
        onnx_testing.assert_onnx_program(
            onnx_program,
            args=((torch.randn(4, 4), torch.randn(6, 4)), torch.randn(4, 4)),
        )

    def test_export_with_lazy_module_kwargs(self):
        class LazyModule(torch.nn.modules.lazy.LazyModuleMixin, torch.nn.Module):
            def initialize_parameters(self, *args, **kwargs):
                pass

            def forward(self, x, y):
                return x + y

        m = LazyModule()
        dim = torch.export.Dim("dim")
        dynamic_shapes = ({0: dim}, {0: dim})
        onnx_program = self.export(
            m,
            (),
            {"x": torch.randn(3, 3), "y": torch.randn(3, 3)},
            dynamic_shapes=dynamic_shapes,
        )

        inputs = {"x": torch.randn(6, 3), "y": torch.randn(6, 3)}
        onnx_testing.assert_onnx_program(onnx_program, kwargs=inputs)

    def test_export_of_rename_dynamic_axes_required_model_with_mixed_type_of_dynamic_shapes(
        self,
    ):
        class NestedModel(torch.nn.Module):
            def forward(
                self,
                x: torch.Tensor,
                ys: list[torch.Tensor],
                zs: dict[str, torch.Tensor],
                c: torch.Tensor,
            ):
                y = ys[0] + ys[1] + zs["a"] + zs["b"]
                w = 5
                if x.shape[0] < 3 and c.shape[0] != 4:
                    return x + w, x + y, c
                else:
                    return x - w, x - y, c

        input = (
            torch.ones(5),
            [torch.zeros(5), torch.ones(5)],
            {"a": torch.zeros(5), "b": torch.ones(5)},
            torch.ones(6),
        )

        dynamic_shapes = (
            {0: torch.export.Dim("dim_x", min=3)},  # Dim
            [("custom_name_axis_ys_0",), (torch.export.Dim.AUTO,)],  # custom name
            {
                "a": {0: torch.export.Dim.AUTO},
                "b": ("custom_name_axis_zs_b_0",),
            },  # _DimHint
            {0: "custom_name_axis_c_0"},  # custom name
        )

        # 0. Export the model
        # 1. Assert the warning message
        with self.assertWarnsRegex(
            UserWarning,
            "# The axis name: .* will not be used, since it shares the same shape constraints with another axis: .*.",
        ):
            onnx_program = self.export(
                NestedModel(), input, dynamic_shapes=dynamic_shapes, optimize=False
            )
        # 2. Assert the exported model
        input = (
            torch.ones(4),
            [torch.zeros(4), torch.ones(4)],
            {"a": torch.zeros(4), "b": torch.ones(4)},
            torch.ones(5),
        )
        onnx_testing.assert_onnx_program(
            onnx_program,
            args=input,
        )
        # 3. Assert the dynamic axes names
        # Some names are not respected because they share the same shape constraints,
        # so they are the same to ExportedProgram.
        expected_axis_names = [
            "dim_x",
            "dim_x",
            "dim_x",
            "dim_x",
            "dim_x",
            "custom_name_axis_c_0",
        ]

        for expected_axis_name, input in zip(
            expected_axis_names, onnx_program.model.graph.inputs
        ):
            self.assertEqual(input.shape[0].value, expected_axis_name)

    def test_export_of_static_dim_constraints(self):
        # NOTE: This test is to ensure that the static dim constraints are respected.
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.l = torch.nn.Linear(6, 4)

            def forward(self, x, y, z):
                x0 = self.l(x) + y[1:]
                return x0, z * 2.0

        inputs = (torch.randn(4, 6), torch.randn(5, 4), torch.randn(3, 3))
        dx = torch.export.Dim("dx", min=3, max=6)
        dy = dx + 1
        dz = torch.export.Dim("dz", min=3, max=6)

        # all of these should be fine
        dynamic_shapes = (
            {0: dx, 1: torch.export.Dim.AUTO},
            {0: dy, 1: torch.export.Dim.STATIC},
            {0: dz, 1: 3},
        )
        onnx_program = self.export(Model(), inputs, dynamic_shapes=dynamic_shapes)
        onnx_testing.assert_onnx_program(onnx_program)
        # make sre the naming is working
        self.assertEqual(onnx_program.model.graph.inputs[0].shape[0], "dx")

    def test_export_sym_max(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                return torch.sym_max(*x.shape)

        inputs = (torch.zeros((2, 3)),)
        dynamic_shapes = ({0: torch.export.Dim.DYNAMIC, 1: torch.export.Dim.DYNAMIC},)
        onnx_program = self.export(Model(), inputs, dynamic_shapes=dynamic_shapes)
        onnx_testing.assert_onnx_program(onnx_program)
        self.assertIn(
            "Max",
            [node.op_type for node in onnx_program.model.graph],
        )

    def test_export_sym_min(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                return torch.sym_min(*x.shape)

        inputs = (torch.zeros((2, 3)),)
        dynamic_shapes = ({0: torch.export.Dim.DYNAMIC, 1: torch.export.Dim.DYNAMIC},)
        onnx_program = self.export(Model(), inputs, dynamic_shapes=dynamic_shapes)
        onnx_testing.assert_onnx_program(onnx_program)
        self.assertIn(
            "Min",
            [node.op_type for node in onnx_program.model.graph],
        )

    def test_export_sym_not(self):
        class SymNotModel(torch.nn.Module):
            def forward(self, x):
                comparison = x.shape[0] == x.shape[1]
                return torch.sym_not(comparison)

        inputs = (torch.zeros((2, 2)),)
        dynamic_shapes = ({0: torch.export.Dim.DYNAMIC, 1: torch.export.Dim.DYNAMIC},)
        onnx_program = self.export(SymNotModel(), inputs, dynamic_shapes=dynamic_shapes)
        onnx_testing.assert_onnx_program(onnx_program)
        self.assertIn(
            "Not",
            [node.op_type for node in onnx_program.model.graph],
        )

    def test_export_sym_float(self):
        class SymFloatModel(torch.nn.Module):
            def forward(self, x):
                a = x.shape[0]
                return torch.sym_float(a)

        inputs = (torch.zeros((2, 2)),)
        dynamic_shapes = ({0: torch.export.Dim.DYNAMIC, 1: torch.export.Dim.DYNAMIC},)
        onnx_program = self.export(
            SymFloatModel(), inputs, dynamic_shapes=dynamic_shapes
        )
        onnx_testing.assert_onnx_program(onnx_program)
        self.assertIn(
            "Cast",
            [node.op_type for node in onnx_program.model.graph],
        )

    def test_scan_cdist_add(self):
        def dist(unused: torch.Tensor, x: torch.Tensor, samex: torch.Tensor):
            sub = samex - x.reshape((1, -1))
            sq = sub * sub
            rd = torch.sqrt(sq.sum(axis=1))
            return [unused.clone(), rd]

        class ScanModel(torch.nn.Module):
            def forward(self, x):
                z = torch.tensor([0], dtype=torch.float32)
                y = x.clone()
                out = torch.ops.higher_order.scan(dist, [z], [x], additional_inputs=[y])
                return out[1]

        inputs = (
            torch.tensor(
                [[1, 2, 3, -1], [4, 5, 6, -1], [7, 8, 9, -1]], dtype=torch.float32
            ),
        )
        onnx_program = self.export(ScanModel(), inputs)
        onnx_testing.assert_onnx_program(onnx_program)

    def test_scan_cdist_dynamic_shapes(self):
        def dist(y: torch.Tensor, scanned_x: torch.Tensor):
            sub = y - scanned_x.reshape((1, -1))
            sq = sub * sub
            rd = torch.sqrt(sq.sum(axis=1))
            return [y.clone(), rd]

        class ScanModel(torch.nn.Module):
            def forward(self, x, y):
                carry, out = torch.ops.higher_order.scan(
                    dist, [y], [x], additional_inputs=[]
                )
                return out

        x_rows = torch.export.Dim("x_rows")
        y_rows = torch.export.Dim("y_rows")
        dim = torch.export.Dim("dim")
        inputs = (torch.randn(3, 4), torch.randn(5, 4))
        onnx_program = self.export(
            ScanModel(),
            inputs,
            dynamic_shapes=({0: x_rows, 1: dim}, {0: y_rows, 1: dim}),
        )
        onnx_testing.assert_onnx_program(onnx_program)

    @pytest.mark.xfail(reason="Data dependent error.")
    def test_scan_loop_inplace(self):
        def dummy_loop(padded: torch.Tensor, pos: torch.Tensor):
            copy = torch.zeros(padded.shape)
            for i in range(pos.shape[0]):
                p = pos[i]
                copy[i, :p] = padded[i, :p]
            return copy

        def dummy_loop_with_scan(padded: torch.Tensor, pos: torch.Tensor):
            def pad_row(padded, p):
                row = torch.zeros((padded.shape[0],))
                torch._check(p.item() > 0)
                torch._check(p.item() < padded.shape[0])
                # this check is not always true, we add it anyway to make this dimension >= 2
                # and avoid raising an exception about dynamic dimension in {0, 1}
                if torch.compiler.is_exporting():
                    torch._check(p.item() > 1)
                row[: p.item()] = padded[: p.item()]
                return (row,)

            return torch.ops.higher_order.scan(pad_row, [], [padded, pos], [])

        def select_when_exporting(f, f_scan):
            return f_scan if torch.compiler.is_exporting() else f

        class ScanModel(torch.nn.Module):
            def forward(self, images, position):
                return select_when_exporting(dummy_loop, dummy_loop_with_scan)(
                    images, position
                )

        DYN = torch.export.Dim.DYNAMIC
        x = torch.randn((5, 6))
        y = torch.arange(5, dtype=torch.int64) + 1
        ep = torch.export.export(
            ScanModel(),
            (x, y),
            dynamic_shapes={"images": {0: DYN, 1: DYN}, "position": {0: DYN}},
            strict=False,
        )
        onnx_program = self.export(ep)
        onnx_testing.assert_onnx_program(onnx_program)


@common_utils.instantiate_parametrized_tests
class DynamoExporterNewOpsetsTest(common_utils.TestCase, _WithExport):
    def test_group_norm_opset_21(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                return torch.nn.functional.group_norm(x, 4)

        x = torch.randn(1, 4, 4, 4, dtype=torch.float32)
        onnx_program = self.export(Model(), (x,), opset_version=21)
        # TODO(after ort support): As of ONNX Runtime 1.22, the operator is not implemented yet.
        # call assert_onnx_program after ort support
        self.assertIn(
            "GroupNormalization",
            [node.op_type for node in onnx_program.model.graph],
        )

    def test_attention_opset_23(self):
        class Model(torch.nn.Module):
            def forward(self, query, key, value):
                return torch.nn.functional.scaled_dot_product_attention(
                    query, key, value
                )

        query = torch.rand(32, 8, 128, 64, dtype=torch.float16)
        key = torch.rand(32, 8, 128, 64, dtype=torch.float16)
        value = torch.rand(32, 8, 128, 64, dtype=torch.float16)

        onnx_program = self.export(Model(), (query, key, value), opset_version=23)
        self.assertEqual(["Attention"], [n.op_type for n in onnx_program.model.graph])

        onnx_testing.assert_onnx_program(onnx_program, atol=1e-2, rtol=1)

    def test_rms_norm(self):
        """Test RMS normalization with various configurations."""

        class RMSNormModel(torch.nn.Module):
            def forward(self, x):
                return torch.nn.functional.rms_norm(x, [3])

        x = torch.randn(2, 5, 3)
        onnx_program = self.export(RMSNormModel(), (x,), opset_version=23)
        onnx_testing.assert_onnx_program(onnx_program)

        # Test with multi-dimensional normalized_shape
        class RMSNormModel2D(torch.nn.Module):
            def forward(self, x):
                return torch.nn.functional.rms_norm(x, [7, 3])

        x = torch.randn(2, 5, 7, 3)
        onnx_program = self.export(RMSNormModel2D(), (x,), opset_version=23)
        onnx_testing.assert_onnx_program(onnx_program)

    def test_rms_norm_with_weight(self):
        """Test RMS normalization with weight parameter."""

        class RMSNormWithWeight(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(3))

            def forward(self, x):
                return torch.nn.functional.rms_norm(x, [3], weight=self.weight)

        x = torch.randn(2, 5, 3)

        onnx_program = self.export(RMSNormWithWeight(), (x,), opset_version=23)

        onnx_testing.assert_onnx_program(onnx_program)

    def test_rms_norm_with_eps(self):
        """Test RMS normalization with custom epsilon."""

        class RMSNormWithEps(torch.nn.Module):
            def forward(self, x):
                return torch.nn.functional.rms_norm(x, [3], eps=1e-5)

        x = torch.randn(2, 5, 3)

        onnx_program = self.export(RMSNormWithEps(), (x,), opset_version=23)

        onnx_testing.assert_onnx_program(onnx_program)

    def test_enable_gqa_in_attention_23_with_dropout(self):
        class Model(torch.nn.Module):
            def forward(self, q, k, v):
                return torch.nn.functional.scaled_dot_product_attention(  # pylint: disable=not-callable
                    q, k, v, enable_gqa=True, dropout_p=0.1
                )

        model = Model()

        query = torch.randn(2, 4, 8, 16)
        key = torch.randn(2, 2, 8, 16)
        value = torch.randn(2, 2, 8, 16)

        onnx_program = self.export(
            model,
            (
                query,
                key,
                value,
            ),
            opset_version=23,
        )
        # opset23 only uses manually gqa path when dropout is enabled,
        # and dropout makes the output non-deterministic,
        # so we check for the presence of the ops used in that path.
        all_ops = [node.op_type for node in onnx_program.model.graph]
        self.assertIn("Unsqueeze", all_ops)
        self.assertIn("Expand", all_ops)
        self.assertIn("Reshape", all_ops)


if __name__ == "__main__":
    common_utils.run_tests()
