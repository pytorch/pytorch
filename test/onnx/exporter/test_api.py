# Owner(s): ["module: onnx"]
"""Simple API tests for the ONNX exporter."""

from __future__ import annotations

import contextlib
import io
import logging
import os
import warnings
from unittest import mock

import sympy
from onnxscript import FLOAT, opset18 as op

import torch
from torch.onnx._internal.exporter import (
    _fx_passes,
    _registration,
    _specialization,
    _testing as onnx_testing,
)
from torch.onnx._internal.exporter._errors import DynamicShapeSpecializationWarning
from torch.testing._internal import common_utils
from torch.testing._internal.common_utils import HardwareClassification
from torch.utils._sympy.functions import FloorDiv, PythonMod


class SampleModel(torch.nn.Module):
    def forward(self, x):
        y = x + 1
        z = y.relu()
        return (y, z)


class SampleModelTwoInputs(torch.nn.Module):
    def forward(self, x, b):
        y = x + b
        z = y.relu()
        return (y, z)


class SampleModelReduction(torch.nn.Module):
    def forward(self, x):
        return x.sum()


class SampleModelForDynamicShapes(torch.nn.Module):
    def forward(self, x, b):
        return x.relu(), b.sigmoid()


class NestedModelForDynamicShapes(torch.nn.Module):
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


class SampleModelForDimOne(torch.nn.Module):
    def forward(self, x, y, z):
        return torch.cat((x, y), axis=1) + z


class TestExportAPIDynamo(common_utils.TestCase):
    """Tests for the ONNX exporter API when dynamo=True."""

    hw_classification = HardwareClassification.GENERIC

    def test_verbose_export_with_cp1252_stdout(self):
        output = io.BytesIO()
        with io.TextIOWrapper(output, encoding="cp1252", errors="strict") as stdout:
            with contextlib.redirect_stdout(stdout):
                onnx_program = torch.onnx.export(
                    SampleModel(),
                    (torch.randn(2),),
                    io.BytesIO(),
                    dynamo=True,
                    verbose=True,
                )
            stdout.flush()
            progress = output.getvalue().decode("cp1252")

        self.assertIsNotNone(onnx_program)
        self.assertIn("Obtain model graph", progress)
        self.assertIn("strict=False)`... [OK]", progress)
        self.assertIn("Run decompositions... [OK]", progress)
        self.assertIn("Translate the graph into ONNX... [OK]", progress)

    def assert_export(
        self, *args, strategy: str | None = "TorchExportNonStrictStrategy", **kwargs
    ):
        onnx_program = torch.onnx.export(*args, **kwargs, dynamo=True, verbose=False)
        if onnx_program is None:
            raise AssertionError("onnx_program is None")
        onnx_testing.assert_onnx_program(onnx_program, strategy=strategy)
        return onnx_program

    def test_args_normalization_with_no_kwargs(self):
        self.assert_export(
            SampleModelTwoInputs(),
            (torch.randn(1, 1, 2), torch.randn(1, 1, 2)),
        )

    def test_lower_opset_support(self):
        # First test that opset 18 (torchlib opset works)
        onnx_program = self.assert_export(
            SampleModelReduction(), (torch.randn(1, 1, 2),), opset_version=18
        )
        self.assertEqual(onnx_program.model.opset_imports[""], 18)

        onnx_program = self.assert_export(
            SampleModelReduction(), (torch.randn(1, 1, 2),), opset_version=16
        )
        self.assertEqual(onnx_program.model.opset_imports[""], 16)

    def test_symbolic_argument_user_input_is_supported_by_report_and_call(self):
        class constant_plus_tensor_inputs(torch.nn.Module):
            def forward(self, a, x):
                return a + torch.tensor(1) + x

        # Capture log output
        log_capture = io.StringIO()
        log_handler = logging.StreamHandler(log_capture)
        log_handler.setLevel(logging.ERROR)
        # Get the logger used in _core.py
        logger = logging.getLogger("torch.onnx._internal.exporter._core")
        original_level = logger.level
        logger.addHandler(log_handler)
        logger.setLevel(logging.ERROR)

        try:
            with common_utils.TemporaryDirectoryName() as temp_dir:
                self.assert_export(
                    constant_plus_tensor_inputs(),
                    (
                        1,
                        torch.ones(2),
                    ),
                    dynamic_shapes=(
                        torch.export.Dim.DYNAMIC,
                        {0: torch.export.Dim.DYNAMIC},
                    ),
                    report=True,
                    artifacts_dir=temp_dir,
                )
                # Check if the expected error was logged
                log_output = log_capture.getvalue()
                self.assertNotIn("Failed to save report due to an error", log_output)
                self.assertNotIn("KeyError: 'tensor_meta'", log_output)
                # Note: We don't call assert_onnx_program here because it will fail
                # due to the input name mismatch issue mentioned in your error

        finally:
            # Clean up logging
            logger.removeHandler(log_handler)
            logger.setLevel(original_level)

    def test_constant_argument_user_input_is_omitted_in_onnx_graph(self):
        class constant_plus_tensor_inputs(torch.nn.Module):
            def forward(self, a, x):
                return a + torch.tensor(1) + x

        onnx_program = torch.onnx.export(
            constant_plus_tensor_inputs(),
            (
                1,
                torch.ones(2),
            ),
            dynamic_shapes=(
                None,
                {0: torch.export.Dim.DYNAMIC},
            ),
            dynamo=True,
        )

        self.assertEqual(len(onnx_program.model.graph.inputs), 1)

    def test_dynamic_axes_enable_dynamic_shapes_with_fully_specified_axes(self):
        self.assert_export(
            SampleModelForDynamicShapes(),
            (torch.randn(2, 2, 3), {"b": torch.randn(2, 2, 3)}),
            dynamic_axes={
                "x": {0: "customx_dim_0", 1: "customx_dim_1", 2: "customx_dim_2"},
                "b": {0: "customb_dim_0", 1: "customb_dim_1", 2: "customb_dim_2"},
            },
        )

    def test_dynamic_axes_enable_dynamic_shapes_with_default_axe_names(self):
        self.assert_export(
            SampleModelForDynamicShapes(),
            (torch.randn(2, 2, 3), {"b": torch.randn(2, 2, 3)}),
            dynamic_axes={
                "x": [0, 1, 2],
                "b": [0, 1, 2],
            },
        )

    def test_dynamic_axes_supports_partial_dynamic_shapes(self):
        self.assert_export(
            SampleModelForDynamicShapes(),
            (torch.randn(2, 2, 3), {"b": torch.randn(2, 2, 3)}),
            input_names=["x", "b"],
            dynamic_axes={
                "b": [0, 1, 2],
            },
        )

    def test_dynamic_axes_supports_output_names(self):
        self.assert_export(
            SampleModelForDynamicShapes(),
            (torch.randn(2, 2, 3), {"b": torch.randn(2, 2, 3)}),
            input_names=["x", "b"],
            dynamic_axes={
                "b": [0, 1, 2],
            },
        )
        self.assert_export(
            SampleModelForDynamicShapes(),
            (
                torch.randn(2, 2, 3),
                torch.randn(2, 2, 3),
            ),
            input_names=["x", "b"],
            output_names=["x_out", "b_out"],
            dynamic_axes={"b": [0, 1, 2], "b_out": [0, 1, 2]},
        )

    def test_from_dynamic_axes_to_dynamic_shapes_deprecation_warning(self):
        with self.assertWarnsRegex(
            DeprecationWarning,
            "from_dynamic_axes_to_dynamic_shapes is deprecated and will be removed in a future release. "
            "This function converts 'dynamic_axes' format \\(including custom axis names\\) to 'dynamic_shapes' format. "
            "Instead of relying on this conversion, provide 'dynamic_shapes' directly with custom names.",
        ):
            self.assert_export(
                SampleModelForDynamicShapes(),
                (torch.randn(2, 2, 3), {"b": torch.randn(2, 2, 3)}),
                dynamic_axes={
                    "x": [0, 1, 2],
                    "b": [0, 1, 2],
                },
            )

    def test_from_dynamic_axes_to_dynamic_shapes_keeps_custom_axis_names(self):
        model = SampleModelForDynamicShapes()
        input = (
            torch.randn(2, 2, 3),
            {"b": torch.randn(2, 2, 3)},
        )
        dynamic_axes = {
            "x": {0: "customx_x_0", 1: "customx_x_1", 2: "customx_x_2"},
            "b": {0: "customb_b_0", 1: "customb_b_1", 2: "customb_b_2"},
            "x_out": {0: "customx_out_x_0", 1: "customx_out_x_1", 2: "customx_out_x_2"},
            "b_out": {0: "customb_out_b_0", 1: "customb_out_b_1", 2: "customb_out_b_2"},
        }
        onnx_program = torch.onnx.export(
            model,
            input,
            dynamic_axes=dynamic_axes,
            input_names=["x", "b"],
            output_names=["x_out", "b_out"],
            dynamo=True,
        )

        # Check whether the dynamic dimension names are preserved
        self.assertIs(onnx_program.model.graph.inputs[0].shape[0].value, "customx_x_0")
        self.assertIs(onnx_program.model.graph.inputs[0].shape[1].value, "customx_x_1")
        self.assertIs(onnx_program.model.graph.inputs[0].shape[2].value, "customx_x_2")
        self.assertIs(onnx_program.model.graph.inputs[1].shape[0].value, "customb_b_0")
        self.assertIs(onnx_program.model.graph.inputs[1].shape[1].value, "customb_b_1")
        self.assertIs(onnx_program.model.graph.inputs[1].shape[2].value, "customb_b_2")

    def test_saved_f_exists_after_export(self):
        with common_utils.TemporaryFileName(suffix=".onnx") as path:
            _ = torch.onnx.export(
                SampleModel(), (torch.randn(1, 1, 2),), path, dynamo=True
            )
            self.assertTrue(os.path.exists(path))

    def test_dynamic_shapes_with_fully_specified_axes(self):
        ep = torch.export.export(
            SampleModelForDynamicShapes(),
            (
                torch.randn(2, 2, 3),
                torch.randn(2, 2, 3),
            ),
            dynamic_shapes={
                "x": {
                    0: torch.export.Dim("customx_dim_0"),
                    1: torch.export.Dim("customx_dim_1"),
                    2: torch.export.Dim("customx_dim_2"),
                },
                "b": {
                    0: torch.export.Dim("customb_dim_0"),
                    1: torch.export.Dim("customb_dim_1"),
                    2: torch.export.Dim("customb_dim_2"),
                },
            },
            strict=True,
        )

        self.assert_export(ep, strategy=None)

    def test_partial_dynamic_shapes(self):
        self.assert_export(
            SampleModelForDynamicShapes(),
            (
                torch.randn(2, 2, 3),
                torch.randn(2, 2, 3),
            ),
            dynamic_shapes={
                "x": None,
                "b": {
                    0: torch.export.Dim("customb_dim_0"),
                    1: torch.export.Dim("customb_dim_1"),
                    2: torch.export.Dim("customb_dim_2"),
                },
            },
        )

    def test_dynamic_shapes_supports_nested_input_model_with_input_names_assigned(self):
        # kwargs can still be renamed as long as it's in order
        input_names = ["input_x", "input_y", "input_z", "d", "e", "f"]

        dynamic_axes = {
            "input_x": {0: "dim"},
            "input_y": {0: "dim"},
            "input_z": {0: "dim"},
            "d": {0: "dim"},
            "e": {0: "dim"},
        }

        model = NestedModelForDynamicShapes()
        input = (
            torch.ones(5),
            [torch.zeros(5), torch.ones(5)],
            {"a": torch.zeros(5), "b": torch.ones(5)},
            torch.ones(4),
        )

        self.assert_export(
            model, input, dynamic_axes=dynamic_axes, input_names=input_names
        )

        # Check whether inputs are dynamically shaped
        onnx_program = torch.onnx.export(
            model,
            input,
            dynamic_axes=dynamic_axes,
            input_names=input_names,
            dynamo=True,
        )
        self.assertTrue(
            all(
                [
                    input.type.tensor_type.shape.dim[0].dim_param
                    for input in onnx_program.model_proto.graph.input
                ][:-1]
            )
        )

    def test_upgraded_torchlib_impl(self):
        class GeluModel(torch.nn.Module):
            def forward(self, input):
                # Use GELU activation function
                return torch.nn.functional.gelu(input, approximate="tanh")

        input = (torch.randn(1, 3, 4, 4),)
        onnx_program_op18 = torch.onnx.export(
            GeluModel(),
            input,
            opset_version=18,
            dynamo=True,
        )
        all_nodes_op18 = [n.op_type for n in onnx_program_op18.model.graph]
        self.assertIn("Tanh", all_nodes_op18)
        self.assertNotIn("Gelu", all_nodes_op18)

        onnx_program_op20 = torch.onnx.export(
            GeluModel(),
            input,
            opset_version=20,
            dynamo=True,
        )
        all_nodes_op20 = [n.op_type for n in onnx_program_op20.model.graph]
        self.assertIn("Gelu", all_nodes_op20)

    def test_refine_dynamic_shapes_with_onnx_export(self):
        # NOTE: From test/export/test_export.py

        # refine lower, upper bound
        class TestRefineDynamicShapeModel(torch.nn.Module):
            def forward(self, x, y):
                if x.shape[0] >= 6 and y.shape[0] <= 16:
                    return x * 2.0, y + 1

        inps = (torch.randn(16), torch.randn(12))
        dynamic_shapes = {
            "x": (torch.export.Dim("dx"),),
            "y": (torch.export.Dim("dy"),),
        }
        self.assert_export(
            TestRefineDynamicShapeModel(), inps, dynamic_shapes=dynamic_shapes
        )

    def test_zero_output_aten_node(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                torch.ops.aten._assert_async.msg(torch.tensor(True), "assertion failed")
                return x + x

        input = torch.randn(2)
        self.assert_export(Model(), (input))

    def test_export_successful_when_dynamic_dimension_is_one(self):
        self.assert_export(
            SampleModelForDimOne(),
            (torch.randn(1, 3), torch.randn(1, 5), torch.randn(1, 8)),
            dynamic_shapes=(
                {0: "batch", 1: "sequence"},
                {0: "batch", 1: "sequence"},
                {0: "batch", 1: "sequence"},
            ),
        )

    def test_is_in_onnx_export(self):
        class Mod(torch.nn.Module):
            def forward(self, x):
                def f(x):
                    return x.sin() if torch.onnx.is_in_onnx_export() else x.cos()

                return f(x)

        self.assertFalse(torch.onnx.is_in_onnx_export())
        onnx_program = torch.onnx.export(
            Mod(),
            (torch.randn(3, 4),),
            dynamo=True,
        )
        self.assertFalse(torch.onnx.is_in_onnx_export())

        node_names = [n.op_type for n in onnx_program.model.graph]
        self.assertIn("Sin", node_names)

    def test_torchscript_exporter_raises_deprecation_warning(self):
        # Test that the deprecation warning is raised when using torchscript exporter
        with self.assertWarnsRegex(
            DeprecationWarning, "You are using the legacy TorchScript-based ONNX export"
        ):
            torch.onnx.export(
                SampleModel(), (torch.randn(1, 1, 2),), io.BytesIO(), dynamo=False
            )

    def test_model_output_can_be_none(self):
        class ModelWithNoneOutput(torch.nn.Module):
            def forward(self, x):
                return x + 1, None

        onnx_program = torch.onnx.export(
            ModelWithNoneOutput(),
            (torch.randn(1, 1, 2),),
            dynamo=True,
        )
        onnx_testing.assert_onnx_program(onnx_program)


class TestCustomTranslationTable(common_utils.TestCase):
    hw_classification = HardwareClassification.GENERIC

    def test_custom_translation_table_overrides_ops(self):
        from onnxscript import opset18 as op

        class Model(torch.nn.Module):
            def forward(self, x, y):
                return x + y

        def custom_add(self, other):
            # Replace add with sub
            return op.Sub(self, other)

        custom_translation_table = {torch.ops.aten.add.Tensor: custom_add}

        onnx_program = torch.onnx.export(
            Model(),
            (torch.randn(2, 2), torch.randn(2, 2)),
            custom_translation_table=custom_translation_table,
            dynamo=True,
        )
        all_nodes = [n.op_type for n in onnx_program.model.graph]
        self.assertIn("Sub", all_nodes)
        self.assertNotIn("Add", all_nodes)

    def test_custom_translation_table_supports_custom_op_as_target(self):
        # Define the custom op and use it in the model
        @torch.library.custom_op("custom::add", mutates_args=())
        def custom_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return a + b

        @custom_add.register_fake
        def _(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return torch.empty_like(a) + torch.empty_like(b)

        class Model(torch.nn.Module):
            def forward(self, x, y):
                return custom_add(x, y)

        def onnx_add(self: FLOAT, other: FLOAT) -> FLOAT:
            # Replace add with Sub
            return op.Sub(self, other)

        custom_translation_table = {
            torch.ops.custom.add.default: onnx_add,
        }

        onnx_program = torch.onnx.export(
            Model(),
            (torch.tensor(1, dtype=torch.bool), torch.tensor(1, dtype=torch.bool)),
            custom_translation_table=custom_translation_table,
            dynamo=True,
        )
        all_nodes = [n.op_type for n in onnx_program.model.graph]
        self.assertIn("Sub", all_nodes)
        self.assertNotIn("Add", all_nodes)

    def test_custom_translation_table_supports_custom_op_with_its_decomp(self):
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            torch.library.define(
                "mylib::foo",
                "(Tensor a, Tensor b) -> Tensor",
                tags=torch.Tag.pt2_compliant_tag,
                lib=lib,
            )

            @torch.library.impl("mylib::foo", "CompositeImplicitAutograd", lib=lib)
            @torch.library.register_fake("mylib::foo")
            def foo_impl(a, b):
                return a + b

            class M(torch.nn.Module):
                def forward(self, x, y):
                    return torch.ops.mylib.foo(x, y)

            def onnx_add(self: FLOAT, other: FLOAT) -> FLOAT:
                # Replace add with Sub
                return op.Sub(self, other)

            # With the custom op defined, we can use it in the model
            # and replace it with a custom translation table
            custom_translation_table = {
                torch.ops.mylib.foo.default: onnx_add,
            }
            onnx_program = torch.onnx.export(
                M(),
                (torch.ones(3, 3), torch.ones(3, 3)),
                custom_translation_table=custom_translation_table,
                dynamo=True,
            )
            all_nodes = [n.op_type for n in onnx_program.model.graph]
            self.assertIn("Sub", all_nodes)
            self.assertNotIn("Add", all_nodes)

            # Without the custom op defined, it's going to be decomposed
            onnx_program_decomp = torch.onnx.export(
                M(), (torch.ones(3, 3), torch.ones(3, 3)), dynamo=True
            )
            all_nodes_decomp = [n.op_type for n in onnx_program_decomp.model.graph]
            self.assertIn("Add", all_nodes_decomp)
            self.assertNotIn("Sub", all_nodes_decomp)

    def test_01_specialization_with_run_decomp_is_supported(self):
        # Phi3RMSNorm changes and redo shape inference after `run_decompositions` call
        # We need this test to make sure everything we do on fx graph is covered by
        # backed_size_oblivious
        class Phi3RMSNorm(torch.nn.Module):
            def __init__(self, hidden_size, eps=1e-6):
                """
                Phi3RMSNorm is equivalent to T5LayerNorm
                """
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(hidden_size))
                self.variance_epsilon = eps

            def forward(self, hidden_states):
                input_dtype = hidden_states.dtype
                hidden_states = hidden_states.to(torch.float32)
                variance = hidden_states.pow(2).mean(-1, keepdim=True)
                hidden_states = hidden_states * torch.rsqrt(
                    variance + self.variance_epsilon
                )
                return self.weight * hidden_states.to(input_dtype)

        op = torch.onnx.export(
            Phi3RMSNorm(256).eval(),
            args=(),
            kwargs={"hidden_states": torch.rand((1, 32, 256))},
            dynamic_shapes={
                "hidden_states": {
                    0: "batch_size",
                    1: "seq_len",
                }
            },
            dynamo=True,
        )
        # batch size is not fixed to 1
        self.assertNotEqual(op.model.graph.outputs[0].shape[0], 1)


@common_utils.instantiate_parametrized_tests
class TestDynamicShapeSpecialization(common_utils.TestCase):
    hw_classification = HardwareClassification.GENERIC

    def test_adaptive_max_pool2d(self):
        # This relies on torchlib not implementing adaptive_max_pool2d; the
        # custom-op tests provide coverage that does not depend on torchlib.
        class Model(torch.nn.Module):
            def forward(self, x):
                return torch.nn.functional.adaptive_max_pool2d(x, (2, 2))

        with self.assertWarns(DynamicShapeSpecializationWarning) as caught:
            torch.onnx.export(
                Model(),
                (torch.randn(1, 2, 6, 6),),
                dynamic_shapes=(
                    {2: torch.export.Dim.DYNAMIC, 3: torch.export.Dim.DYNAMIC},
                ),
                verbose=False,
            )
        message = str(caught.warning)
        self.assertIn("x.size()[2]", message)
        self.assertIn("x.size()[3]", message)
        self.assertIn("aten.adaptive_max_pool2d", message)
        self.assertIn("aten.adaptive_max_pool2d.default: Eq(", message)
        self.assertNotIn("may be responsible", message)
        self.assertNotIn("1 <=", message)
        expected = "  aten.adaptive_max_pool2d.default: Eq((s0//2), 3); Eq((s53//2), 3); Eq(Mod(s0, 2), 0); Eq(Mod(s53, 2), 0)"
        self.assertIn(expected, message.splitlines())
        self.assertIn("will give wrong results (or fail) for other sizes", message)

    def test_custom_specializing_decomposition(self):
        with torch.library._scoped_library("specialization_test", "FRAGMENT") as lib:
            lib.define(
                "specialize(Tensor x) -> Tensor", tags=torch.Tag.pt2_compliant_tag
            )
            lib.impl("specialize", lambda x: x + x.shape[0] // 2, "CPU")
            torch.library.register_fake(
                "specialization_test::specialize",
                lambda x: torch.empty_like(x),
                lib=lib,
            )
            target = torch.ops.specialization_test.specialize.default

            def decompose(x):
                return x + int(x.shape[0] // 2)

            class Model(torch.nn.Module):
                def forward(self, x):
                    return target(x)

            ep = torch.export.export(
                Model(),
                (torch.randn(6, 2),),
                dynamic_shapes=({0: torch.export.Dim.DYNAMIC},),
                strict=False,
            )
            with (
                mock.patch.dict(torch._decomp.decomposition_table, {target: decompose}),
                self.assertWarns(DynamicShapeSpecializationWarning) as caught,
            ):
                torch.onnx.export(ep, verbose=False)
            message = str(caught.warning)
            self.assertIn("x.size()[0]", message)
            self.assertIn("specialization_test.specialize.default", message)
            self.assertIn("Constraints added while decomposing", message)

    def test_op_inside_cond_branch_is_attributed(self):
        with torch.library._scoped_library("specialization_test", "FRAGMENT") as lib:
            lib.define(
                "specialize(Tensor x) -> Tensor", tags=torch.Tag.pt2_compliant_tag
            )
            lib.impl("specialize", lambda x: x + x.shape[0] // 2, "CPU")
            torch.library.register_fake(
                "specialization_test::specialize",
                lambda x: torch.empty_like(x),
                lib=lib,
            )
            target = torch.ops.specialization_test.specialize.default

            def decompose(x):
                return x + int(x.shape[0] // 2)

            class Model(torch.nn.Module):
                def forward(self, x):
                    return torch.cond(
                        x.sum() > 0, lambda x: target(x), lambda x: x.clone(), (x,)
                    )

            ep = torch.export.export(
                Model(),
                (torch.randn(6, 2),),
                dynamic_shapes=({0: torch.export.Dim.DYNAMIC},),
                strict=False,
            )
            create_table = _fx_passes._decomp.create_onnx_friendly_decomposition_table
            with (
                mock.patch.object(
                    _fx_passes._decomp,
                    "create_onnx_friendly_decomposition_table",
                    side_effect=lambda ops: create_table(ops) | {target: decompose},
                ),
                self.assertWarns(DynamicShapeSpecializationWarning) as caught,
            ):
                torch.onnx.export(ep, verbose=False)
        message = str(caught.warning)
        self.assertIn("  specialization_test.specialize.default: Eq(", message)
        self.assertNotIn("may be responsible", message)

    def test_only_specializing_op_is_named(self):
        with torch.library._scoped_library("specialization_test", "FRAGMENT") as lib:
            for name in ("pin", "benign"):
                lib.define(
                    f"{name}(Tensor x) -> Tensor", tags=torch.Tag.pt2_compliant_tag
                )
                lib.impl(name, lambda x: x + 1, "CPU")
                torch.library.register_fake(
                    f"specialization_test::{name}",
                    lambda x: torch.empty_like(x),
                    lib=lib,
                )
            pin = torch.ops.specialization_test.pin.default
            benign = torch.ops.specialization_test.benign.default

            def decompose_pin(x):
                return x + int(x.shape[0] // 2)

            class Model(torch.nn.Module):
                def forward(self, x):
                    return benign(pin(x))

            ep = torch.export.export(
                Model(),
                (torch.randn(6, 2),),
                dynamic_shapes=({0: torch.export.Dim.DYNAMIC},),
                strict=False,
            )
            create_table = _fx_passes._decomp.create_onnx_friendly_decomposition_table
            decomps = {pin: decompose_pin, benign: lambda x: x + 1}
            with (
                mock.patch.object(
                    _fx_passes._decomp,
                    "create_onnx_friendly_decomposition_table",
                    side_effect=lambda ops: create_table(ops) | decomps,
                ),
                self.assertWarns(DynamicShapeSpecializationWarning) as caught,
            ):
                torch.onnx.export(ep, verbose=False)
            message = str(caught.warning)
            self.assertIn("specialization_test.pin.default: Eq(", message)
            self.assertNotIn("specialization_test.benign", message)
            self.assertNotIn("may be responsible", message)

    @common_utils.parametrize("constraint_kind", ["bound", "divisibility"])
    def test_custom_decomposition_constraint_kind(self, constraint_kind):
        with torch.library._scoped_library("specialization_test", "FRAGMENT") as lib:
            lib.define("checked(Tensor x) -> Tensor", tags=torch.Tag.pt2_compliant_tag)
            lib.impl("checked", lambda x: x.sin(), "CPU")
            torch.library.register_fake(
                "specialization_test::checked", lambda x: torch.empty_like(x), lib=lib
            )
            target = torch.ops.specialization_test.checked.default

            def decompose(x):
                if constraint_kind == "bound":
                    torch._check(x.shape[0] >= 4)
                else:
                    torch._check(x.shape[0] % 2 == 0)
                return x.sin()

            class Model(torch.nn.Module):
                def forward(self, x):
                    return target(x)

            ep = torch.export.export(
                Model(),
                (torch.randn(6, 2),),
                dynamic_shapes=({0: torch.export.Dim.DYNAMIC},),
                strict=False,
            )
            create_table = _fx_passes._decomp.create_onnx_friendly_decomposition_table
            with (
                mock.patch.object(
                    _fx_passes._decomp,
                    "create_onnx_friendly_decomposition_table",
                    side_effect=lambda ops: create_table(ops) | {target: decompose},
                ),
                warnings.catch_warnings(record=True) as caught,
            ):
                warnings.simplefilter("always", DynamicShapeSpecializationWarning)
                torch.onnx.export(ep, verbose=False)
            messages = [
                str(w.message)
                for w in caught
                if issubclass(w.category, DynamicShapeSpecializationWarning)
            ]
            if constraint_kind == "bound":
                self.assertEqual(messages, [])
            else:
                self.assertEqual(len(messages), 1)
                self.assertIn(
                    "may give wrong results for sizes that do not satisfy these constraints",
                    messages[0],
                )
                self.assertNotIn("will give", messages[0])
                self.assertIn(
                    "specialization_test.checked.default: Eq(Mod(", messages[0]
                )
                self.assertIn("To fix:", messages[0])

    def test_odd_branch_divisibility_warns(self):
        with torch.library._scoped_library("specialization_test", "FRAGMENT") as lib:
            lib.define("parity(Tensor x) -> Tensor", tags=torch.Tag.pt2_compliant_tag)
            lib.impl("parity", lambda x: x + 1, "CPU")
            torch.library.register_fake(
                "specialization_test::parity", lambda x: torch.empty_like(x), lib=lib
            )
            target = torch.ops.specialization_test.parity.default

            def decompose(x):
                return x * 2 if x.shape[0] % 2 == 0 else x + 1

            class Model(torch.nn.Module):
                def forward(self, x):
                    return target(x)

            ep = torch.export.export(
                Model(),
                (torch.randn(5, 2),),
                dynamic_shapes=({0: torch.export.Dim.DYNAMIC},),
                strict=False,
            )
            create_table = _fx_passes._decomp.create_onnx_friendly_decomposition_table
            with (
                mock.patch.object(
                    _fx_passes._decomp,
                    "create_onnx_friendly_decomposition_table",
                    side_effect=lambda ops: create_table(ops) | {target: decompose},
                ),
                self.assertWarns(DynamicShapeSpecializationWarning) as caught,
            ):
                torch.onnx.export(ep, verbose=False)
        message = str(caught.warning)
        self.assertRegex(
            message, r"specialization_test\.parity\.default: Ne\(Mod\(s\d+, 2\), 0\)"
        )
        self.assertIn("may give wrong results", message)
        self.assertNotIn("will give wrong results", message)

    def test_attribution_preserves_decomposed_graph(self):
        class ResnetBlock(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(2, 2, 3, padding=1)
                self.bn = torch.nn.BatchNorm2d(2)

            def forward(self, x):
                return (self.bn(self.conv(x)).relu() + x).relu()

        model = ResnetBlock().eval()
        inputs = (torch.randn(2, 2, 6, 6),)
        registry = _registration.ONNXRegistry.from_torchlib()
        codes = []
        for with_hook in (False, True):
            ep = torch.export.export(
                model,
                inputs,
                dynamic_shapes=(
                    {0: torch.export.Dim.DYNAMIC, 2: torch.export.Dim.DYNAMIC},
                ),
                strict=False,
            )
            before = _specialization.snapshot(ep, registry)
            if with_hook:
                self.assertTrue(before.candidate_targets)
            hook = _specialization.attribution_hook(before) if with_hook else None
            decomposed = _fx_passes.decompose_with_registry(
                ep, registry, decomp_table_hook=hook
            )
            codes.append(decomposed.graph_module.code)
        self.assertEqual(codes[0], codes[1])

    @common_utils.parametrize("warnings_as_errors", [False, True])
    def test_custom_decomposition_pins_input_dimension(self, warnings_as_errors):
        with torch.library._scoped_library("specialization_test", "FRAGMENT") as lib:
            lib.define("pin_dim(Tensor x) -> Tensor", tags=torch.Tag.pt2_compliant_tag)
            lib.impl("pin_dim", lambda x: x.reshape(x.shape[0], -1), "CPU")
            torch.library.register_fake(
                "specialization_test::pin_dim", lambda x: torch.empty_like(x), lib=lib
            )
            target = torch.ops.specialization_test.pin_dim.default

            def decompose(x):
                return x.reshape(int(x.shape[0]), -1)

            class Model(torch.nn.Module):
                def forward(self, x):
                    return target(x)

            ep = torch.export.export(
                Model(),
                (torch.randn(6, 2),),
                dynamic_shapes=({0: torch.export.Dim.DYNAMIC},),
                strict=False,
            )
            with mock.patch.dict(
                torch._decomp.decomposition_table, {target: decompose}
            ):
                if warnings_as_errors:
                    with warnings.catch_warnings():
                        warnings.simplefilter(
                            "error", torch.onnx.errors.OnnxExporterWarning
                        )
                        with self.assertRaises(
                            torch.onnx.errors.OnnxExporterError
                        ) as error:
                            torch.onnx.export(ep, verbose=False)
                    cause = error.exception.__cause__
                    self.assertIsInstance(cause, DynamicShapeSpecializationWarning)
                    message = str(cause)
                else:
                    with self.assertWarns(DynamicShapeSpecializationWarning) as caught:
                        torch.onnx.export(ep, verbose=False)
                    message = str(caught.warning)
            self.assertIn("x.size()[0]", message)
            self.assertRegex(message, r"Eq\([^)]*, 6\)")

    @contextlib.contextmanager
    def _pin_dim_program(self):
        with torch.library._scoped_library("specialization_test", "FRAGMENT") as lib:
            lib.define("pin_dim(Tensor x) -> Tensor", tags=torch.Tag.pt2_compliant_tag)
            lib.impl("pin_dim", lambda x: x.reshape(x.shape[0], -1), "CPU")
            torch.library.register_fake(
                "specialization_test::pin_dim", lambda x: torch.empty_like(x), lib=lib
            )
            target = torch.ops.specialization_test.pin_dim.default

            class Model(torch.nn.Module):
                def forward(self, x):
                    return target(x)

            ep = torch.export.export(
                Model(),
                (torch.randn(6, 2),),
                dynamic_shapes=({0: torch.export.Dim.DYNAMIC},),
                strict=False,
            )
            create_table = _fx_passes._decomp.create_onnx_friendly_decomposition_table
            decomps = {target: lambda x: x.reshape(int(x.shape[0]), -1)}
            with mock.patch.object(
                _fx_passes._decomp,
                "create_onnx_friendly_decomposition_table",
                side_effect=lambda ops: create_table(ops) | decomps,
            ):
                yield ep

    def test_repeated_export_warns_each_time(self):
        with self._pin_dim_program() as ep:
            with warnings.catch_warnings():
                warnings.simplefilter("error", torch.onnx.errors.OnnxExporterWarning)
                with self.assertRaises(torch.onnx.errors.OnnxExporterError) as error:
                    torch.onnx.export(ep, verbose=False)
            with self.assertWarns(DynamicShapeSpecializationWarning) as caught:
                torch.onnx.export(ep, verbose=False)
        messages = [str(error.exception.__cause__), str(caught.warning)]
        for message in messages:
            self.assertRegex(
                message, r"specialization_test\.pin_dim\.default: Eq\([^)]*, 6\)"
            )
        self.assertNotIn("already decomposed by an earlier ONNX export", messages[0])
        self.assertIn("already decomposed by an earlier ONNX export", messages[1])

    @common_utils.parametrize(
        "model_kind", ["resnet", "interpolate", "layer_norm", "static", "check"]
    )
    def test_no_specialization(self, model_kind):
        class CheckedModel(torch.nn.Module):
            def forward(self, x):
                torch._check(x.shape[0] % 2 == 0)
                return x.sin()

        class LayerNorm(torch.nn.Module):
            def forward(self, x):
                return torch.nn.functional.layer_norm(x, x.shape[-1:])

        models = {
            "resnet": torch.nn.Sequential(
                torch.nn.Conv2d(2, 4, 3),
                torch.nn.BatchNorm2d(4),
                torch.nn.ReLU(),
                torch.nn.AdaptiveAvgPool2d((1, 1)),
            ).eval(),
            "interpolate": torch.nn.Upsample(scale_factor=2, mode="bilinear"),
            "layer_norm": LayerNorm(),
            "static": torch.nn.ReLU(),
            "check": CheckedModel(),
        }
        dynamic = torch.export.Dim.DYNAMIC
        dynamic_shapes = ({2: dynamic, 3: dynamic},)
        if model_kind == "static":
            dynamic_shapes = None
        elif model_kind == "check":
            dynamic_shapes = ({0: dynamic},)
        elif model_kind == "layer_norm":
            dynamic_shapes = ({0: dynamic, 2: dynamic},)
        ep = torch.export.export(
            models[model_kind],
            (torch.randn(4, 2, 6, 6),),
            dynamic_shapes=dynamic_shapes,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", DynamicShapeSpecializationWarning)
            for _ in range(2):
                torch.onnx.export(ep, verbose=False)
        self.assertEqual(
            [
                w
                for w in caught
                if issubclass(w.category, DynamicShapeSpecializationWarning)
            ],
            [],
        )


@common_utils.instantiate_parametrized_tests
class TestSpecializationHelpers(common_utils.TestCase):
    hw_classification = HardwareClassification.GENERIC

    @common_utils.parametrize(
        "expression, expected",
        [
            ("Eq(Mod(s, 2), 0)", "divisibility"),
            ("Eq(0, Mod(s, 2))", "divisibility"),
            ("Eq(s, 6)", "pin"),
            ("Eq(6, s)", "pin"),
            ("Eq(FloorDiv(s, 2), 3)", "pin"),
            ("Eq(s, t)", "relation"),
            ("Ne(Mod(s, 2), 0)", "divisibility"),
            ("Ne(PythonMod(s, 2), 0)", "divisibility"),
            ("Ne(s, 1)", "bound"),
            ("Ne(s, 0)", "bound"),
            ("s < 6", "bound"),
            ("s <= 6", "bound"),
            ("Or(Eq(s, 6), Eq(s, 8))", "bound"),
            ("And(Eq(s, 6), Eq(Mod(t, 2), 0))", "pin"),
            ("And(Eq(s, t), Eq(Mod(t, 2), 0))", "divisibility"),
            ("And(Eq(s, t), t >= 2)", "relation"),
            ("And(s >= 2, t >= 2)", "bound"),
        ],
    )
    def test_constraint_classification(self, expression, expected):
        names = {"FloorDiv": FloorDiv, "PythonMod": PythonMod}
        expr = sympy.sympify(expression, locals=names)
        self.assertEqual(_specialization._constraint_kind(expr), expected)

    def test_snapshot_without_shape_env(self):
        with mock.patch.object(_specialization, "_get_shape_env", return_value=None):
            self.assertIsNone(
                _specialization.snapshot(mock.Mock(), _registration.ONNXRegistry())
            )
            _specialization.warn_if_specialized(mock.Mock(), mock.Mock())
        _specialization.warn_if_specialized(mock.Mock(), None)
        self.assertIsNone(_specialization.attribution_hook(None))


if __name__ == "__main__":
    common_utils.run_tests()
