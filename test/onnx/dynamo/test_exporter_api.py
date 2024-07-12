# Owner(s): ["module: onnx"]
import io
import os

import onnx
from beartype import roar

import torch
from torch.onnx import dynamo_export, ExportOptions, ONNXProgram
from torch.onnx._internal import exporter, io_adapter
from torch.onnx._internal.exporter import (
    LargeProtobufONNXProgramSerializer,
    ONNXProgramSerializer,
    ProtobufONNXProgramSerializer,
    ResolvedExportOptions,
)
from torch.onnx._internal.fx import diagnostics

from torch.testing._internal import common_utils


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


class SampleModelForDynamicShapes(torch.nn.Module):
    def forward(self, x, b):
        return x.relu(), b.sigmoid()


class _LargeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.randn(2**28))  # 1GB
        self.param2 = torch.nn.Parameter(torch.randn(2**28))  # 1GB

    def forward(self, x):
        return self.param + self.param2 + x


class TestExportOptionsAPI(common_utils.TestCase):
    def test_raise_on_invalid_argument_type(self):
        expected_exception_type = roar.BeartypeException
        with self.assertRaises(expected_exception_type):
            ExportOptions(dynamic_shapes=2)  # type: ignore[arg-type]
        with self.assertRaises(expected_exception_type):
            ExportOptions(diagnostic_options="DEBUG")  # type: ignore[arg-type]
        with self.assertRaises(expected_exception_type):
            ResolvedExportOptions(options=12)  # type: ignore[arg-type]

    def test_dynamic_shapes_default(self):
        options = ResolvedExportOptions(ExportOptions())
        self.assertFalse(options.dynamic_shapes)

    def test_dynamic_shapes_explicit(self):
        options = ResolvedExportOptions(ExportOptions(dynamic_shapes=None))
        self.assertFalse(options.dynamic_shapes)
        options = ResolvedExportOptions(ExportOptions(dynamic_shapes=True))
        self.assertTrue(options.dynamic_shapes)
        options = ResolvedExportOptions(ExportOptions(dynamic_shapes=False))
        self.assertFalse(options.dynamic_shapes)


class TestDynamoExportAPI(common_utils.TestCase):
    def test_default_export(self):
        output = dynamo_export(SampleModel(), torch.randn(1, 1, 2))
        self.assertIsInstance(output, ONNXProgram)
        self.assertIsInstance(output.model_proto, onnx.ModelProto)

    def test_export_with_options(self):
        self.assertIsInstance(
            dynamo_export(
                SampleModel(),
                torch.randn(1, 1, 2),
                export_options=ExportOptions(
                    dynamic_shapes=True,
                ),
            ),
            ONNXProgram,
        )

    def test_save_to_file_default_serializer(self):
        with common_utils.TemporaryFileName() as path:
            dynamo_export(SampleModel(), torch.randn(1, 1, 2)).save(path)
            onnx.load(path)

    def test_save_to_existing_buffer_default_serializer(self):
        buffer = io.BytesIO()
        dynamo_export(SampleModel(), torch.randn(1, 1, 2)).save(buffer)
        onnx.load(buffer)

    def test_save_to_file_using_specified_serializer(self):
        expected_buffer = "I am not actually ONNX"

        class CustomSerializer(ONNXProgramSerializer):
            def serialize(
                self, onnx_program: ONNXProgram, destination: io.BufferedIOBase
            ) -> None:
                destination.write(expected_buffer.encode())

        with common_utils.TemporaryFileName() as path:
            dynamo_export(SampleModel(), torch.randn(1, 1, 2)).save(
                path, serializer=CustomSerializer()
            )
            with open(path) as fp:
                self.assertEqual(fp.read(), expected_buffer)

    def test_save_to_file_using_specified_serializer_without_inheritance(self):
        expected_buffer = "I am not actually ONNX"

        # NOTE: Inheritance from `ONNXProgramSerializer` is not required.
        # Because `ONNXProgramSerializer` is a Protocol class.
        # `beartype` will not complain.
        class CustomSerializer:
            def serialize(
                self, onnx_program: ONNXProgram, destination: io.BufferedIOBase
            ) -> None:
                destination.write(expected_buffer.encode())

        with common_utils.TemporaryFileName() as path:
            dynamo_export(SampleModel(), torch.randn(1, 1, 2)).save(
                path, serializer=CustomSerializer()
            )
            with open(path) as fp:
                self.assertEqual(fp.read(), expected_buffer)

    def test_save_succeeds_when_model_greater_than_2gb_and_destination_is_str(self):
        with common_utils.TemporaryFileName() as path:
            dynamo_export(_LargeModel(), torch.randn(1)).save(path)

    def test_save_raises_when_model_greater_than_2gb_and_destination_is_not_str(self):
        with self.assertRaisesRegex(
            ValueError,
            "'destination' should be provided as a path-like string when saving a model larger than 2GB. ",
        ):
            dynamo_export(_LargeModel(), torch.randn(1)).save(io.BytesIO())

    def test_save_sarif_log_to_file_with_successful_export(self):
        with common_utils.TemporaryFileName(suffix=".sarif") as path:
            dynamo_export(SampleModel(), torch.randn(1, 1, 2)).save_diagnostics(path)
            self.assertTrue(os.path.exists(path))

    def test_save_sarif_log_to_file_with_failed_export(self):
        class ModelWithExportError(torch.nn.Module):
            def forward(self, x):
                raise RuntimeError("Export error")

        with self.assertRaises(RuntimeError):
            dynamo_export(ModelWithExportError(), torch.randn(1, 1, 2))
        self.assertTrue(os.path.exists(exporter._DEFAULT_FAILED_EXPORT_SARIF_LOG_PATH))

    def test_onnx_program_accessible_from_exception_when_export_failed(self):
        class ModelWithExportError(torch.nn.Module):
            def forward(self, x):
                raise RuntimeError("Export error")

        with self.assertRaises(torch.onnx.OnnxExporterError) as cm:
            dynamo_export(ModelWithExportError(), torch.randn(1, 1, 2))
        self.assertIsInstance(cm.exception, torch.onnx.OnnxExporterError)
        self.assertIsInstance(cm.exception.onnx_program, ONNXProgram)

    def test_access_onnx_program_model_proto_raises_when_onnx_program_is_emitted_from_failed_export(
        self,
    ):
        class ModelWithExportError(torch.nn.Module):
            def forward(self, x):
                raise RuntimeError("Export error")

        with self.assertRaises(torch.onnx.OnnxExporterError) as cm:
            dynamo_export(ModelWithExportError(), torch.randn(1, 1, 2))
        onnx_program = cm.exception.onnx_program
        with self.assertRaises(RuntimeError):
            onnx_program.model_proto

    def test_raise_from_diagnostic_warning_when_diagnostic_option_warning_as_error_is_true(
        self,
    ):
        with self.assertRaises(torch.onnx.OnnxExporterError):
            dynamo_export(
                SampleModel(),
                torch.randn(1, 1, 2),
                export_options=ExportOptions(
                    diagnostic_options=torch.onnx.DiagnosticOptions(
                        warnings_as_errors=True
                    )
                ),
            )

    def test_raise_on_invalid_save_argument_type(self):
        with self.assertRaises(roar.BeartypeException):
            ONNXProgram(torch.nn.Linear(2, 3))  # type: ignore[arg-type]
        onnx_program = ONNXProgram(
            onnx.ModelProto(),
            io_adapter.InputAdapter(),
            io_adapter.OutputAdapter(),
            diagnostics.DiagnosticContext("test", "1.0"),
            fake_context=None,
        )
        with self.assertRaises(roar.BeartypeException):
            onnx_program.save(None)  # type: ignore[arg-type]
        onnx_program.model_proto


class TestProtobufONNXProgramSerializerAPI(common_utils.TestCase):
    def test_raise_on_invalid_argument_type(self):
        with self.assertRaises(roar.BeartypeException):
            serializer = ProtobufONNXProgramSerializer()
            serializer.serialize(None, None)  # type: ignore[arg-type]

    def test_serialize_raises_when_model_greater_than_2gb(self):
        onnx_program = torch.onnx.dynamo_export(_LargeModel(), torch.randn(1))
        serializer = ProtobufONNXProgramSerializer()
        with self.assertRaisesRegex(ValueError, "exceeds maximum protobuf size of 2GB"):
            serializer.serialize(onnx_program, io.BytesIO())


class TestLargeProtobufONNXProgramSerializerAPI(common_utils.TestCase):
    def test_serialize_succeeds_when_model_greater_than_2gb(self):
        onnx_program = torch.onnx.dynamo_export(_LargeModel(), torch.randn(1))
        with common_utils.TemporaryFileName() as path:
            serializer = LargeProtobufONNXProgramSerializer(path)
            # `io.BytesIO()` is unused, but required by the Protocol interface.
            serializer.serialize(onnx_program, io.BytesIO())


class TestONNXExportWithDynamo(common_utils.TestCase):
    def test_args_normalization_with_no_kwargs(self):
        exported_program = torch.export.export(
            SampleModelTwoInputs(),
            (
                torch.randn(1, 1, 2),
                torch.randn(1, 1, 2),
            ),
        )
        onnx_program_from_new_exporter = torch.onnx.dynamo_export(
            exported_program, torch.randn(1, 1, 2), torch.randn(1, 1, 2)
        )
        onnx_program_from_old_exporter = torch.onnx.export(
            SampleModelTwoInputs(),
            (torch.randn(1, 1, 2), torch.randn(1, 1, 2)),
            dynamo=True,
        )
        self.assertEqual(
            onnx_program_from_new_exporter.model_proto,
            onnx_program_from_old_exporter.model_proto,
        )

    def test_args_is_tensor_not_tuple(self):
        exported_program = torch.export.export(SampleModel(), (torch.randn(1, 1, 2),))
        onnx_program_from_new_exporter = torch.onnx.dynamo_export(
            exported_program, torch.randn(1, 1, 2)
        )
        onnx_program_from_old_exporter = torch.onnx.export(
            SampleModel(), torch.randn(1, 1, 2), dynamo=True
        )
        self.assertEqual(
            onnx_program_from_new_exporter.model_proto,
            onnx_program_from_old_exporter.model_proto,
        )

    def test_args_normalization_with_kwargs(self):
        exported_program = torch.export.export(
            SampleModelTwoInputs(), (torch.randn(1, 1, 2),), {"b": torch.randn(1, 1, 2)}
        )
        onnx_program_from_new_exporter = torch.onnx.dynamo_export(
            exported_program, torch.randn(1, 1, 2), b=torch.randn(1, 1, 2)
        )
        onnx_program_from_old_exporter = torch.onnx.export(
            SampleModelTwoInputs(),
            (torch.randn(1, 1, 2), {"b": torch.randn(1, 1, 2)}),
            dynamo=True,
        )
        self.assertEqual(
            onnx_program_from_new_exporter.model_proto,
            onnx_program_from_old_exporter.model_proto,
        )

    def test_args_normalization_with_empty_dict_at_the_tail(self):
        exported_program = torch.export.export(
            SampleModelTwoInputs(), (torch.randn(1, 1, 2),), {"b": torch.randn(1, 1, 2)}
        )
        onnx_program_from_new_exporter = torch.onnx.dynamo_export(
            exported_program, torch.randn(1, 1, 2), b=torch.randn(1, 1, 2)
        )
        onnx_program_from_old_exporter = torch.onnx.export(
            SampleModelTwoInputs(),
            (torch.randn(1, 1, 2), {"b": torch.randn(1, 1, 2)}, {}),
            dynamo=True,
        )
        self.assertEqual(
            onnx_program_from_new_exporter.model_proto,
            onnx_program_from_old_exporter.model_proto,
        )

    def test_dynamic_axes_enable_dynamic_shapes_with_fully_specified_axes(self):
        exported_program = torch.export.export(
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
        )
        onnx_program_from_new_exporter = torch.onnx.dynamo_export(
            exported_program,
            torch.randn(2, 2, 3),
            b=torch.randn(2, 2, 3),
        )
        onnx_program_from_old_exporter = torch.onnx.export(
            SampleModelForDynamicShapes(),
            (torch.randn(2, 2, 3), {"b": torch.randn(2, 2, 3)}, {}),
            dynamic_axes={
                "x": {0: "customx_dim_0", 1: "customx_dim_1", 2: "customx_dim_2"},
                "b": {0: "customb_dim_0", 1: "customb_dim_1", 2: "customb_dim_2"},
            },
            dynamo=True,
        )
        self.assertEqual(
            onnx_program_from_new_exporter.model_proto,
            onnx_program_from_old_exporter.model_proto,
        )

    def test_dynamic_axes_enable_dynamic_shapes_with_default_axe_names(self):
        exported_program = torch.export.export(
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
        )
        onnx_program_from_new_exporter = torch.onnx.dynamo_export(
            exported_program,
            torch.randn(2, 2, 3),
            b=torch.randn(2, 2, 3),
        )
        onnx_program_from_old_exporter = torch.onnx.export(
            SampleModelForDynamicShapes(),
            (torch.randn(2, 2, 3), {"b": torch.randn(2, 2, 3)}, {}),
            dynamic_axes={
                "x": [0, 1, 2],
                "b": [0, 1, 2],
            },
            dynamo=True,
        )
        self.assertEqual(
            onnx_program_from_new_exporter.model_proto,
            onnx_program_from_old_exporter.model_proto,
        )

    def test_dynamic_axes_supports_partial_dynamic_shapes(self):
        exported_program = torch.export.export(
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
        onnx_program_from_new_exporter = torch.onnx.dynamo_export(
            exported_program,
            torch.randn(2, 2, 3),
            b=torch.randn(2, 2, 3),
        )
        onnx_program_from_old_exporter = torch.onnx.export(
            SampleModelForDynamicShapes(),
            (torch.randn(2, 2, 3), {"b": torch.randn(2, 2, 3)}, {}),
            dynamic_axes={
                "b": [0, 1, 2],
            },
            dynamo=True,
        )
        self.assertEqual(
            onnx_program_from_new_exporter.model_proto,
            onnx_program_from_old_exporter.model_proto,
        )

    def test_raises_unrelated_parameters_warning(self):
        message = (
            "f, export_params, verbose, training, input_names, output_names, operator_export_type, opset_version, "
            "do_constant_folding, keep_initializers_as_inputs, custom_opsets, export_modules_as_functions, and "
            "autograd_inlining are not supported for dynamo export at the moment."
        )

        with self.assertWarnsOnceRegex(UserWarning, message):
            _ = torch.onnx.export(
                SampleModel(),
                (torch.randn(1, 1, 2),),
                dynamo=True,
            )

    def test_input_names_are_not_yet_supported_in_dynamic_axes(self):
        with self.assertRaisesRegex(
            ValueError,
            "Assinging new input names is not supported yet. Please use model forward signature "
            "to specify input names in dynamix_axes.",
        ):
            _ = torch.onnx.export(
                SampleModelForDynamicShapes(),
                (
                    torch.randn(2, 2, 3),
                    torch.randn(2, 2, 3),
                ),
                input_names=["input"],
                dynamic_axes={"input": [0, 1]},
                dynamo=True,
            )

    def test_dynamic_shapes_hit_constraints_in_dynamo(self):
        # SampleModelTwoInputs has constraints becuse of add of two inputs,
        # so the two input shapes are related.
        with self.assertRaisesRegex(
            torch._dynamo.exc.UserError,
            "Constraints violated",
        ):
            _ = torch.onnx.export(
                SampleModelTwoInputs(),
                (torch.randn(2, 2, 3), torch.randn(2, 2, 3)),
                dynamic_axes={
                    "x": {0: "x_dim_0", 1: "x_dim_1", 2: "x_dim_2"},
                    "b": {0: "b_dim_0", 1: "b_dim_1", 2: "b_dim_2"},
                },
                dynamo=True,
            )

    def test_saved_f_exists_after_export(self):
        with common_utils.TemporaryFileName(suffix=".onnx") as path:
            _ = torch.onnx.export(
                SampleModel(), torch.randn(1, 1, 2), path, dynamo=True
            )
            self.assertTrue(os.path.exists(path))

    def test_raises_error_when_input_is_script_module(self):
        class ScriptModule(torch.jit.ScriptModule):
            def forward(self, x):
                return x

        with self.assertRaisesRegex(
            TypeError,
            "Dynamo export does not support ScriptModule or ScriptFunction.",
        ):
            _ = torch.onnx.export(ScriptModule(), torch.randn(1, 1, 2), dynamo=True)


if __name__ == "__main__":
    common_utils.run_tests()
