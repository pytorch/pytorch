# Owner(s): ["module: onnx"]
import io
import os

import onnx
import torch
from beartype import roar
from torch.onnx import dynamo_export, ExportOptions, ExportOutput
from torch.onnx._internal import exporter, io_adapter
from torch.onnx._internal.exporter import (
    ExportOutputSerializer,
    ProtobufExportOutputSerializer,
    ResolvedExportOptions,
)
from torch.onnx._internal.fx import diagnostics

from torch.testing._internal import common_utils


class SampleModel(torch.nn.Module):
    def forward(self, x):
        y = x + 1
        z = y.relu()
        return (y, z)


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
        options = ResolvedExportOptions(None)
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
        self.assertIsInstance(output, ExportOutput)
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
            ExportOutput,
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

        class CustomSerializer(ExportOutputSerializer):
            def serialize(
                self, export_output: ExportOutput, destination: io.BufferedIOBase
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

        # NOTE: Inheritance from `ExportOutputSerializer` is not required.
        # Because `ExportOutputSerializer` is a Protocol class.
        # `beartype` will not complain.
        class CustomSerializer:
            def serialize(
                self, export_output: ExportOutput, destination: io.BufferedIOBase
            ) -> None:
                destination.write(expected_buffer.encode())

        with common_utils.TemporaryFileName() as path:
            dynamo_export(SampleModel(), torch.randn(1, 1, 2)).save(
                path, serializer=CustomSerializer()
            )
            with open(path) as fp:
                self.assertEqual(fp.read(), expected_buffer)

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

    def test_export_output_accessible_from_exception_when_export_failed(self):
        class ModelWithExportError(torch.nn.Module):
            def forward(self, x):
                raise RuntimeError("Export error")

        with self.assertRaises(torch.onnx.OnnxExporterError) as cm:
            dynamo_export(ModelWithExportError(), torch.randn(1, 1, 2))
        self.assertIsInstance(cm.exception, torch.onnx.OnnxExporterError)
        self.assertIsInstance(cm.exception.export_output, ExportOutput)

    def test_access_export_output_model_proto_raises_when_export_output_is_emitted_from_failed_export(
        self,
    ):
        class ModelWithExportError(torch.nn.Module):
            def forward(self, x):
                raise RuntimeError("Export error")

        with self.assertRaises(torch.onnx.OnnxExporterError) as cm:
            dynamo_export(ModelWithExportError(), torch.randn(1, 1, 2))
        export_output = cm.exception.export_output
        with self.assertRaises(RuntimeError):
            export_output.model_proto

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
            ExportOutput(torch.nn.Linear(2, 3))  # type: ignore[arg-type]
        export_output = ExportOutput(
            onnx.ModelProto(),
            io_adapter.InputAdapter(),
            io_adapter.OutputAdapter(),
            diagnostics.DiagnosticContext("test", "1.0"),
            fake_context=None,
        )
        with self.assertRaises(roar.BeartypeException):
            export_output.save(None)  # type: ignore[arg-type]
        export_output.model_proto


class TestProtobufExportOutputSerializerAPI(common_utils.TestCase):
    def test_raise_on_invalid_argument_type(self):
        with self.assertRaises(roar.BeartypeException):
            serializer = ProtobufExportOutputSerializer()
            serializer.serialize(None, None)  # type: ignore[arg-type]


if __name__ == "__main__":
    common_utils.run_tests()
