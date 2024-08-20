# Owner(s): ["module: onnx"]
import io
import os

import onnx

import torch
from torch.onnx import dynamo_export, ExportOptions, ONNXProgram
from torch.onnx._internal import _exporter_legacy
from torch.onnx._internal._exporter_legacy import (
    ONNXProgramSerializer,
    ResolvedExportOptions,
)
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


class TestExportOptionsAPI(common_utils.TestCase):
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
        self.assertTrue(
            os.path.exists(_exporter_legacy._DEFAULT_FAILED_EXPORT_SARIF_LOG_PATH)
        )

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


if __name__ == "__main__":
    common_utils.run_tests()
