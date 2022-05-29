"""ONNX exporter exceptions."""


class OnnxExporterError(RuntimeError):
    """Errors raised by the ONNX exporter."""

    pass


class CheckerError(OnnxExporterError):
    r"""Raised when ONNX checker detects an invalid model."""

    pass
