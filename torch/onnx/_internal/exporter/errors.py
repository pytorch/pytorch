class ExporterError(RuntimeError):
    """Error during export."""


class TorchExportError(ExporterError):
    """Error during torch.export.export."""


class OnnxConversionError(ExporterError):
    """Error during ONNX conversion."""


class DispatchError(OnnxConversionError):
    """Error during ONNX Funtion dispatching."""


class GraphConstructionError(OnnxConversionError):
    """Error during graph construction."""


class OnnxCheckerError(ExporterError):
    """Error during ONNX model checking."""


class OnnxRuntimeError(ExporterError):
    """Error during ONNX Runtime execution."""


class OnnxValidationError(ExporterError):
    """Output value mismatch."""
