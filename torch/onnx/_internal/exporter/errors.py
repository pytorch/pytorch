class ExporterError(RuntimeError):
    """Error during export."""

    pass


class TorchExportError(ExporterError):
    """Error during torch.export.export."""

    pass


class OnnxConversionError(ExporterError):
    """Error during ONNX conversion."""

    pass


class DispatchError(OnnxConversionError):
    """Error during ONNX Funtion dispatching."""

    pass


class GraphConstructionError(OnnxConversionError):
    """Error during graph construction."""

    pass


class OnnxCheckerError(ExporterError):
    """Error during ONNX model checking."""

    pass


class OnnxRuntimeError(ExporterError):
    """Error during ONNX Runtime execution."""

    pass


class OnnxValidationError(ExporterError):
    """Output value mismatch."""

    pass
