"""Error classes for the ONNX exporter."""

from __future__ import annotations

import torch.onnx.errors


class TorchExportError(torch.onnx.errors.OnnxExporterError):
    """Error during graph capturing using torch.export."""


class ConversionError(torch.onnx.errors.OnnxExporterError):
    """Error during ExportedProgram to ONNX conversion."""


class DispatchError(ConversionError):
    """Error during ONNX Function dispatching."""


class GraphConstructionError(ConversionError):
    """Error during ONNX graph construction."""
