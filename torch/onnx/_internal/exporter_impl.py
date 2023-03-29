import onnx

from torch.onnx._internal import _beartype
from torch.onnx._internal.exporter import Exporter, ExportOutput


class DynamoExporter(Exporter):
    @_beartype.beartype
    def run(self) -> ExportOutput:
        return ExportOutput(onnx.ModelProto())


__all__ = ["DynamoExporter"]
