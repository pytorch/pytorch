from .driver import ExportOutput

try:
    import onnx
except ImportError:
    raise


class Exporter:
    def __init__(self, model):
        ...

    def export(self, *args, **kwargs) -> ExportOutput:
        return ExportOutput(onnx.ModelProto())
