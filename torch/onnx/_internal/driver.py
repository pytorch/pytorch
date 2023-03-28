from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    import onnx


class ExportOutput:
    model_proto: onnx.ModelProto

    def __init__(self, model_proto: onnx.ModelProto):
        self.model_proto = model_proto
