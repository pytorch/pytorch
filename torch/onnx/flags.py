"""Flags that control the ONNX exporter."""

from typing import Optional

import torch.onnx
from torch.onnx import _constants


class _InternalFlags:
    """Flags used internally by ONNX exporter."""

    def __init__(self):
        self._export_onnx_opset_version = _constants.onnx_default_opset
        self.operator_export_type: Optional[torch.onnx.OperatorExportTypes] = None
        self.training_mode = None
        self.onnx_shape_inference: bool = False

    @property
    def export_onnx_opset_version(self):
        return self._export_onnx_opset_version

    @export_onnx_opset_version.setter
    def export_onnx_opset_version(self, value: int):
        supported_versions = [_constants.onnx_main_opset]
        supported_versions.extend(_constants.onnx_stable_opsets)
        if value not in supported_versions:
            raise ValueError(f"Unsupported ONNX opset version: {value}")
        self._export_onnx_opset_version = value


_FLAGS = _InternalFlags()
