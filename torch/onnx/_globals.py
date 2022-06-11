"""Globals used internally by the ONNX exporter.

Do not use this module outside of `torch.onnx` and its tests.

Be very judicious when adding any new global variables. Do not create new global
variables unless they are absolutely necessary.
"""

from typing import Optional

import torch._C._onnx as _C_onnx

# This module should only depend on _constants and nothing else in torch.onnx to keep
# dependency direction clean.
from torch.onnx import _constants


class _InternalGlobals:
    """Globals used internally by ONNX exporter.

    NOTE: Be very judicious when adding any new variables. Do not create new
    global variables unless they are absolutely necessary.
    """

    def __init__(self):
        self._export_onnx_opset_version = _constants.onnx_default_opset
        self._training_mode: _C_onnx.TrainingMode = _C_onnx.TrainingMode.EVAL
        # Whether the user's model is training during export
        self.model_training: bool = False
        self.operator_export_type: Optional[_C_onnx.OperatorExportTypes] = None
        self.onnx_shape_inference: bool = False

    @property
    def training_mode(self):
        """The training mode for the exporter."""
        return self._training_mode

    @training_mode.setter
    def training_mode(self, training_mode: _C_onnx.TrainingMode):
        if not isinstance(training_mode, _C_onnx.TrainingMode):
            raise TypeError(
                "training_mode must be of type 'torch.onnx.TrainingMode'. This is "
                "likely a bug in torch.onnx."
            )
        self._training_mode = training_mode

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


GLOBALS = _InternalGlobals()
