# mypy: allow-untyped-defs
"""Globals used internally by the ONNX exporter.

Do not use this module outside of `torch.onnx` and its tests.

Be very judicious when adding any new global variables. Do not create new global
variables unless they are absolutely necessary.
"""

import torch._C._onnx as _C_onnx

# This module should only depend on _constants and nothing else in torch.onnx to keep
# dependency direction clean.
from torch.onnx import _constants


class _InternalGlobals:
    """Globals used internally by ONNX exporter.

    NOTE: Be very judicious when adding any new variables. Do not create new
    global variables unless they are absolutely necessary.
    """

    def __init__(self) -> None:
        self._export_onnx_opset_version = _constants.ONNX_DEFAULT_OPSET
        self._training_mode: _C_onnx.TrainingMode = _C_onnx.TrainingMode.EVAL
        self._in_onnx_export: bool = False
        # Whether the user's model is training during export
        self.export_training: bool = False
        self.operator_export_type: _C_onnx.OperatorExportTypes = (
            _C_onnx.OperatorExportTypes.ONNX
        )
        self.onnx_shape_inference: bool = True
        self._autograd_inlining: bool = True

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
    def export_onnx_opset_version(self) -> int:
        """Opset version used during export."""
        return self._export_onnx_opset_version

    @export_onnx_opset_version.setter
    def export_onnx_opset_version(self, value: int):
        supported_versions = range(
            _constants.ONNX_MIN_OPSET, _constants.ONNX_MAX_OPSET + 1
        )
        if value not in supported_versions:
            raise ValueError(f"Unsupported ONNX opset version: {value}")
        self._export_onnx_opset_version = value

    @property
    def in_onnx_export(self) -> bool:
        """Whether it is in the middle of ONNX export."""
        return self._in_onnx_export

    @in_onnx_export.setter
    def in_onnx_export(self, value: bool):
        if type(value) is not bool:
            raise TypeError("in_onnx_export must be a boolean")
        self._in_onnx_export = value

    @property
    def autograd_inlining(self) -> bool:
        """Whether Autograd must be inlined."""
        return self._autograd_inlining

    @autograd_inlining.setter
    def autograd_inlining(self, value: bool):
        if type(value) is not bool:
            raise TypeError("autograd_inlining must be a boolean")
        self._autograd_inlining = value


GLOBALS = _InternalGlobals()
