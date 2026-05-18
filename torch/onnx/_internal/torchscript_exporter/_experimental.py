"""Experimental classes and functions used by ONNX export."""

import dataclasses
from collections.abc import Mapping, Sequence

import torch
import torch._C._onnx as _C_onnx


@dataclasses.dataclass
class ExportOptions:
    """Arguments used by :func:`torch.onnx.export`."""

    # TODO(justinchuby): Deprecate and remove this class.

    export_params: bool = True
    verbose: bool = False
    training: _C_onnx.TrainingMode = _C_onnx.TrainingMode.EVAL
    input_names: Sequence[str] | None = None
    output_names: Sequence[str] | None = None
    operator_export_type: _C_onnx.OperatorExportTypes = _C_onnx.OperatorExportTypes.ONNX
    opset_version: int | None = None
    do_constant_folding: bool = True
    dynamic_axes: Mapping[str, Mapping[int, str] | Sequence[int]] | None = None
    keep_initializers_as_inputs: bool | None = None
    custom_opsets: Mapping[str, int] | None = None
    export_modules_as_functions: bool | set[type[torch.nn.Module]] = False
