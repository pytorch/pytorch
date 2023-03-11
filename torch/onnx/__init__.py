"""ONNX exporter."""

from torch import _C
from torch._C import _onnx as _C_onnx
from torch._C._onnx import (
    _CAFFE2_ATEN_FALLBACK,
    OperatorExportTypes,
    TensorProtoDataType,
    TrainingMode,
)

from . import (  # usort:skip. Keep the order instead of sorting lexicographically
    _deprecation,
    errors,
    symbolic_caffe2,
    symbolic_helper,
    symbolic_opset7,
    symbolic_opset8,
    symbolic_opset9,
    symbolic_opset10,
    symbolic_opset11,
    symbolic_opset12,
    symbolic_opset13,
    symbolic_opset14,
    symbolic_opset15,
    symbolic_opset16,
    symbolic_opset17,
    symbolic_opset18,
    utils,
)

# TODO(After 1.13 release): Remove the deprecated SymbolicContext
from ._exporter_states import ExportTypes, SymbolicContext
from ._type_utils import JitScalarType
from .errors import CheckerError  # Backwards compatibility
from .utils import (
    _optimize_graph,
    _run_symbolic_function,
    _run_symbolic_method,
    export,
    export_to_pretty_string,
    is_in_onnx_export,
    register_custom_op_symbolic,
    select_model_mode_for_export,
    unregister_custom_op_symbolic,
)

__all__ = [
    # Modules
    "symbolic_helper",
    "utils",
    "errors",
    # All opsets
    "symbolic_caffe2",
    "symbolic_opset7",
    "symbolic_opset8",
    "symbolic_opset9",
    "symbolic_opset10",
    "symbolic_opset11",
    "symbolic_opset12",
    "symbolic_opset13",
    "symbolic_opset14",
    "symbolic_opset15",
    "symbolic_opset16",
    "symbolic_opset17",
    "symbolic_opset18",
    # Enums
    "ExportTypes",
    "OperatorExportTypes",
    "TrainingMode",
    "TensorProtoDataType",
    "JitScalarType",
    # Public functions
    "export",
    "export_to_pretty_string",
    "is_in_onnx_export",
    "select_model_mode_for_export",
    "register_custom_op_symbolic",
    "unregister_custom_op_symbolic",
    "disable_log",
    "enable_log",
    # Errors
    "CheckerError",  # Backwards compatibility
]

# Set namespace for exposed private names
ExportTypes.__module__ = "torch.onnx"
JitScalarType.__module__ = "torch.onnx"

producer_name = "pytorch"
producer_version = _C_onnx.PRODUCER_VERSION


@_deprecation.deprecated(
    since="1.12.0", removed_in="2.0", instructions="use `torch.onnx.export` instead"
)
def _export(*args, **kwargs):
    return utils._export(*args, **kwargs)


# TODO(justinchuby): Deprecate these logging functions in favor of the new diagnostic module.

# Returns True iff ONNX logging is turned on.
is_onnx_log_enabled = _C._jit_is_onnx_log_enabled


def enable_log() -> None:
    r"""Enables ONNX logging."""
    _C._jit_set_onnx_log_enabled(True)


def disable_log() -> None:
    r"""Disables ONNX logging."""
    _C._jit_set_onnx_log_enabled(False)


"""Sets output stream for ONNX logging.

Args:
    stream_name (str, default "stdout"): Only 'stdout' and 'stderr' are supported
        as ``stream_name``.
"""
set_log_stream = _C._jit_set_onnx_log_output_stream


"""A simple logging facility for ONNX exporter.

Args:
    args: Arguments are converted to string, concatenated together with a newline
        character appended to the end, and flushed to output stream.
"""
log = _C._jit_onnx_log
