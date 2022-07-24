"""ONNX exporter."""
import warnings

from torch import _C
from torch._C import _onnx as _C_onnx
from torch._C._onnx import (
    _CAFFE2_ATEN_FALLBACK,
    OperatorExportTypes,
    TensorProtoDataType,
    TrainingMode,
)

from . import (
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
    symbolic_registry,
    utils,
)
from ._exporter_states import ExportTypes, SymbolicContext
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
    "symbolic_registry",
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
    # Enums
    "ExportTypes",
    "OperatorExportTypes",
    "TrainingMode",
    "TensorProtoDataType",
    # Classes
    "SymbolicContext",
    # Public functions
    "export",
    "export_to_pretty_string",
    "is_in_onnx_export",
    "select_model_mode_for_export",
    "register_custom_op_symbolic",
    "unregister_custom_op_symbolic",
    "disable_log",
    "enable_log",
    "is_onnx_log_enabled",
    "log",
    "set_log_stream",
    # Errors
    "CheckerError",  # Backwards compatibility
]

# Set namespace for exposed private names
ExportTypes.__module__ = "torch.onnx"
SymbolicContext.__module__ = "torch.onnx"

producer_name = "pytorch"
producer_version = _C_onnx.PRODUCER_VERSION


@_deprecation.deprecated(
    since="1.12.0", removed_in="TBD", instructions="use `torch.onnx.export` instead"
)
def _export(*args, **kwargs):
    return utils._export(*args, **kwargs)


def is_onnx_log_enabled() -> bool:
    r"""Returns True iff ONNX logging is turned on."""
    return _C._jit_is_onnx_log_enabled()


def enable_log() -> None:
    r"""Enables ONNX logging."""
    _C._jit_set_onnx_log_enabled(True)


def disable_log() -> None:
    r"""Disables ONNX logging."""
    _C._jit_set_onnx_log_enabled(False)


def set_log_stream(stream_name: str = "stdout") -> None:
    r"""Sets output stream for ONNX logging.

    Args:
        stream_name (str, default "stdout"): Only 'stdout' and 'stderr' are supported
            as ``stream_name``.
    """
    _C._jit_set_onnx_log_output_stream(stream_name)


def log(*args) -> None:
    r"""A simple logging facility for ONNX exporter.

    Args:
        args: Arguments are converted to string, concatenated together with a newline
            character appended to the end, and flushed to output stream.
    """
    _C._jit_onnx_log(*args)
