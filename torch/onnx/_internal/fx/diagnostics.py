# mypy: allow-untyped-defs
from __future__ import annotations

import dataclasses

import functools
import logging

from typing import Any, Optional

import onnxscript  # type: ignore[import]
from onnxscript.function_libs.torch_lib import graph_building  # type: ignore[import]

import torch
import torch.fx
from torch.onnx._internal import diagnostics
from torch.onnx._internal.diagnostics import infra
from torch.onnx._internal.diagnostics.infra import decorator, formatter
from torch.onnx._internal.fx import registration, type_utils as fx_type_utils

# NOTE: The following limits are for the number of items to display in diagnostics for
# a list, tuple or dict. The limit is picked such that common useful scenarios such as
# operator arguments are covered, while preventing excessive processing loads on considerably
# large containers such as the dictionary mapping from fx to onnx nodes.
_CONTAINER_ITEM_LIMIT: int = 10

# NOTE(bowbao): This is a shim over `torch.onnx._internal.diagnostics`, which is
# used in `torch.onnx`, and loaded with `torch`. Hence anything related to `onnxscript`
# cannot be put there.

# [NOTE: `dynamo_export` diagnostics logging]
# The 'dynamo_export' diagnostics leverages the PT2 artifact logger to handle the verbosity
# level of logs that are recorded in each SARIF log diagnostic. In addition to SARIF log,
# terminal logging is by default disabled. Terminal logging can be activated by setting
# the environment variable `TORCH_LOGS="onnx_diagnostics"`. When the environment variable
# is set, it also fixes logging level to `logging.DEBUG`, overriding the verbosity level
# specified in the diagnostic options.
# See `torch/_logging/__init__.py` for more on PT2 logging.
_ONNX_DIAGNOSTICS_ARTIFACT_LOGGER_NAME = "onnx_diagnostics"
diagnostic_logger = torch._logging.getArtifactLogger(
    "torch.onnx", _ONNX_DIAGNOSTICS_ARTIFACT_LOGGER_NAME
)


def is_onnx_diagnostics_log_artifact_enabled() -> bool:
    return torch._logging._internal.log_state.is_artifact_enabled(
        _ONNX_DIAGNOSTICS_ARTIFACT_LOGGER_NAME
    )


@functools.singledispatch
def _format_argument(obj: Any) -> str:
    return formatter.format_argument(obj)


def format_argument(obj: Any) -> str:
    formatter = _format_argument.dispatch(type(obj))
    return formatter(obj)


# NOTE: EDITING BELOW? READ THIS FIRST!
#
# The below functions register the `format_argument` function for different types via
# `functools.singledispatch` registry. These are invoked by the diagnostics system
# when recording function arguments and return values as part of a diagnostic.
# Hence, code with heavy workload should be avoided. Things to avoid for example:
# `torch.fx.GraphModule.print_readable()`.


@_format_argument.register
def _torch_nn_module(obj: torch.nn.Module) -> str:
    return f"torch.nn.Module({obj.__class__.__name__})"


@_format_argument.register
def _torch_fx_graph_module(obj: torch.fx.GraphModule) -> str:
    return f"torch.fx.GraphModule({obj.__class__.__name__})"


@_format_argument.register
def _torch_fx_node(obj: torch.fx.Node) -> str:
    node_string = f"fx.Node({obj.target})[{obj.op}]:"
    if "val" not in obj.meta:
        return node_string + "None"
    return node_string + format_argument(obj.meta["val"])


@_format_argument.register
def _torch_fx_symbolic_bool(obj: torch.SymBool) -> str:
    return f"SymBool({obj})"


@_format_argument.register
def _torch_fx_symbolic_int(obj: torch.SymInt) -> str:
    return f"SymInt({obj})"


@_format_argument.register
def _torch_fx_symbolic_float(obj: torch.SymFloat) -> str:
    return f"SymFloat({obj})"


@_format_argument.register
def _torch_tensor(obj: torch.Tensor) -> str:
    return f"Tensor({fx_type_utils.from_torch_dtype_to_abbr(obj.dtype)}{_stringify_shape(obj.shape)})"


@_format_argument.register
def _int(obj: int) -> str:
    return str(obj)


@_format_argument.register
def _float(obj: float) -> str:
    return str(obj)


@_format_argument.register
def _bool(obj: bool) -> str:
    return str(obj)


@_format_argument.register
def _str(obj: str) -> str:
    return obj


@_format_argument.register
def _registration_onnx_function(obj: registration.ONNXFunction) -> str:
    # TODO: Compact display of `param_schema`.
    return f"registration.ONNXFunction({obj.op_full_name}, is_custom={obj.is_custom}, is_complex={obj.is_complex})"


@_format_argument.register
def _list(obj: list) -> str:
    list_string = f"List[length={len(obj)}](\n"
    if not obj:
        return list_string + "None)"
    for i, item in enumerate(obj):
        if i >= _CONTAINER_ITEM_LIMIT:
            # NOTE: Print only first _CONTAINER_ITEM_LIMIT items.
            list_string += "...,\n"
            break
        list_string += f"{format_argument(item)},\n"
    return list_string + ")"


@_format_argument.register
def _tuple(obj: tuple) -> str:
    tuple_string = f"Tuple[length={len(obj)}](\n"
    if not obj:
        return tuple_string + "None)"
    for i, item in enumerate(obj):
        if i >= _CONTAINER_ITEM_LIMIT:
            # NOTE: Print only first _CONTAINER_ITEM_LIMIT items.
            tuple_string += "...,\n"
            break
        tuple_string += f"{format_argument(item)},\n"
    return tuple_string + ")"


@_format_argument.register
def _dict(obj: dict) -> str:
    dict_string = f"Dict[length={len(obj)}](\n"
    if not obj:
        return dict_string + "None)"
    for i, (key, value) in enumerate(obj.items()):
        if i >= _CONTAINER_ITEM_LIMIT:
            # NOTE: Print only first _CONTAINER_ITEM_LIMIT items.
            dict_string += "...\n"
            break
        dict_string += f"{key}: {format_argument(value)},\n"
    return dict_string + ")"


@_format_argument.register
def _torch_nn_parameter(obj: torch.nn.Parameter) -> str:
    return f"Parameter({format_argument(obj.data)})"


@_format_argument.register
def _onnxscript_torch_script_tensor(obj: graph_building.TorchScriptTensor) -> str:
    return f"`TorchScriptTensor({fx_type_utils.from_torch_dtype_to_abbr(obj.dtype)}{_stringify_shape(obj.shape)})`"  # type: ignore[arg-type]  # noqa: B950


@_format_argument.register
def _onnxscript_onnx_function(obj: onnxscript.OnnxFunction) -> str:
    return f"`OnnxFunction({obj.name})`"


@_format_argument.register
def _onnxscript_traced_onnx_function(obj: onnxscript.TracedOnnxFunction) -> str:
    return f"`TracedOnnxFunction({obj.name})`"


# from torch/fx/graph.py to follow torch format
def _stringify_shape(shape: Optional[torch.Size]) -> str:
    if shape is None:
        return ""
    return f"[{', '.join(str(x) for x in shape)}]"


rules = diagnostics.rules
levels = diagnostics.levels
RuntimeErrorWithDiagnostic = infra.RuntimeErrorWithDiagnostic
LazyString = formatter.LazyString
DiagnosticOptions = infra.DiagnosticOptions


@dataclasses.dataclass
class Diagnostic(infra.Diagnostic):
    logger: logging.Logger = dataclasses.field(init=False, default=diagnostic_logger)

    def log(self, level: int, message: str, *args, **kwargs) -> None:
        if self.logger.isEnabledFor(level):
            formatted_message = message % args
            if is_onnx_diagnostics_log_artifact_enabled():
                # Only log to terminal if artifact is enabled.
                # See [NOTE: `dynamo_export` diagnostics logging] for details.
                self.logger.log(level, formatted_message, **kwargs)

            self.additional_messages.append(formatted_message)


@dataclasses.dataclass
class DiagnosticContext(infra.DiagnosticContext[Diagnostic]):
    logger: logging.Logger = dataclasses.field(init=False, default=diagnostic_logger)
    _bound_diagnostic_type: type[Diagnostic] = dataclasses.field(
        init=False, default=Diagnostic
    )

    def __enter__(self):
        self._previous_log_level = self.logger.level
        # Adjust the logger level based on `options.verbosity_level` and the environment
        # variable `TORCH_LOGS`. See [NOTE: `dynamo_export` diagnostics logging] for details.
        if not is_onnx_diagnostics_log_artifact_enabled():
            return super().__enter__()
        else:
            return self


diagnose_call = functools.partial(
    decorator.diagnose_call,
    diagnostic_type=Diagnostic,
    format_argument=format_argument,
)


@dataclasses.dataclass
class UnsupportedFxNodeDiagnostic(Diagnostic):
    unsupported_fx_node: Optional[torch.fx.Node] = None

    def __post_init__(self) -> None:
        super().__post_init__()
        # NOTE: This is a hack to make sure that the additional fields must be set and
        # not None. Ideally they should not be set as optional. But this is a known
        # limitation with `dataclasses`. Resolvable in Python 3.10 with `kw_only=True`.
        # https://stackoverflow.com/questions/69711886/python-dataclasses-inheritance-and-default-values
        if self.unsupported_fx_node is None:
            raise ValueError("unsupported_fx_node must be specified.")
