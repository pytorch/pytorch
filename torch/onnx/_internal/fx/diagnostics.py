from __future__ import annotations

import dataclasses

import functools
from typing import Any, Optional

import onnxscript  # type: ignore[import]
from onnxscript.function_libs.torch_lib import graph_building  # type: ignore[import]

import torch
import torch.fx
from torch.onnx._internal import diagnostics
from torch.onnx._internal.diagnostics import infra
from torch.onnx._internal.diagnostics.infra import decorator, formatter, utils

_LENGTH_LIMIT: int = 89

# NOTE(bowbao): This is a shim over `torch.onnx._internal.diagnostics`, which is
# used in `torch.onnx`, and loaded with `torch`. Hence anything related to `onnxscript`
# cannot be put there.


@functools.singledispatch
def _format_argument(obj: Any) -> str:
    return formatter.format_argument(obj)


def format_argument(obj: Any) -> str:
    formatter = _format_argument.dispatch(type(obj))
    result_str = formatter(obj)

    if len(result_str) > _LENGTH_LIMIT:
        # TODO(bowbao): group diagnostics.
        #   Related fields of sarif.Result: occurance_count, fingerprints.
        #   Do a final process to group results before outputing sarif log.
        diag = infra.Diagnostic(
            *diagnostics.rules.arg_format_too_verbose.format(
                level=infra.levels.WARNING,
                length=len(result_str),
                length_limit=_LENGTH_LIMIT,
                argument_type=type(obj),
                formatter_type=type(format_argument),
            )
        )
        diag.with_location(utils.function_location(formatter))
        diagnostics.export_context().log(diag)

    return result_str


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
    return f"torch.fx.Node(target: {obj.target})"


@_format_argument.register
def _torch_tensor(obj: torch.Tensor) -> str:
    return f"Tensor(shape={obj.shape}, dtype={obj.dtype})"


@_format_argument.register
def _torch_nn_parameter(obj: torch.nn.Parameter) -> str:
    return f"Parameter({format_argument(obj.data)})"


@_format_argument.register
def _onnxscript_torch_script_tensor(obj: graph_building.TorchScriptTensor) -> str:
    # TODO(bowbao) obj.dtype throws error.
    return f"`TorchScriptTensor({obj.name}, {obj.onnx_dtype}, {obj.shape}, {obj.symbolic_value()})`"


@_format_argument.register
def _onnxscript_onnx_function(obj: onnxscript.values.OnnxFunction) -> str:
    return f"`OnnxFunction({obj.name})`"


diagnose_call = functools.partial(
    decorator.diagnose_call,
    diagnostic_type=diagnostics.ExportDiagnostic,
    format_argument=format_argument,
)

rules = diagnostics.rules
levels = diagnostics.levels
DiagnosticContext = infra.DiagnosticContext
Diagnostic = infra.Diagnostic
RuntimeErrorWithDiagnostic = infra.RuntimeErrorWithDiagnostic


@dataclasses.dataclass
class UnsupportedFxNodeDiagnostic(Diagnostic):
    unsupported_fx_node: Optional[torch.fx.Node] = None

    def __post_init__(self) -> None:
        super().__post_init__()
        # NOTE: This is a hack to make sure that the additional fields must be set and
        # not None. Ideally they should not be set as optional. But this is a known
        # limiation with `dataclasses`. Resolvable in Python 3.10 with `kw_only=True`.
        # https://stackoverflow.com/questions/69711886/python-dataclasses-inheritance-and-default-values
        if self.unsupported_fx_node is None:
            raise ValueError("unsupported_fx_node must be specified.")
