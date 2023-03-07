import functools
from typing import Any

import onnxscript  # type: ignore[import]
from onnxscript.function_libs.torch_aten import graph_building  # type: ignore[import]

import torch
from torch.onnx._internal import diagnostics
from torch.onnx._internal.diagnostics import infra
from torch.onnx._internal.diagnostics.infra import decorator, formatter, utils

_LENGTH_LIMIT: int = 80

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
        diagnostics.export_context().add_diagnostic(diag)

    return result_str


@_format_argument.register
def _torch_nn_module(obj: torch.nn.Module) -> str:
    return f"{obj.__class__.__name__}"


@_format_argument.register
def _torch_fx_graph_module(obj: torch.fx.GraphModule) -> str:
    return f"{obj.print_readable(print_output=False)}"


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
    diagnostics.export_context,
    diagnostic_type=diagnostics.ExportDiagnostic,
    format_argument=format_argument,
)

diagnose_step = functools.partial(
    decorator.diagnose_step,
    diagnostics.export_context,
    format_argument=format_argument,
)

rules = diagnostics.rules
export_context = diagnostics.export_context
levels = diagnostics.levels
