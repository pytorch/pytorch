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

from torch.onnx._internal.fx import type_utils as fx_type_utils

# NOTE: Symbolic shapes could be a calculation of values, such as
# Tensor(i64[s0, 64, (s1//2) - 2, (s1//2) - 2]) where s0 and s1 are symbolic
# so we need to relax the length limit.
_LENGTH_LIMIT: int = 120

# NOTE(bowbao): This is a shim over `torch.onnx._internal.diagnostics`, which is
# used in `torch.onnx`, and loaded with `torch`. Hence anything related to `onnxscript`
# cannot be put there.


@functools.singledispatch
def _format_argument(obj: Any) -> str:
    return formatter.format_argument(obj)


def format_argument(obj: Any) -> str:
    formatter = _format_argument.dispatch(type(obj))
    result_str = formatter(obj)

    result_str_lines = result_str.splitlines()
    for line in result_str_lines:
        if len(line) > _LENGTH_LIMIT:
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
    node_string = f"fx.Node({obj.target})[{obj.op}]:"
    if "val" not in obj.meta:
        return node_string + "None"
    return node_string + _format_nested_argument_by_dtype(obj.meta["val"])


@_format_argument.register
def _torch_fx_symbolic_value(
    obj,  # NOTE: functools.singledispatch does not support Union until 3.11, so we use Any here.
) -> str:
    return f"Sym({obj})"


@_format_argument.register
def _torch_tensor(obj: torch.Tensor) -> str:
    return f"Tensor({fx_type_utils.from_torch_dtype_to_abbr(obj.dtype)}{_stringify_shape(obj.shape)})"


@_format_argument.register
def _list(obj: list) -> str:
    list_string = f"List[length={len(obj)}](\n"
    if not obj:
        return list_string + "None)"
    for item in obj:
        list_string += f"{_format_nested_argument_by_dtype(item)},\n"
    return list_string + ")"


@_format_argument.register
def _tuple(obj: tuple) -> str:
    tuple_string = f"Tuple[length={len(obj)}](\n"
    if not obj:
        return tuple_string + "None)"
    for item in obj:
        tuple_string += f"{_format_nested_argument_by_dtype(item)},\n"
    return tuple_string + ")"


@_format_argument.register
def _dict(obj: dict) -> str:
    dict_string = f"Dict[length={len(obj)}](\n"
    if not obj:
        return dict_string + "None)"
    for key, value in obj.items():
        dict_string += f"{key}: {_format_nested_argument_by_dtype(value)},\n"
    return dict_string + ")"


@_format_argument.register
def _torch_nn_parameter(obj: torch.nn.Parameter) -> str:
    return f"Parameter({format_argument(obj.data)})"


@_format_argument.register
def _onnxscript_torch_script_tensor(obj: graph_building.TorchScriptTensor) -> str:
    return f"`TorchScriptTensor({fx_type_utils.from_torch_dtype_to_abbr(obj.dtype)}{_stringify_shape(obj.shape)})`"


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


def _format_nested_argument_by_dtype(obj: Any) -> str:
    """Dispatch to the correct formatter based on the type of the argument."""
    if isinstance(obj, torch.Tensor):
        return _torch_tensor(obj)
    if isinstance(obj, torch.nn.Parameter):
        return _torch_nn_parameter(obj)
    if isinstance(obj, torch.fx.Node):
        return _torch_fx_node(obj)
    if fx_type_utils.is_torch_symbolic_type(obj):
        return _torch_fx_symbolic_value(obj)
    if isinstance(obj, graph_building.TorchScriptTensor):
        return _onnxscript_torch_script_tensor(obj)
    if isinstance(obj, onnxscript.OnnxFunction):
        return _onnxscript_onnx_function(obj)
    if isinstance(obj, onnxscript.TracedOnnxFunction):
        return _onnxscript_traced_onnx_function(obj)
    if isinstance(obj, list):
        return _list(obj)
    if isinstance(obj, tuple):
        return _tuple(obj)
    if isinstance(obj, dict):
        return _dict(obj)
    return format_argument(obj)


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
