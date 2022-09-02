import inspect
from typing import Dict, List, Union

from torch import _C
from torch.onnx import _constants
from torch.onnx._internal import registration


class _TorchSchema:
    def __init__(self, schema: Union[_C.FunctionSchema, str]) -> None:
        if isinstance(schema, _C.FunctionSchema):
            self.name: str = schema.name
            self.overload_name: str = schema.overload_name
            self.arguments: List[str] = [arg.name for arg in schema.arguments]
            self.optional_arguments: List[str] = []
            self.returns: List[str] = [ret.name for ret in schema.returns]
            self.opsets: List[int] = []
        else:
            self.name = schema
            self.overload_name = ""
            self.arguments = []
            self.optional_arguments = []
            self.returns = []
            self.opsets = []

    def __str__(self) -> str:
        s = (
            f"{self.name}.{self.overload_name}("
            + ", ".join(self.arguments)
            + ") -> ("
            + ", ".join(self.returns)
            + ")"
            + " in opsets "
            + ", ".join(str(opset) for opset in self.opsets)
        )
        return s

    def __hash__(self):
        # TODO(thiagocrepaldi): handle overload_name?
        return hash(self.name)

    def __eq__(self, other) -> bool:
        if not isinstance(other, _TorchSchema):
            return False
        # TODO(thiagocrepaldi): handle overload_name?
        return self.name == other.name

    def is_aten(self) -> bool:
        return self.name.startswith("aten::")

    def is_backward(self) -> bool:
        return "backward" in self.name


def _all_forward_schemas():
    """Creates a list of _TorchSchema for all schemas."""
    torch_schemas = [_TorchSchema(s) for s in _C._jit_get_all_schemas()]
    torch_schemas = sorted(torch_schemas, key=lambda x: x.name)
    aten_schemas = [s for s in torch_schemas if not s.is_backward()]
    return aten_schemas


def _symbolic_argument_count(func):
    params = []
    signature = inspect.signature(func)
    optional_params = []
    for name, parameter in signature.parameters.items():
        if name in {"_outputs", "g"}:
            continue
        if parameter.default is parameter.empty:
            optional_params.append(parameter)
        else:
            params.append(str(parameter))
    return params


def _all_symbolics_schemas() -> Dict[str, _TorchSchema]:
    symbolics_schemas = {}

    for name in registration.registry.all_functions():
        func_group = registration.registry.get_function_group(name)
        assert func_group is not None
        symbolics_schema = _TorchSchema(name)
        func = func_group.get(_constants.onnx_main_opset)
        if func is not None:
            symbolics_schema.arguments = _symbolic_argument_count(
                func_group.get(_constants.onnx_main_opset)
            )
            symbolics_schema.opsets = list(
                range(func_group.get_min_supported(), _constants.onnx_main_opset + 1)
            )
        else:
            # Only support opset < 9
            func = func_group.get(7)
            symbolics_schema.opsets = list(range(7, _constants.ONNX_BASE_OPSET))

        symbolics_schemas[name] = symbolics_schema

    return symbolics_schemas


def onnx_supported_ops():
    all_schemas = _all_forward_schemas()
    symbolic_schemas = _all_symbolics_schemas()
    torch_schemas = set(symbolic_schemas.values())
    supported_ops = []
    onnx_supported = []
    for schema in all_schemas:
        if schema in torch_schemas:
            opname = schema.name
            opsets = symbolic_schemas[opname].opsets
            if schema not in supported_ops:
                supported_ops.append(symbolic_schemas[opname])
                onnx_supported.append(
                    (
                        opname,
                        f"{opsets[0]}-{opsets[-1]}"
                        if len(opsets) > 1
                        else f"{opsets[0]}",
                    )
                )
    return sorted(onnx_supported, key=lambda x: x[0])
