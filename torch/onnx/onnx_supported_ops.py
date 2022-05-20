import inspect
from typing import Dict, List, Union

from torch import _C
from torch.onnx import _constants, symbolic_registry

for v in _constants.onnx_stable_opsets:
    symbolic_registry.register_version("", v)
symbolic_registry.register_version("", _constants.onnx_main_opset)


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
        s = f"{self.name}.{self.overload_name}("
        s += ", ".join(self.arguments)
        s += ") -> ("
        s += ", ".join(self.returns)
        s += ")"
        s += " in opsets "
        s += ", ".join(str(opset) for opset in self.opsets)
        return s

    def __hash__(self):
        # TODO(thiagocrepaldi): handle overload_name?
        return hash((self.name))

    def __eq__(self, other) -> bool:
        if not isinstance(other, _TorchSchema):
            return False
        # TODO(thiagocrepaldi): handle overload_name?
        return self.name == other.name

    def is_aten(self) -> bool:
        return self.name.startswith("aten::")

    def is_backward(self) -> bool:
        return "backward" in self.name


def _all_aten_forward_schemas():
    """Creates a list of _TorchSchema for all aten schemas."""
    torch_schemas = [_TorchSchema(s) for s in _C._jit_get_all_schemas()]
    torch_schemas = sorted(torch_schemas, key=lambda x: x.name)
    aten_schemas = [s for s in torch_schemas if s.is_aten() and not s.is_backward()]
    return aten_schemas


def _symbolic_argument_count(func):
    params = []
    sig = inspect.signature(func)
    optional_params = []
    has_var = False
    for name, p in sig.parameters.items():
        if p.kind.name == "VAR_POSITIONAL":
            has_var = True
        elif name == "_outputs" or name == "g":
            continue
        elif p.default != inspect._empty:  # type: ignore[attr-defined]
            optional_params.append(p)
        else:
            params.append(str(p))
    return params


def _all_symbolics_schemas():
    symbolics_schemas: Dict[str, _TorchSchema] = dict()

    for domain, version in symbolic_registry._registry:
        for opname, sym_func in symbolic_registry._registry[(domain, version)].items():
            symbolics_schema = _TorchSchema("aten::" + opname)
            symbolics_schema.arguments = _symbolic_argument_count(sym_func)
            if opname in symbolics_schemas:
                symbolics_schemas[opname].opsets.append(version)
            else:
                symbolics_schema.opsets = [version]
                symbolics_schemas[opname] = symbolics_schema
    return symbolics_schemas


def onnx_supported_ops():
    aten_schemas = _all_aten_forward_schemas()
    symbolic_schemas = _all_symbolics_schemas()
    torch_schemas = set(symbolic_schemas.values())
    supported_ops, unsupported_ops = list(), list()
    onnx_supported_ops = list()
    for schema in aten_schemas:
        if schema in torch_schemas:
            opname = schema.name[6:]  # without "aten::" prefix
            opsets = symbolic_schemas[opname].opsets
            if schema not in supported_ops:
                supported_ops.append(symbolic_schemas[opname])
                onnx_supported_ops.append((opname, " ".join(str(o) for o in opsets)))
        else:
            unsupported_ops.append(schema)
    return sorted(onnx_supported_ops, key=lambda x: x[0])
