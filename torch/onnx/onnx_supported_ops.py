from inspect import signature, _empty  # type: ignore[attr-defined]
from torch.onnx.symbolic_registry import _registry, register_version
from typing import List, Dict


# Class for Torch Schema
class TorchSchema:
    name: str
    overload_name: str
    arguments: List[str]
    optional_arguments: List[str]
    returns: List[str]
    opsets: List[int]

    def __init__(self, schema, symbolic=False) -> None:
        if not symbolic:
            self.name = schema.name
            self.overload_name = schema.overload_name
            self.arguments = [arg.name for arg in schema.arguments]
            self.optional_arguments = []
            self.returns = [ret.name for ret in schema.returns]
            self.opsets = []
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

    def __eq__(self, other) -> bool:
        if not isinstance(other, TorchSchema):
            return False
        if self.name == other.name:
            # TODO: Handle overloads
            return True
        return False

    def is_aten(self) -> bool:
        return self.name[:6] == "aten::"

    def is_backward(self) -> bool:
        return "backward" in self.name


# Create TorchSchema object directory of all aten schemas
def get_all_aten_forward_schemas():
    from torch._C import _jit_get_all_schemas
    torch_schemas = [TorchSchema(s) for s in _jit_get_all_schemas()]
    torch_schemas = sorted(torch_schemas, key=lambda x: x.name)
    aten_schemas = [s for s in torch_schemas if s.is_aten() and not s.is_backward()]
    return aten_schemas


# TODO: Do not hard code opset here
# Create TorchSchema object directory of all registered symbolics
# get_registered_op(opname, domain, version):
for i in range(7, 16):
    register_version("", i)


def get_symbolic_argument_count(func):
    params = []
    try:
        sig = signature(func)
        optional_params = []
        has_var = False
        for name, p in sig.parameters.items():
            if p.kind.name == "VAR_POSITIONAL":
                has_var = True
            elif name == "_outputs" or name == "g":
                continue
            elif p.default != _empty:
                optional_params.append(p)
            else:
                params.append(str(p))
    except Exception:
        pass
    return params


def get_all_symbolics_schemas():
    symbolics_schemas: Dict[str, TorchSchema] = dict()

    for domain, version in _registry:
        for opname, sym_func in _registry[(domain, version)].items():
            symbolics_schema = TorchSchema("aten::" + opname, symbolic=True)
            symbolics_schema.arguments = get_symbolic_argument_count(sym_func)
            if symbolics_schema in symbolics_schemas.values():
                symbolics_schemas[opname].opsets.append(version)
            else:
                symbolics_schema.opsets = [version]
                symbolics_schemas[opname] = symbolics_schema
    return symbolics_schemas


def get_onnx_supported_ops():
    aten_schemas = get_all_aten_forward_schemas()
    symbolic_schemas = get_all_symbolics_schemas()
    supported_ops, unsupported_ops = list(), list()
    onnx_supported_ops = list()
    for schema in aten_schemas:
        if schema in symbolic_schemas.values():
            opname, opsets = schema.name[6:], symbolic_schemas[schema.name[6:]].opsets
            if schema not in supported_ops:
                supported_ops.append(symbolic_schemas[opname])
                onnx_supported_ops.append((opname, " ".join([str(o) for o in opsets])))
        else:
            unsupported_ops.append(schema)
    onnx_supported_ops = sorted(onnx_supported_ops, key=lambda x: x[0])
    return onnx_supported_ops
