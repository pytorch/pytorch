import torch
import inspect
from inspect import signature
from torch.onnx.symbolic_registry import is_registered_op, get_registered_op, _registry, register_version
from typing import List


# Class for Torch Schema
class TorchSchema:
    name: str
    overload_name: str
    arguments: List[str]
    optional_arguments: List[str]
    returns: List[str]
    opset: List[int]

    def __init__(self, schema, symbolic=False) -> None:
        if symbolic is False:
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
        if (
            self.name == other.name
            # TODO: Handle overloads
            #and
            #    (
            #        len(self.arguments) == len(other.arguments)
            #        or len(self.optional_arguments) == 1
            #    )
        ):
            return True
        return False


    def is_aten(self) -> bool:
        return self.name[:6] == "aten::"

    def is_backward(self) -> bool:
        return "backward" in self.name

    def overload_handler(self) -> bool:

        def has_out(self):
            return self.overload_name == "out" or \
                   self.overload_name == "output" or \
                   "output" in self.overload_name or \
                   "_out" in self.overload_name

        # Named tensors not supported by TorchScript compiler
        def has_dimname(self):
            return self.overload_name == "Dimname" or \
                   self.overload_name == "dimname" or \
                   "name" in self.overload_name
    
        def is_inplace(self):
            return self.name[-1] == "_"
    
        def add_to_optional_arguments(self):
            if self.overload_name in self.arguments:
                #self.arguments.remove(self.overload_name)
                self.optional_arguments.append(self.overload_name)
            #else:
            #    if self.overload_name == "padding":
            #        print("Re")
            #    self.arguments.append(self.overload_name)

        elim_cond = has_dimname(self) or has_out(self) or is_inplace(self)
        if not elim_cond:
            add_to_optional_arguments(self)
        return elim_cond

    # TODO: several variants
    #   1. vec: upsample_nearest2d.vec(input, output_size, scale_factors)
    #   2. out: vdot.out(self, other, out)
    #   3. Scalar: true_divide.Scalar / true_divide.Tensor
    #   4. name, Dimname, names_dim:
    #           transpose.Dimname(self, dim0, dim1)
    #           var.names_dim(self, dim, unbiased, keepdim)


# Create TorchSchema object directory of all aten schemas
def get_all_aten_forward_schemas():
    torch_schemas = [TorchSchema(s) for s in torch._C._jit_get_all_schemas()]
    torch_schemas = sorted(torch_schemas, key=lambda x : x.name)
    aten_schemas = [s for s in torch_schemas if s.is_aten() and not s.is_backward()]
    return aten_schemas


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
            elif p.default != inspect._empty:
                optional_params.append(p)
            else:
                params.append(str(p))
    except:
        pass
    return params

def get_all_symbolics_schemas():
    symbolics_schemas = dict()
    for domain, version in _registry:
        for opname, sym_func in _registry[(domain, version)].items():
            symbolics_schema = TorchSchema("aten::"+opname, symbolic=True)
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
    onnx_supported_ops = sorted(onnx_supported_ops, key=lambda x : x[0])
    return onnx_supported_ops
