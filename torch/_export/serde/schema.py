# NOTE: This is a placeholder for iterating on export serialization schema design.
#       Anything is subject to change and no guarantee is provided at this point.

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Annotated, Dict, List, Optional

from torch._export.serde.union import _Union

# NOTE: Please update this value if any modifications are made to the schema
SCHEMA_VERSION = (8, 3)
TREESPEC_VERSION = 1


# NOTE: If you updated the schema, please run `scripts/export/update_schema.py`
# to update the auto generated files.
class ScalarType(IntEnum):
    UNKNOWN = 0
    BYTE = 1
    CHAR = 2
    SHORT = 3
    INT = 4
    LONG = 5
    HALF = 6
    FLOAT = 7
    DOUBLE = 8
    COMPLEXHALF = 9
    COMPLEXFLOAT = 10
    COMPLEXDOUBLE = 11
    BOOL = 12
    BFLOAT16 = 13
    UINT16 = 28
    FLOAT8E4M3FN = 29
    FLOAT8E5M2 = 30


class Layout(IntEnum):
    Unknown = 0
    SparseCoo = 1
    SparseCsr = 2
    SparseCsc = 3
    SparseBsr = 4
    SparseBsc = 5
    _mkldnn = 6
    Strided = 7


class MemoryFormat(IntEnum):
    Unknown = 0
    ContiguousFormat = 1
    ChannelsLast = 2
    ChannelsLast3d = 3
    PreserveFormat = 4


@dataclass
class Device:
    type: Annotated[str, 10]
    index: Annotated[Optional[int], 20] = None


@dataclass(repr=False)
class SymExprHint(_Union):
    as_int: Annotated[int, 10]
    as_bool: Annotated[bool, 20]
    as_float: Annotated[float, 30]


# This is for storing the symbolic expressions behind symints/symfloats/symbools
# For example, we can get something like
# SymExpr(expr_str="s0 + s1", hint=SymExprHint(as_int=4)
# if we also have the hint that s0 and s1 are both 2.
@dataclass
class SymExpr:
    expr_str: Annotated[str, 10]
    hint: Annotated[Optional[SymExprHint], 20] = None


@dataclass(repr=False)
class SymInt(_Union):
    as_expr: Annotated[SymExpr, 10]
    as_int: Annotated[int, 20]

@dataclass(repr=False)
class SymFloat(_Union):
    as_expr: Annotated[SymExpr, 10]
    as_float: Annotated[float, 20]


@dataclass(repr=False)
class SymBool(_Union):
    as_expr: Annotated[SymExpr, 10]
    as_bool: Annotated[bool, 20]


@dataclass
class TensorMeta:
    dtype: Annotated[ScalarType, 10]
    sizes: Annotated[List[SymInt], 20]
    requires_grad: Annotated[bool, 30]
    device: Annotated[Device, 40]
    strides: Annotated[List[SymInt], 50]
    storage_offset: Annotated[SymInt, 60]
    layout: Annotated[Layout, 70]


# In most cases we will use the "as_name" field to store arguments which are
# SymInts.
# The "as_int" field is used in the case where we have a list containing a mix
# of SymInt and ints (ex. [1, s0, ...]). We will serialize this type of list to
# be List[SymIntArgument] and map the SymInts to the "as_name" field, and ints
# to the "as_int" field.
@dataclass(repr=False)
class SymIntArgument(_Union):
    as_name: Annotated[str, 10]
    as_int: Annotated[int, 20]

# In most cases we will use the "as_name" field to store arguments which are
# SymFloats.
# The "as_float" field is used in the case where we have a list containing a mix
# of SymFloat and float (ex. [1.0, s0, ...]). We will serialize this type of list to
# be List[SymFloatArgument] and map the SymFloats to the "as_name" field, and ints
# to the "as_float" field.
@dataclass(repr=False)
class SymFloatArgument(_Union):
    as_name: Annotated[str, 10]
    as_float: Annotated[float, 20]

# In most cases we will use the "as_name" field to store arguments which are
# SymBools.
# The "as_bool" field is used in the case where we have a list containing a mix
# of SymBool and bools (ex. [True, i0, ...]). We will serialize this type of list to
# be List[SymboolArgument] and map the SymBools to the "as_name" field, and bools
# to the "as_bool" field.
@dataclass(repr=False)
class SymBoolArgument(_Union):
    as_name: Annotated[str, 10]
    as_bool: Annotated[bool, 20]


@dataclass
class TensorArgument:
    name: Annotated[str, 10]


@dataclass
class TokenArgument:
    name: Annotated[str, 10]


# This is use for storing the contents of a list which contain optional tensors
# (Tensor?[], ex. [Tensor, None, ...]), where the list will be serialized to the
# type List[OptionalTensorArgument], with tensor values seiralized to the
# "as_tensor" field, and None values serialized to the "as_none" field.
@dataclass(repr=False)
class OptionalTensorArgument(_Union):
    as_tensor: Annotated[TensorArgument, 20]
    as_none: Annotated[bool, 10]


@dataclass
class GraphArgument:
    name: Annotated[str, 10]
    graph: Annotated['Graph', 20]


@dataclass
class CustomObjArgument:
    name: Annotated[str, 10]
    class_fqn: Annotated[str, 20]


# This is actually a union type
@dataclass(repr=False)
class Argument(_Union):
    as_none: Annotated[bool, 10]
    as_tensor: Annotated[TensorArgument, 20]
    as_tensors: Annotated[List[TensorArgument], 30]
    as_int: Annotated[int, 50]
    as_ints: Annotated[List[int], 70]
    as_float: Annotated[float, 80]
    as_floats: Annotated[List[float], 90]
    as_string: Annotated[str, 100]
    as_strings: Annotated[List[str], 101]
    as_sym_int: Annotated[SymIntArgument, 110]
    as_sym_ints: Annotated[List[SymIntArgument], 120]
    as_scalar_type: Annotated[ScalarType, 130]
    as_memory_format: Annotated[MemoryFormat, 140]
    as_layout: Annotated[Layout, 150]
    as_device: Annotated[Device, 160]
    as_bool: Annotated[bool, 170]
    as_bools: Annotated[List[bool], 180]
    as_sym_bool: Annotated[SymBoolArgument, 182]
    as_sym_bools: Annotated[List[SymBoolArgument], 184]
    as_graph: Annotated[GraphArgument, 200]
    as_optional_tensors: Annotated[List[OptionalTensorArgument], 190]
    as_custom_obj: Annotated[CustomObjArgument, 210]
    as_operator: Annotated[str, 220]
    as_sym_float: Annotated[SymFloatArgument, 230]
    as_sym_floats: Annotated[List[SymFloatArgument], 240]

@dataclass
class NamedArgument:
    # Argument name from the operator schema
    name: Annotated[str, 10]
    arg: Annotated[Argument, 20]


@dataclass
class Node:
    target: Annotated[str, 10]
    inputs: Annotated[List[NamedArgument], 20]
    outputs: Annotated[List[Argument], 30]
    metadata: Annotated[Dict[str, str], 40]


@dataclass
class Graph:
    inputs: Annotated[List[Argument], 10]
    outputs: Annotated[List[Argument], 20]
    nodes: Annotated[List[Node], 30]
    tensor_values: Annotated[Dict[str, TensorMeta], 40]
    sym_int_values: Annotated[Dict[str, SymInt], 50]
    sym_bool_values: Annotated[Dict[str, SymBool], 60]
    # This is for deserializing the submodule graphs from higher order ops
    # (ex. cond, map) where single tensor returns will just return a single
    # tensor, rather than following export schema and returning a singleton
    # list.
    is_single_tensor_return: Annotated[bool, 70] = False
    custom_obj_values: Annotated[Dict[str, CustomObjArgument], 80] = field(default_factory=dict)
    sym_float_values: Annotated[Dict[str, SymFloat], 90] = field(default_factory=dict)

@dataclass
class UserInputSpec:
    # Actually, only tensors and SymInts are allowed here
    arg: Annotated[Argument, 10]


@dataclass(repr=False)
class ConstantValue(_Union):
    as_none: Annotated[bool, 10]
    as_int: Annotated[int, 20]
    as_float: Annotated[float, 30]
    as_string: Annotated[str, 40]
    as_bool: Annotated[bool, 50]


@dataclass
class InputToConstantInputSpec:
    name: Annotated[str, 10]
    value: Annotated[ConstantValue, 20]


@dataclass
class InputToParameterSpec:
    arg: Annotated[TensorArgument, 10]
    parameter_name: Annotated[str, 20]


@dataclass
class InputToBufferSpec:
    arg: Annotated[TensorArgument, 10]
    buffer_name: Annotated[str, 20]
    persistent: Annotated[bool, 30]



@dataclass
class InputToTensorConstantSpec:
    arg: Annotated[TensorArgument, 10]
    tensor_constant_name: Annotated[str, 20]


@dataclass
class InputToCustomObjSpec:
    arg: Annotated[CustomObjArgument, 10]
    custom_obj_name: Annotated[str, 20]


@dataclass
class InputTokenSpec:
    arg: Annotated[TokenArgument, 10]


@dataclass(repr=False)
class InputSpec(_Union):
    user_input: Annotated[UserInputSpec, 10]
    parameter: Annotated[InputToParameterSpec, 20]
    buffer: Annotated[InputToBufferSpec, 30]
    tensor_constant: Annotated[InputToTensorConstantSpec, 40]
    custom_obj: Annotated[InputToCustomObjSpec, 50]
    token: Annotated[InputTokenSpec, 70]
    constant_input: Annotated[InputToConstantInputSpec, 60]


@dataclass
class UserOutputSpec:
    arg: Annotated[Argument, 10]


@dataclass
class LossOutputSpec:
    arg: Annotated[TensorArgument, 10]


@dataclass
class BufferMutationSpec:
    arg: Annotated[TensorArgument, 10]
    buffer_name: Annotated[str, 20]


@dataclass
class GradientToParameterSpec:
    arg: Annotated[TensorArgument, 10]
    parameter_name: Annotated[str, 20]


@dataclass
class GradientToUserInputSpec:
    arg: Annotated[TensorArgument, 10]
    user_input_name: Annotated[str, 20]


@dataclass
class UserInputMutationSpec:
    arg: Annotated[TensorArgument, 10]
    user_input_name: Annotated[str, 20]


@dataclass
class OutputTokenSpec:
    arg: Annotated[TokenArgument, 10]


@dataclass(repr=False)
class OutputSpec(_Union):
    user_output: Annotated[UserOutputSpec, 10]
    loss_output: Annotated[LossOutputSpec, 20]
    buffer_mutation: Annotated[BufferMutationSpec, 30]
    gradient_to_parameter: Annotated[GradientToParameterSpec, 40]
    gradient_to_user_input: Annotated[GradientToUserInputSpec, 50]
    user_input_mutation: Annotated[UserInputMutationSpec, 60]
    token: Annotated[OutputTokenSpec, 70]


@dataclass
class GraphSignature:
    input_specs: Annotated[List[InputSpec], 10]
    output_specs: Annotated[List[OutputSpec], 20]


@dataclass
class RangeConstraint:
    min_val: Annotated[Optional[int], 10]
    max_val: Annotated[Optional[int], 20]


@dataclass
class ModuleCallSignature:
    inputs: Annotated[List[Argument], 10]
    outputs: Annotated[List[Argument], 20]

    # These are serialized by calling pytree.treespec_loads
    # And deserialized by calling pytree.treespec_dumps
    in_spec: Annotated[str, 30]
    out_spec: Annotated[str, 40]

    # This field is used to prettify the graph placeholders
    # after we ser/der and retrace
    forward_arg_names: Annotated[Optional[List[str]], 50] = None


@dataclass
class ModuleCallEntry:
    fqn: Annotated[str, 10]
    signature: Annotated[Optional[ModuleCallSignature], 30] = None


@dataclass
class GraphModule:
    graph: Annotated[Graph, 10]
    signature: Annotated[GraphSignature, 50]
    # This is used for unflattening, by tracking the calling structure of all of
    # the modules in order to unflatten the modules back to the eager calling
    # conventions.
    module_call_graph: Annotated[List[ModuleCallEntry], 60]
    metadata: Annotated[Dict[str, str], 40] = field(default_factory=dict)


# Invariant: Every time a change is made to the schema, one of the versions
#            should be upadted.
@dataclass
class SchemaVersion:
    major: Annotated[int, 10]  # Major version number is bumped every time a breaking change is made.
    minor: Annotated[int, 20]  # Minor version number is bumped when a compatible change is made.


@dataclass
class ExportedProgram:
    graph_module: Annotated[GraphModule, 10]
    # Key is the opset namespace (ex. aten), and value is the version number
    opset_version: Annotated[Dict[str, int], 20]
    range_constraints: Annotated[Dict[str, RangeConstraint], 30]
    schema_version: Annotated[SchemaVersion, 60]
    verifiers: Annotated[List[str], 70] = field(default_factory=list)
    torch_version: Annotated[str, 80] = "<=2.4"
