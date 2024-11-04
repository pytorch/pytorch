# NOTE: This is a placeholder for iterating on export serialization schema design.
#       Anything is subject to change and no guarantee is provided at this point.

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Tuple

from torch._export.serde.union import _Union

# NOTE: Please update this value if any modifications are made to the schema
SCHEMA_VERSION = (8, 1)
TREESPEC_VERSION = 1


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
    type: str
    index: Optional[int] = None


@dataclass(repr=False)
class SymExprHint(_Union):
    as_int: int
    as_float: float
    as_bool: bool


# This is for storing the symbolic expressions behind symints/symfloats/symbools
# For example, we can get something like
# SymExpr(expr_str="s0 + s1", hint=SymExprHint(as_int=4)
# if we also have the hint that s0 and s1 are both 2.
@dataclass
class SymExpr:
    expr_str: str
    hint: Optional[SymExprHint] = None


@dataclass(repr=False)
class SymInt(_Union):
    as_expr: SymExpr
    as_int: int


@dataclass(repr=False)
class SymBool(_Union):
    as_expr: SymExpr
    as_bool: bool


@dataclass
class TensorMeta:
    dtype: ScalarType
    sizes: List[SymInt]
    requires_grad: bool
    device: Device
    strides: List[SymInt]
    storage_offset: SymInt
    layout: Layout


# In most cases we will use the "as_name" field to store arguments which are
# SymInts.
# The "as_int" field is used in the case where we have a list containing a mix
# of SymInt and ints (ex. [1, s0, ...]). We will serialize this type of list to
# be List[SymIntArgument] and map the SymInts to the "as_name" field, and ints
# to the "as_int" field.
@dataclass(repr=False)
class SymIntArgument(_Union):
    as_name: str
    as_int: int


# In most cases we will use the "as_name" field to store arguments which are
# SymBools.
# The "as_bool" field is used in the case where we have a list containing a mix
# of SymBool and bools (ex. [True, i0, ...]). We will serialize this type of list to
# be List[SymboolArgument] and map the SymBools to the "as_name" field, and bools
# to the "as_bool" field.
@dataclass(repr=False)
class SymBoolArgument(_Union):
    as_name: str
    as_bool: bool


@dataclass
class TensorArgument:
    name: str


@dataclass
class TokenArgument:
    name: str


# This is use for storing the contents of a list which contain optional tensors
# (Tensor?[], ex. [Tensor, None, ...]), where the list will be serialized to the
# type List[OptionalTensorArgument], with tensor values seiralized to the
# "as_tensor" field, and None values serialized to the "as_none" field.
@dataclass(repr=False)
class OptionalTensorArgument(_Union):
    as_tensor: TensorArgument
    as_none: Tuple[()]


@dataclass
class GraphArgument:
    name: str
    graph: 'Graph'


@dataclass
class CustomObjArgument:
    name: str
    class_fqn: str


# This is actually a union type
@dataclass(repr=False)
class Argument(_Union):
    as_none: Tuple[()]
    as_tensor: TensorArgument
    as_tensors: List[TensorArgument]
    as_int: int
    as_ints: List[int]
    as_float: float
    as_floats: List[float]
    as_string: str
    as_strings: List[str]
    as_sym_int: SymIntArgument
    as_sym_ints: List[SymIntArgument]
    as_scalar_type: ScalarType
    as_memory_format: MemoryFormat
    as_layout: Layout
    as_device: Device
    as_bool: bool
    as_bools: List[bool]
    as_sym_bool: SymBoolArgument
    as_sym_bools: List[SymBoolArgument]
    as_graph: GraphArgument
    as_optional_tensors: List[OptionalTensorArgument]
    as_custom_obj: CustomObjArgument
    as_operator: str


@dataclass
class NamedArgument:
    # Argument name from the operator schema
    name: str
    arg: Argument


@dataclass
class Node:
    target: str
    inputs: List[NamedArgument]
    outputs: List[Argument]
    metadata: Dict[str, str]


@dataclass
class Graph:
    inputs: List[Argument]
    outputs: List[Argument]
    nodes: List[Node]
    tensor_values: Dict[str, TensorMeta]
    sym_int_values: Dict[str, SymInt]
    sym_bool_values: Dict[str, SymBool]
    # This is for deserializing the submodule graphs from higher order ops
    # (ex. cond, map) where single tensor returns will just return a single
    # tensor, rather than following export schema and returning a singleton
    # list.
    is_single_tensor_return: bool = False
    custom_obj_values: Dict[str, CustomObjArgument] = field(default_factory=dict)


@dataclass
class UserInputSpec:
    # Actually, only tensors and SymInts are allowed here
    arg: Argument


@dataclass(repr=False)
class ConstantValue(_Union):
    as_none: Tuple[()]
    as_int: int
    as_float: float
    as_string: str
    as_bool: bool


@dataclass
class ConstantInputSpec:
    name: str
    value: ConstantValue


@dataclass
class InputToParameterSpec:
    arg: TensorArgument
    parameter_name: str


@dataclass
class InputToBufferSpec:
    arg: TensorArgument
    buffer_name: str
    persistent: bool



@dataclass
class InputToTensorConstantSpec:
    arg: TensorArgument
    tensor_constant_name: str


@dataclass
class InputToCustomObjSpec:
    arg: CustomObjArgument
    custom_obj_name: str


@dataclass
class InputTokenSpec:
    arg: TokenArgument


@dataclass(repr=False)
class InputSpec(_Union):
    user_input: UserInputSpec
    parameter: InputToParameterSpec
    buffer: InputToBufferSpec
    tensor_constant: InputToTensorConstantSpec
    custom_obj: InputToCustomObjSpec
    token: InputTokenSpec
    constant_input: ConstantInputSpec


@dataclass
class UserOutputSpec:
    arg: Argument


@dataclass
class LossOutputSpec:
    arg: TensorArgument


@dataclass
class BufferMutationSpec:
    arg: TensorArgument
    buffer_name: str


@dataclass
class GradientToParameterSpec:
    arg: TensorArgument
    parameter_name: str


@dataclass
class GradientToUserInputSpec:
    arg: TensorArgument
    user_input_name: str


@dataclass
class UserInputMutationSpec:
    arg: TensorArgument
    user_input_name: str


@dataclass
class OutputTokenSpec:
    arg: TokenArgument


@dataclass(repr=False)
class OutputSpec(_Union):
    user_output: UserOutputSpec
    loss_output: LossOutputSpec
    buffer_mutation: BufferMutationSpec
    gradient_to_parameter: GradientToParameterSpec
    gradient_to_user_input: GradientToUserInputSpec
    user_input_mutation: UserInputMutationSpec
    token: OutputTokenSpec


@dataclass
class GraphSignature:
    input_specs: List[InputSpec]
    output_specs: List[OutputSpec]


@dataclass
class RangeConstraint:
    min_val: Optional[int]
    max_val: Optional[int]


@dataclass
class ModuleCallSignature:
    inputs: List[Argument]
    outputs: List[Argument]

    # These are serialized by calling pytree.treespec_loads
    # And deserialized by calling pytree.treespec_dumps
    in_spec: str
    out_spec: str

    # This field is used to prettify the graph placeholders
    # after we ser/der and retrace
    forward_arg_names: Optional[List[str]] = None


@dataclass
class ModuleCallEntry:
    fqn: str
    signature: Optional[ModuleCallSignature] = None


@dataclass
class GraphModule:
    graph: Graph
    signature: GraphSignature
    # This is used for unflattening, by tracking the calling structure of all of
    # the modules in order to unflatten the modules back to the eager calling
    # conventions.
    module_call_graph: List[ModuleCallEntry]
    metadata: Dict[str, str] = field(default_factory=dict)


# Invariant: Every time a change is made to the schema, one of the versions
#            should be upadted.
@dataclass
class SchemaVersion:
    major: int  # Major version number is bumped every time a breaking change is made.
    minor: int  # Minor version number is bumped when a compatible change is made.


@dataclass
class ExportedProgram:
    graph_module: GraphModule
    # Key is the opset namespace (ex. aten), and value is the version number
    opset_version: Dict[str, int]
    range_constraints: Dict[str, RangeConstraint]
    schema_version: SchemaVersion
    verifiers: List[str] = field(default_factory=list)
    torch_version: str = "<=2.4"
