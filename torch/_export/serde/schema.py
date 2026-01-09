# NOTE: This is a placeholder for iterating on export serialization schema design.
#       Anything is subject to change and no guarantee is provided at this point.

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Annotated, Optional

from torch._export.serde.union import _Union, _union_dataclass


# NOTE: Please update this value if any modifications are made to the schema
SCHEMA_VERSION = (8, 16)
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
    FLOAT8E4M3FNUZ = 31
    FLOAT8E5M2FNUZ = 32


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


@_union_dataclass
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


@_union_dataclass
class SymInt(_Union):
    as_expr: Annotated[SymExpr, 10]
    as_int: Annotated[int, 20]


@_union_dataclass
class SymFloat(_Union):
    as_expr: Annotated[SymExpr, 10]
    as_float: Annotated[float, 20]


@_union_dataclass
class SymBool(_Union):
    as_expr: Annotated[SymExpr, 10]
    as_bool: Annotated[bool, 20]


@dataclass
class TensorMeta:
    dtype: Annotated[ScalarType, 10]
    sizes: Annotated[list[SymInt], 20]
    requires_grad: Annotated[bool, 30]
    device: Annotated[Device, 40]
    strides: Annotated[list[SymInt], 50]
    storage_offset: Annotated[SymInt, 60]
    layout: Annotated[Layout, 70]


# In most cases we will use the "as_name" field to store arguments which are
# SymInts.
# The "as_int" field is used in the case where we have a list containing a mix
# of SymInt and ints (ex. [1, s0, ...]). We will serialize this type of list to
# be List[SymIntArgument] and map the SymInts to the "as_name" field, and ints
# to the "as_int" field.
@_union_dataclass
class SymIntArgument(_Union):
    as_name: Annotated[str, 10]
    as_int: Annotated[int, 20]


# In most cases we will use the "as_name" field to store arguments which are
# SymFloats.
# The "as_float" field is used in the case where we have a list containing a mix
# of SymFloat and float (ex. [1.0, s0, ...]). We will serialize this type of list to
# be List[SymFloatArgument] and map the SymFloats to the "as_name" field, and ints
# to the "as_float" field.
@_union_dataclass
class SymFloatArgument(_Union):
    as_name: Annotated[str, 10]
    as_float: Annotated[float, 20]


# In most cases we will use the "as_name" field to store arguments which are
# SymBools.
# The "as_bool" field is used in the case where we have a list containing a mix
# of SymBool and bools (ex. [True, i0, ...]). We will serialize this type of list to
# be List[SymboolArgument] and map the SymBools to the "as_name" field, and bools
# to the "as_bool" field.
@_union_dataclass
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
# type List[OptionalTensorArgument], with tensor values serialized to the
# "as_tensor" field, and None values serialized to the "as_none" field.
@_union_dataclass
class OptionalTensorArgument(_Union):
    as_tensor: Annotated[TensorArgument, 20]
    as_none: Annotated[bool, 10]


@dataclass
class GraphArgument:
    name: Annotated[str, 10]
    graph: Annotated["Graph", 20]


@dataclass
class CustomObjArgument:
    name: Annotated[str, 10]
    class_fqn: Annotated[str, 20]


@dataclass
class ComplexValue:
    real: Annotated[float, 10]
    imag: Annotated[float, 20]


# This is actually a union type
@_union_dataclass
class Argument(_Union):
    as_none: Annotated[bool, 10]
    as_tensor: Annotated[TensorArgument, 20]
    as_tensors: Annotated[list[TensorArgument], 30]
    as_int: Annotated[int, 50]
    as_ints: Annotated[list[int], 70]
    as_float: Annotated[float, 80]
    as_floats: Annotated[list[float], 90]
    as_string: Annotated[str, 100]
    as_strings: Annotated[list[str], 101]
    as_sym_int: Annotated[SymIntArgument, 110]
    as_sym_ints: Annotated[list[SymIntArgument], 120]
    as_scalar_type: Annotated[ScalarType, 130]
    as_memory_format: Annotated[MemoryFormat, 140]
    as_layout: Annotated[Layout, 150]
    as_device: Annotated[Device, 160]
    as_bool: Annotated[bool, 170]
    as_bools: Annotated[list[bool], 180]
    as_sym_bool: Annotated[SymBoolArgument, 182]
    as_sym_bools: Annotated[list[SymBoolArgument], 184]
    as_graph: Annotated[GraphArgument, 200]
    as_optional_tensors: Annotated[list[OptionalTensorArgument], 190]
    as_custom_obj: Annotated[CustomObjArgument, 210]
    as_operator: Annotated[str, 220]
    as_sym_float: Annotated[SymFloatArgument, 230]
    as_sym_floats: Annotated[list[SymFloatArgument], 240]
    as_optional_tensor: Annotated[OptionalTensorArgument, 250]
    as_complex: Annotated[ComplexValue, 260]
    as_int_lists: Annotated[list[list[int]], 280]
    as_string_to_argument: Annotated[dict[str, "Argument"], 290]


class ArgumentKind(IntEnum):
    UNKNOWN = 0
    POSITIONAL = 1
    KEYWORD = 2


@dataclass
class NamedArgument:
    # Argument name from the operator schema
    name: Annotated[str, 10]
    arg: Annotated[Argument, 20]
    kind: Annotated[Optional[ArgumentKind], 30] = None


@dataclass
class Node:
    target: Annotated[str, 10]
    inputs: Annotated[list[NamedArgument], 20]
    outputs: Annotated[list[Argument], 30]
    metadata: Annotated[dict[str, str], 40]
    is_hop_single_tensor_return: Annotated[Optional[bool], 50] = None
    # For BC, default is None so older serialized models without 'name' can be loaded.
    name: Annotated[Optional[str], 60] = None


@dataclass
class Graph:
    inputs: Annotated[list[Argument], 10]
    outputs: Annotated[list[Argument], 20]
    nodes: Annotated[list[Node], 30]
    tensor_values: Annotated[dict[str, TensorMeta], 40]
    sym_int_values: Annotated[dict[str, SymInt], 50]
    sym_bool_values: Annotated[dict[str, SymBool], 60]
    # This is for deserializing the submodule graphs from higher order ops
    # (ex. cond, map) where single tensor returns will just return a single
    # tensor, rather than following export schema and returning a singleton
    # list.
    is_single_tensor_return: Annotated[bool, 70] = False
    custom_obj_values: Annotated[dict[str, CustomObjArgument], 80] = field(
        default_factory=dict
    )
    sym_float_values: Annotated[dict[str, SymFloat], 90] = field(default_factory=dict)


@dataclass
class UserInputSpec:
    # Actually, only tensors and SymInts are allowed here
    arg: Annotated[Argument, 10]


@_union_dataclass
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


@_union_dataclass
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
class ParameterMutationSpec:
    arg: Annotated[TensorArgument, 10]
    parameter_name: Annotated[str, 20]


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


@_union_dataclass
class OutputSpec(_Union):
    user_output: Annotated[UserOutputSpec, 10]
    loss_output: Annotated[LossOutputSpec, 20]
    buffer_mutation: Annotated[BufferMutationSpec, 30]
    gradient_to_parameter: Annotated[GradientToParameterSpec, 40]
    gradient_to_user_input: Annotated[GradientToUserInputSpec, 50]
    user_input_mutation: Annotated[UserInputMutationSpec, 60]
    token: Annotated[OutputTokenSpec, 70]
    parameter_mutation: Annotated[ParameterMutationSpec, 80]


@dataclass
class GraphSignature:
    input_specs: Annotated[list[InputSpec], 10]
    output_specs: Annotated[list[OutputSpec], 20]


@dataclass
class RangeConstraint:
    min_val: Annotated[Optional[int], 10]
    max_val: Annotated[Optional[int], 20]


@dataclass
class ModuleCallSignature:
    inputs: Annotated[list[Argument], 10]
    outputs: Annotated[list[Argument], 20]

    # These are serialized by calling pytree.treespec_loads
    # And deserialized by calling pytree.treespec_dumps
    in_spec: Annotated[str, 30]
    out_spec: Annotated[str, 40]

    # This field is used to prettify the graph placeholders
    # after we Ser/Der and retrace
    forward_arg_names: Annotated[Optional[list[str]], 50] = None


@dataclass
class ModuleCallEntry:
    fqn: Annotated[str, 10]
    signature: Annotated[Optional[ModuleCallSignature], 30] = None


@dataclass
class NamedTupleDef:
    field_names: Annotated[list[str], 10]


@dataclass
class GraphModule:
    graph: Annotated[Graph, 10]
    signature: Annotated[GraphSignature, 50]
    # This is used for unflattening, by tracking the calling structure of all of
    # the modules in order to unflatten the modules back to the eager calling
    # conventions.
    module_call_graph: Annotated[list[ModuleCallEntry], 60]
    metadata: Annotated[dict[str, str], 40] = field(default_factory=dict)
    # Mapping of namedtuple types to namedtuple field names, used for BC
    treespec_namedtuple_fields: Annotated[dict[str, NamedTupleDef], 70] = field(
        default_factory=dict
    )


# Invariant: Every time a change is made to the schema, one of the versions
#            should be updated.
@dataclass
class SchemaVersion:
    major: Annotated[
        int, 10
    ]  # Major version number is bumped every time a breaking change is made.
    minor: Annotated[
        int, 20
    ]  # Minor version number is bumped when a compatible change is made.


@dataclass
class ExportedProgram:
    graph_module: Annotated[GraphModule, 10]
    # Key is the opset namespace (ex. aten), and value is the version number
    opset_version: Annotated[dict[str, int], 20]
    range_constraints: Annotated[dict[str, RangeConstraint], 30]
    schema_version: Annotated[SchemaVersion, 60]
    verifiers: Annotated[list[str], 70] = field(default_factory=list)
    torch_version: Annotated[str, 80] = "<=2.4"
    guards_code: Annotated[list[str], 90] = field(default_factory=list)


#########################################################################
# Container types for inference tasks, not being used directly for export.
#########################################################################


# The metadata for payload saved in PT2 archive.
# payload includes params, buffers, tensor constants, and custom objects.
@dataclass
class PayloadMeta:
    # the path of the payload in the archive file, e.g. "weight_0"
    path_name: Annotated[str, 10]
    is_param: Annotated[bool, 20]
    # whether the payload is serialized using pickle.
    # Only custom objects and tensor subclasses that are not fake tensors
    # are serialized using pickle.
    use_pickle: Annotated[bool, 30]
    # Custom Objects don't have tensor_meta and will be serialized using pickle
    tensor_meta: Annotated[Optional[TensorMeta], 40]


# The mapping from payload FQN to its metadata.
@dataclass
class PayloadConfig:
    config: Annotated[dict[str, PayloadMeta], 10]


#
# The structure is used to serialize instances of AOTInductorModel to pass
# them from the publishing pipeline to the predictor.
#
# All new fields should be marked as optional.
#
@dataclass
class AOTInductorModelPickleData:
    # Base name of an associated .so AOTInductor library. Typically looks like:
    # "abc.so".
    library_basename: Annotated[str, 1]

    # AOTInductor engine input names.
    input_names: Annotated[list[str], 2]

    # AOTInductor engine output names.
    output_names: Annotated[list[str], 3]

    # These fields tell whether floating point inputs/outputs should be converted to
    # a certain type. If None, the dtypes that the AOTInductor engine inferred from the sample
    # inputs are used.
    floating_point_input_dtype: Annotated[Optional[int], 4] = None
    floating_point_output_dtype: Annotated[Optional[int], 5] = None

    # Whether AOTInductor runtime is for CPU.
    aot_inductor_model_is_cpu: Annotated[Optional[bool], 6] = None


@dataclass
class ExternKernelNode:
    # name is not the unique identifier of the node
    name: Annotated[str, 10]
    node: Annotated[Node, 20]


@dataclass
class ExternKernelNodes:
    nodes: Annotated[list[ExternKernelNode], 10]
