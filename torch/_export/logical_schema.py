# type: ignore[assignment]

from dataclasses import dataclass
from enum import auto, Enum
from typing import List, Union, Dict

################################################################################
# Following section is the defining the permissible argument types for operators

# Copied from torchgen/model.py
class ScalarType(Enum):
    u8 = auto()     # torch.uint8
    i8 = auto()     # torch.int8
    i16 = auto()    # torch.int16 or torch.short
    i32 = auto()    # torch.int32 or torch.int
    i64 = auto()    # torch.int64 or torch.long
    f16 = auto()    # torch.float16 or torch.half
    f32 = auto()    # torch.float32 or torch.float
    f64 = auto()    # torch.float64 or torch.double
    c32 = auto()    # torch.complex32
    c64 = auto()    # torch.complex64 or torch.cfloat
    c128 = auto()   # torch.complex128 or torch.cdouble
    b8 = auto()     # torch.bool
    bf16 = auto()   # torch.bfloat16

# Copied from torch/_C/__init__.pyi.in
class Layout(Enum):
    # Defined in torch/csrc/utils/tensor_layouts.cpp
    strided = auto()
    sparse_coo = auto()
    sparse_csr = auto()
    sparse_csc = auto()
    sparse_bsr = auto()
    sparse_bsc = auto()
    _mkldnn = auto()


# Copied from torch/_C/__init__.pyi.in
class MemoryFormat(Enum):
    # Defined in torch/csrc/utils/tensor_memoryformats.cpp
    contiguous_format = auto()
    channels_last = auto()
    channels_last_3d = auto()
    preserve_format = auto()


# Copied from torch/_C/__init__.pyi.in
@dataclass
class Device:
    # Defined in torch/csrc/Device.cpp
    type: str
    index: int

@dataclass
class SymInt:  # Union, ONLY EXACTLY ONE of the following fields can be set
    as_int: int = None
    as_sym: str = None

# !!! To support t.item(), we need to introduce SymFloat
# @dataclass
# class SymFloat:  # Union, ONLY EXACTLY ONE of the following fields can be set
#     as_flaot: float = None
#     as_sym: str = None


# This is a Tensor Arugment used in the args of an node
# We intentionally don't store the tensor's storage, nor the tensor's meta data here,
# as the same tensor argument can be used in multiple nodes, and we want to avoid storing the same data multiple times.
# In another word, this field is an reference to the tensor, not the tensor itself.
@dataclass
class TensorArgument:
    name: str   # identifier of the tensor, which must exist in graph's tensor_values

# This is a SymInt Arugment used in the args of an node
# We intentionally don't store the SymInt's value here, as the same SymInt argument can be used in multiple nodes
# This field is an reference to the SymInt
@dataclass
class SymIntArgument:
    name: str   # identifier of the symint, which must exist in graph's symint_values

#  Permissible return types for operators
# !!! Notice: this assumes that a node can only return Tensor(s) and Symint(s), and not other int/float/bool types...
# !!! What about .item()? Do we need to handle this?
@dataclass
class ReturnArgument:  # Union, ONLY EXACTLY ONE of the following fields can be set
    as_tensor: TensorArgument = None

    # !!! ATM, no operator has return type as Tensor[], might need this latter?
    # as_tensors: List[TensorArgument] = None

    as_symint: SymIntArgument = None


# Permissible argument types for operators
# !!! This is a Union struct, but there is no good python construct to model this
@dataclass
class Argument:  # Union, ONLY EXACTLY ONE of the following fields can be set
    # A special type for representing python None in the arguments
    # This must only be used for ops that accepts None as an argument, e.g. Tensor?, Scalar?, int?, int[]?
    as_none: bool = None

    as_tensor: TensorArgument = None
    as_tensors: List[TensorArgument] = None   # Tensor[], used by aten.cat, and condition ops

    as_symint: SymIntArgument = None         # Symint can be an argument, there are symint in native_function.yaml
    as_symints: List[SymIntArgument] = None   # Symint[] can be an argement, there are symint[] in native_function.yaml

    as_bool: bool = None

    # !!! There are use of bool[3] in canonical aten ops, consider if we can simplify this
    as_bools: List[bool] = None     # for bool[]

    as_int: int = None
    as_ints: List[int] = None      # for int[]
    as_float: float = None
    as_floats: List[float] = None    # for float[]
    as_str: str = None
    # List[str],        # !!! There is no str[] in native_function.yaml. Consider if this is needed for expressiveness

    # Graph,            # !!! Consider how to handle condition op, which need to pass in a graph for the branch
    # List[Graph],      # !!! What about list of graphs? Do we need this?
    as_gm: "GraphModule" = None     # !!! ATM, torch.cond models branch as GraphModule

    # !!! Following types doesn't have a list version in native_function.yaml
    as_scalar_type: ScalarType = None
    as_memory_format: MemoryFormat = None
    as_layout: Layout = None
    as_device: Device = None


################################################################################
# Following section is the defining the schema of serializing a concrete tensor

# TensorMeta is a decription of a tensor, without the actual data (,effectively maps to FakeTensor)
# TensorMeta has multliple uses
#   1. Represent the property of a concrete tensor backed by a storage
#     - This is used in the serialization of a concrete tensor, e.g. model weight
#     - In this case, sizes and strides must be concrete ints, and cannot be symbolic
#     - stride and storage_offset have to used to correctly reconstruct the tensor from the storage
#   2. Represent the property of a virtual tensor (see TensorValue below)
#     - In this case, sizes and strides can be either concrete ints or symbolic ints.
#     - device/strides/storage_offset/layout/memory_format are tied to pytorch's implementation.
#       These are faithful capture of tensor's detail in pytorch's executions during tracing
#       However, it's up to downstream system on how to utilized these fields
#       In another word, these feilds are suggestive, rather than mandatory.


@dataclass
class TensorMeta:
    dtype: ScalarType
    sizes: List[SymInt]

    # needed for training
    requires_grad: bool

    # !!! see description above, there are subtle difference on how these fields should be interpreted
    device: Device
    strides: List[SymInt]
    storage_offset: SymInt
    layout: Layout


@dataclass
class Buffer:
    # data stored in big endian
    buffer: bytes


# External data needs to stored in big endian
@dataclass
class ExternalBuffer:
    location: str
    offset: str     # !!! Consider using int, but int has int_max limitation
    length: str     # !!! Consider using int, but int has int_max limitation
    checksum: str


@dataclass
class Storage:
    class DataLocation(Enum):
        Internal = auto()
        External = auto()

    data_location: DataLocation
    data: Union[Buffer, ExternalBuffer]


# This is a concrete tensor backed by storage
@dataclass
class Tensor:
    # storage
    storage: Storage

    # metadata
    meta: TensorMeta


################################################################################
# Following section is defining the schema of 3 level construct: GraphModule, Graph, Node

# TensorValue has no corresponding class in fx
# TensorValue is the "tensor results" that are passed between nodes in the graph
# TensorValue is a named virtual tensor, with an TensorMeta that describes the properties of the tensor
@dataclass
class TensorValue:
    name: str           # unique identifier of the TensorValue, referenced in Argument.as_tensor field
    meta: TensorMeta    # tensor meta


@dataclass
class SymIntValue:
    name: str       # unique identifier of the SymIntValue, referenced in Argument.as_symint field
    value: SymInt

@dataclass
class NodeMetadata:
    stack_trace: str                      # source info of a node
    nn_module_stack: str                  # stack of nn.Module that the node originates from
    extra: Dict[str, str]                 # arbitrary string-string pairs for extra metadata


# Maps to fx.Node
# Node can only be 'call_function' ops
# 'placeholder' and 'output' are serialized as inputs and outputs of the Graph
# 'get_attr' is not needed anymore, as it's an implicit lookup from GraphModule's parameters/buffers
# 'call_method' and 'call_module' is not supported, as it's not used in the canonical FX Graph
@dataclass
class Node:
    # fully qualified name to the target, e.g. aten.add.Tensnor
    # !!! Consider using a structured operator name instead of string
    target: str

    args: List[Argument]

    # kwargs for this node
    # !!! Not all types in Argument are used as kwargs, e.g. TensorArgument should not be used as kwargs
    # Do we want to enforce this in the schema? i.e. only allow certain types to be used as kwargs?
    kwargs: Dict[str, Argument]

    # A list of Argument returned by this node
    outputs: List[ReturnArgument]

    metadata: NodeMetadata          # metadata fields for this node


# Maps to fx.Graph
@dataclass(init=False)
class Graph:
    # Maps to fx.graph's placeholder nodes.
    # !!! Do we allow SymInt as graph input?
    # !!! need to think about where to store the metadata for placeholder nodes
    inputs: List[TensorArgument]

    # Maps to fx.graph's output node.
    # !!! Do we allow SymInt as graph output?
    # !!! need to thinking about where to store the metadata for original output node
    outputs: List[TensorArgument]

    # maps to computations nodes in fx.graph
    # Placeholder nodes and output node are not included in this list.
    # Only call_function can be included in this list
    nodes: List[Node]

    # Tensor values that appear in the graph
    # They could be graph inputs, graph outputs, or intermediate tensor values produced by nodes
    tensor_values: List[TensorValue]

    # SymInt values that appear in the graph
    symint_values: List[SymIntValue]


# Maps to fx.GraphModule
# This the top level construct for the model
@dataclass(init=False)
class GraphModule:
    # A readable name for the model, potentially maps to GraphModule's self.__class__.__name__
    # This is not an identified for GraphModule
    name: str

    graph: Graph    # Only one Graph per GraphModule

    # maps to GraphModule's meta, which is a Dict[str, Any], but we only support string key and string value.
    metadata : Dict[str, str]

    # Stateful fields of the graph module

    # The name of the tensor will be used to bind to the TensorValues of Graph
    # !!! Consider storing them in the Graph.
    # There are functional difference between buffers and parameters, so they are stored separately.
    parameters: Dict[str, Tensor]
    buffers: Dict[str, Tensor]

    # !!! model constants: constant, etc.

    # !!! Might also need to store the shape_env for symints, but it's unclear how downstream system will use it.
    # !!! Consider storing it in the GraphModule, or in the Graph.
