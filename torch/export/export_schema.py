from dataclasses import dataclass
from enum import auto, Enum
from typing import List, Union

################################################################################
# Following section is the defining a few utility data structures

# As many of the serialization library doesn't have the concept of map
# We use a list of sting-string pair to represent a map<string, string>
@dataclass
class StringStringPair:
    key: str
    value: str

################################################################################
# Following section is the defining the permissible argument types for operators

# Copied from torchgen/model.py
# !!! Consider if we want to add aliased dtypes, e.g float == float32. (Sherlock: No!)
class ScalarType(Enum):
    Byte = auto()
    Char = auto()
    Short = auto()
    Int = auto()
    Long = auto()
    Half = auto()
    Float = auto()
    Double = auto()
    ComplexHalf = auto()
    ComplexFloat = auto()
    ComplexDouble = auto()
    Bool = auto()
    BFloat16 = auto()

# Copied from torch/_C/__init__.pyi.in
# !!! Consider if we want to keep all of them, as we figures out what to do with SparseTensor
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

    # !!! This is annoying! preserve_format cannot be used as a tensor propery, as it can only used as an argument for operators
    preserve_format = auto()


# Copied from torch/_C/__init__.pyi.in
@dataclass
class Device:
    # Defined in torch/csrc/Device.cpp
    type: str
    index: int


SymInt = Union[
    int,
    str
]

Scalar = Union[int, float, bool]


# ArgumentType = Union[

#     Tensor
#     Tensor?
#     Tensor[]

#     Scalar
#     Scalar?

#     ScalarType
#     ScalarType?

#     bool
#     bool[3]
#     bool?

#     int
#     int?
#     int[]
#     int[]?

#     float
#     float[]?

#     SymInt
#     SymInt[]
#     SymInt[]?

#     MemoryFormat?

#     Layout
#     Layout?

#     Device
#     Device?
# ]

################################################################################
# Following section is the defining the schema of serializing a concrete tensor

# TensorMeta is a decription of a tensor, without the actual data (,effectively maps to FakeTensor)
# TensorMeta has multliple uses
#   1. Represent the property of a concrete tensor backed by a storage
#     - This is used in the serialization of a concrete tensor, e.g. model weight
#     - In this case, sizes and strides must be concrete ints, and cannot be symbolic
#     - stride and storage_offset have to used to correctly reconstruct the tensor from the storage
#   2. Represent the property of a virtual tensor (see IValue below)
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
    storage_offset: int
    layout: Layout
    memory_format: MemoryFormat


@dataclass
class Buffer:
    buffer: bytes


@dataclass
class ExternalBuffer:
    # !!! Example from onnx
    #   // Data can be stored inside the protobuf file using type-specific fields or raw_data.
    #   // Alternatively, raw bytes data can be stored in an external file, using the external_data field.
    #   // external_data stores key-value pairs describing data location. Recognized keys are:
    #   // - "location" (required) - POSIX filesystem path relative to the directory where the ONNX
    #   //                           protobuf model was stored
    #   // - "offset" (optional) - position of byte at which stored data begins. Integer stored as string.
    #   //                         Offset values SHOULD be multiples 4096 (page size) to enable mmap support.
    #   // - "length" (optional) - number of bytes containing data. Integer stored as string.
    #   // - "checksum" (optional) - SHA1 digest of file specified in under 'location' key.
    #   repeated StringStringEntryProto external_data
    location: str
    offset: str     # !!! Consider using int, but int has int_max limitation
    length: str     # !!! Consider using int, but int has int_max limitation
    checksum: str


@dataclass
class Storage:
    class DataLocation(Enum):
        Default = auto()
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


# !!! consider adding an optional named field to Tensor directly to avoid defining a new class
@dataclass
class NamedTesnsor:
    name: str
    tensor: Tensor

################################################################################
# Following section is the defining the schema of 3 level construct: GraphModule, Graph, Node

# IValue has no corresponding class in fx
# IValue is the "values" that are passed between nodes in the graph
# !!! Assumption: Only Tensor can be passed between nodes, and not other int/float/bool types...
# IValue is a named virtual tensor, with an optional TensorMeta that describes the properties of the tensor
# !!! Consider using a more descriptive name, e.g. TensorValue, TensorPlaceholder, TensorArgument, etc.
@dataclass
class IValue:
    name: str   # A unique identifider name for the IValue
                # The name will be used in the graph and node to refer to the IValue

    meta: TensorMeta


# Maps to fx.Node
@dataclass
class Node:
    op: str         # One of ['placeholder', 'call_function', 'get_attr', 'output'],
                    # call_method and call_module are not supported, as they shouldn't apprear in the Caononical FX Graph
                    # !!! Consider using an enum instead of string

    target: str      # fully qualified name to the target, e.g. aten.add.Tensnor
                     # !!! Consider using a structured operator name instead of string

    inputs: List[str]    # A list of IValue names consumed by this node

    # !!! what about other non-tensor inputs in args?
    # !!! what about kwargs?

    outputs: List[str]  # A list of IValue names that are produced by this node
                        # Notice: this assumes that a node can only return Tensor(s), and not other int/float/bool types...

    metadata: List[StringStringPair]


# Maps to fx.Graph
@dataclass
class Graph:

    inputs: List[Node]    # Maps to fx.graph's placeholder nodes.
                          # Placeholder nodes are stored separately, to clearly distinguish them from other nodes.
                          # !!! Consider making inputs a List[IValue] instead of List[Node], need to think about where to store the metadata for placeholder nodes

    outputs: Node   # Maps to fx.graph's output node.
                    # Output node are stored separately, to clearly distinguish them from other nodes.
                    # A graph can only have a single output node.
                    # !!! Consider making outputs a List[IValue] instead of a Node, need to thinking about where to store the metadata for output node

    nodes: List[Node]     # maps to remaining nodes in fx.graph
                          # Placeholder nodes and output node are not included in this list.

    # Tensor values that appear in the graph
    # They could be graph inputs, graph outputs, or intermediate values produced by nodes
    ivalues: List[IValue]


# Maps to fx.GraphModule
# This the top level construct for the model
@dataclass
class GraphModule:
    name: str       # A readable name for the model, potentially maps to GraphModule's self.__class__.__name__
                    # This is not an identified for GraphModule

    graph: Graph    # Only one Graph per GraphModule

    metadata : List[StringStringPair]   # maps to GraphModule's meta, which is a Dict[str, Any], but we only support string key and string value.

    # Need to store stateful information

    # The name of the tensor will be used to bind to the IValues of Graph Inputs
    # !!! Consider storing them in the GraphModule, or in the Graph.
    # !!! Do we needs to store parameters and buffers separately? (Sherlock: Ideally no.)
    parameters: List[NamedTesnsor]
    buffers: List[NamedTesnsor]

    # !!! model constants: constant, etc.

    # !!! Might also need to store the shape_env for symints, but it's unclear how downstream system will use it.
    # !!! Consider storing it in the GraphModule, or in the Graph.

