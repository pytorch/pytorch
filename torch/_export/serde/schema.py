# NOTE: This is a placeholder for iterating on export serialization schema design.
#       Anything is subject to change and no guarantee is provided at this point.

from dataclasses import dataclass, fields
from enum import Enum
from typing import Dict, List, Optional, Tuple

# TODO (zhxchen17) Move to a separate file.
class _Union:
    @classmethod
    def create(cls, **kwargs):
        assert len(kwargs) == 1
        return cls(**{**{field.name: None for field in fields(cls)}, **kwargs})

    def __post_init__(self):
        assert sum(1 for field in fields(self) if getattr(self, field.name) is not None) == 1

class ScalarType(Enum):
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


class Layout(Enum):
    Unknown = 0
    SparseCoo = 1
    SparseCsr = 2
    SparseCsc = 3
    SparseBsr = 4
    SparseBsc = 5
    _mkldnn = 6
    Strided = 7


class MemoryFormat(Enum):
    Unknown = 0
    ContiguousFormat = 1
    ChannelsLast = 2
    ChannelsLast3d = 3
    PreserveFormat = 4


@dataclass
class Device:
    type: str
    index: Optional[int]


@dataclass
class SymInt(_Union):
    as_symbol: str
    as_int: int


@dataclass
class TensorMeta:
    dtype: ScalarType
    sizes: List[SymInt]
    requires_grad: bool
    device: Device
    strides: List[SymInt]
    storage_offset: int
    layout: Layout


@dataclass
class SymIntArgument(_Union):
    as_name: str
    as_int: int


@dataclass
class TensorArgument:
    name: str


# This is actually a union type
@dataclass
class Argument(_Union):
    as_none: Tuple[()]
    as_tensor: TensorArgument
    as_tensors: List[TensorArgument]
    as_int: int
    as_ints: List[int]
    as_float: float
    as_floats: List[float]
    as_string: str
    as_sym_int: SymIntArgument
    as_sym_ints: List[SymIntArgument]
    as_scalar_type: ScalarType
    as_memory_format: MemoryFormat
    as_layout: Layout
    as_device: Device
    as_bool: bool
    as_bools: List[bool]


@dataclass
class NamedArgument:
    name: str
    arg: Argument


@dataclass
class Operator:
    name: str
    version: Optional[int]


@dataclass
class Node:
    target: Operator
    inputs: List[NamedArgument]
    outputs: List[Argument]
    metadata: Dict[str, str]


@dataclass
class TensorValue:
    meta: TensorMeta


@dataclass
class Graph:
    inputs: List[Argument]
    outputs: List[Argument]
    nodes: List[Node]
    tensor_values: Dict[str, TensorValue]
    sym_int_values: Dict[str, SymInt]


@dataclass
class BackwardSignature:
    gradients_to_parameters: Dict[str, str]
    gradients_to_userInputs: Dict[str, str]
    loss_output: str


@dataclass
class GraphSignature:
    inputs_to_parameters: Dict[str, str]
    inputs_to_buffers: Dict[str, str]
    user_inputs: List[str]
    user_outputs: List[str]
    buffers_to_mutate: Dict[str, str]
    in_spec: str
    out_spec: str
    backward_signature: Optional[BackwardSignature]


@dataclass
class GraphModule:
    graph: Graph
    buffers: Dict[str, TensorMeta]
    parameters: Dict[str, TensorMeta]
    metadata: Dict[str, str]
    signature: GraphSignature
