import enum
from typing import NamedTuple

from torch.fx.graph import Node

class NSSingleResultValuesType(str, enum.Enum):
    WEIGHT = 'weight'
    NODE_OUTPUT = 'node_output'
    NODE_INPUT = 'node_input'

NSSubgraph = NamedTuple(
    'NSSubgraph',
    [('start_node', Node), ('end_node', Node), ('base_op_node', Node)]
)
