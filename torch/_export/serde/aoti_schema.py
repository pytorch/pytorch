from dataclasses import dataclass

from torch._export.serde.schema import Node


@dataclass
class ExternKernelNode:
    name: str
    node: Node


@dataclass
class ExternKernelNodes:
    nodes: list[ExternKernelNode]
