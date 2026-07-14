import json

from torch._export.serde.schema import ExternKernelNode, ExternKernelNodes, Node
from torch._export.serde.serialize import _dataclass_to_dict, EnumEncoder
from torch._inductor.ir import ExternKernelNode as inductor_ExternKernelNode


def serialize_extern_kernel_node(
    extern_kernel_node: inductor_ExternKernelNode,
) -> ExternKernelNode:
    assert isinstance(extern_kernel_node.node, Node)
    return ExternKernelNode(
        name=extern_kernel_node.name,
        node=extern_kernel_node.node,
    )


def extern_node_json_serializer(
    extern_kernel_nodes: list[inductor_ExternKernelNode],
) -> str:
    serialized_nodes = ExternKernelNodes(
        nodes=[serialize_extern_kernel_node(node) for node in extern_kernel_nodes]
    )
    return json.dumps(_dataclass_to_dict(serialized_nodes), cls=EnumEncoder)
