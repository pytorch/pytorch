from __future__ import annotations

import json
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from torch._export.serde.schema import ExternKernelNode
    from torch._inductor.ir import ExternKernelNode as inductor_ExternKernelNode


# torch._export.serde is imported lazily: GraphLowering always references
# extern_node_json_serializer, but it is only called for AOTInductor.
def serialize_extern_kernel_node(
    extern_kernel_node: inductor_ExternKernelNode,
) -> ExternKernelNode:
    from torch._export.serde.schema import ExternKernelNode, Node

    if not isinstance(extern_kernel_node.node, Node):
        raise AssertionError(
            f"expected node to be a Node, got {type(extern_kernel_node.node)}"
        )
    return ExternKernelNode(
        name=extern_kernel_node.name,
        node=extern_kernel_node.node,
    )


def extern_node_json_serializer(
    extern_kernel_nodes: list[inductor_ExternKernelNode],
) -> str:
    from torch._export.serde.schema import ExternKernelNodes
    from torch._export.serde.serialize import _dataclass_to_dict, EnumEncoder

    serialized_nodes = ExternKernelNodes(
        nodes=[serialize_extern_kernel_node(node) for node in extern_kernel_nodes]
    )
    return json.dumps(_dataclass_to_dict(serialized_nodes), cls=EnumEncoder)
