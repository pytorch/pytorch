from collections.abc import Sequence
from typing import Any, Optional

import torch
from torch.fx import Graph, GraphModule, Node
from torch.utils._pytree import tree_flatten
from .common import CantChunk, ChunkingMeta


def is_tangent_node(node: Node) -> bool:
    return node.op == "placeholder" and "tangent" in node.name


def get_args_of_node_type(node: Node) -> Sequence[Node]:
    return [x for x in tree_flatten((node.args, node.kwargs))[0] if isinstance(x, Node)]


def use_tangent(node: Node) -> bool:
    """
    Whether the fx node uses tangent input.
    """

    return any(
        is_tangent_node(arg)  # type: ignore[operator]
        for arg in get_args_of_node_type(node)
    )


def compute_tensor_size(*args: Any, count_bytes: bool = True, **kwargs: Any) -> int:
    """
    Compute total tensor sizes from fx.Node in args & kwargs.
    """
    flat_args, _ = tree_flatten((args, kwargs))
    tot = 0
    for arg in flat_args:
        if (fake_tensor := get_fake_tensor_from_node_arg(arg)) is None:
            continue
        tot += fake_tensor.numel() * (fake_tensor.dtype.itemsize if count_bytes else 1)
    return tot


def get_fake_tensor_from_node_arg(
    node: torch.fx.node.Argument,
) -> Optional[torch.Tensor]:
    if (
        not hasattr(node, "meta")
        or ("val" not in node.meta)  # type: ignore[union-attr]
        or not isinstance(node.meta["val"], torch.Tensor)  # type: ignore[union-attr]
    ):
        return None
    return node.meta["val"]  # type: ignore[union-attr]


def get_nodes_with_chunking_meta(graph: torch.fx.Graph) -> Sequence[Node]:
    from .core import get_chunking_meta

    output = []
    for node in graph.nodes:
        if get_chunking_meta(node):
            output.append(node)
    return output


def format_node_with_chunking_meta(
    node: torch.fx.Node, include_args: bool = False
) -> None:
    """
    Print the node with chunking metadata for the current node if exists.

    If include_args is True, also print chuning metadata for Node arguments.
    """
    from torch._inductor.runtime.runtime_utils import green_text
    from .core import get_chunking_meta

    fake_tensor = get_fake_tensor_from_node_arg(node)
    shape = list(fake_tensor.shape) if fake_tensor is not None else "?"
    print(f"  {shape} {node.format_node()}")

    if meta := get_chunking_meta(node):
        print(f"    {green_text(str(meta))}")

    if include_args:
        for arg in get_args_of_node_type(node):
            if arg_meta := get_chunking_meta(arg):
                print(f"    {arg}: {green_text(str(arg_meta))}")


def has_any_chunking_meta(*node_list: Node) -> bool:
    from .core import get_chunking_meta

    return any(get_chunking_meta(node) for node in node_list)


def get_first_chunking_meta(*node_list: Node) -> Optional[ChunkingMeta]:
    """
    Get the first non-none chunking metadata if there is any.
    """
    from .core import get_chunking_meta

    for node in node_list:
        if (meta := get_chunking_meta(node)) is not None:
            return meta

    return None


def get_scale_by_from_metas(*metas: ChunkingMeta) -> Optional[Node]:
    """
    If there are multiple ChunkingMeta having the scale_by field,
    raise a CantChunk exception.

    If no ChunkingMeta has scale_by field, return None.
    Other wise return the only scale_by field.
    """

    scale_by_list = []

    # don't do dedup on the scale_by field on purpose for this API
    for meta in metas:
        if meta.scale_by is not None:
            scale_by_list.append(meta.scale_by)

    if len(scale_by_list) > 1:
        raise CantChunk("Multiple scale_by")

    return scale_by_list[0] if len(scale_by_list) == 1 else None


def get_scale_by_from_node(node: Node) -> Optional[Node]:
    from .core import get_chunking_meta

    meta = get_chunking_meta(node)
    return meta.scale_by if meta is not None else None


def get_node_is_scalar(nodes: Sequence[Node]) -> dict[Node, bool]:
    """
    Returns a dict map a node to 'is_scalar'.
    """
    node_is_scalar = {}
    for node in nodes:
        ft = get_fake_tensor_from_node_arg(node)
        assert ft is not None
        node_is_scalar[node] = ft.numel() == 1
    return node_is_scalar


def get_node_ndim(nodes: Sequence[Node]) -> dict[Node, int]:
    """
    Returns a dict map a node to 'ndim'.
    """
    node_ndim = {}
    for node in nodes:
        ft = get_fake_tensor_from_node_arg(node)
        assert ft is not None
        node_ndim[node] = ft.ndim
    return node_ndim


def is_chunked_by_dim(node: Node, dim: int) -> bool:
    from .core import get_chunking_meta

    meta = get_chunking_meta(node)
    return meta is not None and meta.chunked_by_dim(dim)


def tangent_has_chunking_meta(gm: GraphModule) -> bool:
    from .core import get_chunking_meta

    return any(
        is_tangent_node(node) and get_chunking_meta(node) is not None
        for node in gm.graph.find_nodes(op="placeholder", sort=False)
    )


def get_tangent_nodes(graph: Graph) -> Sequence[Node]:
    tangents = []
    for node in graph.find_nodes(op="placeholder", sort=False):
        if is_tangent_node(node):
            tangents.append(node)
    return tangents
