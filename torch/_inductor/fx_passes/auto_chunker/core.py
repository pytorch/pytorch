# mypy: allow-untyped-defs
from typing import Optional

import torch
from torch._inductor import config
from torch.fx import Graph, Node
from torch.utils._ordered_set import OrderedSet

from .common import ChunkingMeta
from .utils import (
    compute_tensor_size,
    get_args_of_node_type,
    get_fake_tensor_from_node_arg,
    use_tangent,
)


aten = torch.ops.aten
log = torch._logging.getArtifactLogger(__name__, "auto_chunker")


def set_chunking_meta(node, meta=None, **kwargs):
    """
    kwargs can override fields in the passed in `meta`
    """
    if meta is None:
        meta = ChunkingMeta(**kwargs)
    else:
        # make a copy to avoid override the passed in instance
        meta = meta.copy()
        for k, v in kwargs.items():
            setattr(meta, k, v)

    old_meta = get_chunking_meta(node)
    node.meta["chunking"] = meta
    return old_meta is None or old_meta != meta


def update_chunking_meta(node, **kwargs):
    """
    Unlike set_chunking_mete, this function keeps the existing chunking
    metadata if it's not overriden.
    """
    changed = False
    meta = get_chunking_meta(node)
    if meta is None:
        meta = ChunkingMeta()
        changed = True
    for k, v in kwargs.items():
        if getattr(meta, k, None) != v:
            changed = True
        setattr(meta, k, v)

    node.meta["chunking"] = meta
    return changed


def set_chunking_meta_if_none(nodes, meta, filter_for_nop=None):
    """
    If filter_fop_nop returns true for a node, we set the chunking
    meta to nop instead.
    """
    changed = False
    for node in nodes:
        if get_chunking_meta(node) is None:
            changed = True
            if filter_for_nop and filter_for_nop(node):
                set_chunking_meta(node)
            else:
                set_chunking_meta(node, meta)
    return changed


def copy_chunking_meta(dst_node, src_node):
    if isinstance(src_node, torch.fx.Node):
        src_meta = get_chunking_meta(src_node)
    else:
        assert isinstance(src_node, ChunkingMeta)
        src_meta = src_node
    assert src_meta
    return set_chunking_meta(dst_node, **src_meta.__dict__)


def get_chunking_meta(node):
    return node.meta.get("chunking")


def has_nop_chunking_meta(node):
    return ChunkingMeta.is_nop(get_chunking_meta(node))


def get_chunking_metas(nodes, skip_none=False):
    return [
        get_chunking_meta(node)
        for node in nodes
        if not skip_none or get_chunking_meta(node) is not None
    ]


eligible_amplifier_node = OrderedSet(
    [
        aten.mm.default,
        aten.addmm.default,
    ]
)


def find_amplifier_node(graph: Graph) -> Optional[Node]:
    r"""
    Find the 'amplifier' node which is a not that generates large
    output with small/medium input.

    If there are multiple amplifier nodes, return the one with the largest
    amplification ratio.
    """

    amplifier_nodes_ratio = []
    for node in graph.nodes:
        # We only look for amplifier nodes in the fwd part of the graph
        if use_tangent(node):
            break

    for node in graph.nodes:
        if use_tangent(node):
            # enter backward part of the graph
            break

        # Only trigger chunking for a small set of nodes like matmul for now
        if node.op != "call_function" or node.target not in eligible_amplifier_node:
            continue

        input_size = compute_tensor_size(node.args, node.kwargs)
        output_size = compute_tensor_size(node)

        if input_size == 0:
            continue

        ratio = output_size / input_size
        if (
            output_size > config.AutoChunker.output_size_threshold
            and ratio > config.AutoChunker.amplify_ratio_threshold
        ):
            amplifier_nodes_ratio.append((node, ratio))

    amplifier_nodes_ratio = sorted(
        amplifier_nodes_ratio, key=lambda x: x[1], reverse=True
    )
    return amplifier_nodes_ratio[0][0] if len(amplifier_nodes_ratio) > 0 else None


def reorder_nodes(graph):
    """
    Create a new graph to be like:
    1. all nodes run before `chunking_subgraph_nodes`
    2. all nodes in `chunking_subgraph_nodes`
    3. all nodes run after `chunking_subgraph_nodes`

    Return a new graph so it's easier to fallback.
    """
    from .applier import is_chunking_subgraph_input

    # `pre_chunking_nodes` are all nodes that only depends on
    # nodes inside `pre_chuning_nodes`
    pre_chunking_nodes: OrderedSet[str] = OrderedSet()

    for node in graph.nodes:
        if node.op == "placeholder" or is_chunking_subgraph_input(node):
            # these nodes have chunking meta but they should be placed
            # before we do chunking
            pre_chunking_nodes.add(node)

        if get_chunking_meta(node) is not None:
            continue
        if all(arg in pre_chunking_nodes for arg in get_args_of_node_type(node)):
            pre_chunking_nodes.add(node)

    post_chunking_nodes = []

    def _copy_node(typestr, node):
        fake_tensor = get_fake_tensor_from_node_arg(node)
        shape = list(fake_tensor.shape) if fake_tensor is not None else "?"
        log.debug(" - %s: %s %s", typestr, shape, node.format_node())
        env[node] = new_graph.node_copy(node, lambda x: env[x])
        return env[node]

    # add pre_chunking_nodes
    new_graph = Graph()
    env: dict[Node, Node] = {}
    for node in pre_chunking_nodes:
        _copy_node("prechunking", node)

    # add nodes in the chunking subgraph
    new_chunking_subgraph_nodes: OrderedSet[Node] = OrderedSet()
    for node in graph.nodes:
        if node in pre_chunking_nodes:
            continue
        elif get_chunking_meta(node):
            _copy_node("chunking", node)
            new_chunking_subgraph_nodes.add(env[node])
        else:
            post_chunking_nodes.append(node)

    for node in post_chunking_nodes:
        _copy_node("postchuking", node)

    assert graph._len == new_graph._len
    new_graph.eliminate_dead_code()
    new_graph.lint()

    # Need replace the scale_by node in the chunking meta with the new node
    for node in new_graph.nodes:
        meta = get_chunking_meta(node)
        if meta and meta.scale_by is not None:
            meta.scale_by = env[meta.scale_by]

    return new_graph
