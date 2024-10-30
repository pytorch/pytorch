"""
Collect the source node to chunk and all the nodes that should be put in
the chunking subgraph.
"""

import torch
from torch.utils._pytree import tree_flatten
from typing import Sequence
from torch.fx import Graph, Node
from torch._inductor import config
import itertools

aten = torch.ops.aten
prims = torch.ops.prims

class CantChunk(RuntimeError):
    pass

# Right now only allow matmul/addmm as the source nodes for chunking.
# May extend it to more ops.
# The index points to the argument of the node that should be chunked.
eligible_source_node_op_to_idx = {
    aten.mm.default: 0,
    # arg[0] is the bias. We should chunk arg[1] which is the input
    # matrix for the matmul.
    aten.addmm.default: 1,
}

def maybe_permuted(downstream_node, upstream_node):
    """
    Return true if downstream_node is upstream_node or a permutation of it
    """

    if downstream_node is upstream_node:
        return True

    if downstream_node.target == aten.permute.default and downstream_node.args[0] is upstream_node:
        return True
    return False


def node_is_returned(node):
    return any(n.op == "output" for n in node.users)

def get_args_of_node_type(node):
    return [x for x in tree_flatten((node.args, node.kwargs))[0]
        if isinstance(x, torch.fx.Node)]

def use_tangent(node: torch.fx.Node) -> bool:
    """
    Whether the fx node uses tangent input.
    """

    return any(
        arg.op == "placeholder" and "tangent" in arg.target
        for arg in get_args_of_node_type(node)
    )

def get_fake_tensor_from_node(node: torch.fx.Node) -> torch.Tensor:
    if (
        not hasattr(node, "meta")
        or ("val" not in node.meta)
        or not isinstance(node.meta["val"], torch.Tensor)
    ):
        return None
    return node.meta["val"]


def compute_tensor_size(*args, **kwargs):
    flat_args, _ = tree_flatten((args, kwargs))
    tot = 0
    for arg in flat_args:
        if (fake_tensor := get_fake_tensor_from_node(arg)) is None:
            continue
        tot += fake_tensor.numel() * fake_tensor.dtype.itemsize
    return tot

class Collector:
    @staticmethod
    def collect_source_users(graph: Graph) -> Sequence[Node]:
        r"""
        Find all candidate nodes for chunking in the forward part of the
        graph.

        The candidates are sorted in decreasing order of the amplification
        ratio.
        """
        # A source user is the user of a source node that we want to
        # chunk. The source node is node we start chunking.
        source_users = []
        for node in graph.nodes:
            if use_tangent(node):
                # enter backward part of the graph
                break

            # Only chunk a small set of source nodes like matmul for now
            if node.op != "call_function" or node.target not in eligible_source_node_op_to_idx:
                continue

            argidx = eligible_source_node_op_to_idx[node.target]
    
            input_size = compute_tensor_size(node.args, node.kwargs)
            output_size = compute_tensor_size(node)
    
            if (
                compute_tensor_size(node.args[argidx]) > config.AutoChunker.input_size_threshold
                and output_size / input_size > config.AutoChunker.amplify_ratio_threshold
            ):
                source_users.append((node, output_size / input_size))
    
        source_users = sorted(source_users, key=lambda x: x[1], reverse=True)
        return tuple(map(lambda x: x[0], source_users))


    @classmethod
    def collect_source_user(cls, graph: Graph):
        """ 
        Pick the one with the largest amplification factor for now.
        But an alternative reasonable strategy is to pick the one
        closest to the bwd part
        """
        source_nodes = cls.collect_source_users(graph)

        if len(source_nodes) == 0:
            raise CantChunk("No source nodes found.")

        return source_nodes[0]

    @classmethod
    def collect_source_node(cls, graph: Graph):
        source_user = cls.collect_source_user(graph)
        argidx = eligible_source_node_op_to_idx[source_user.target]
        return source_user.args[argidx]

    @classmethod
    def _find_reachable_nodes(cls, start_node, can_expand = lambda _: True):
        """
        Collect the nodes reachable from the `start_node` via the use chain.
        The output is topologically sorted.
        """
        reachable_nodes = []
        seen = set()
    
        def dfs(node):
            if node in seen:
                return
    
            seen.add(node)
    
            # no need to add the output node since the output node should not
            # see the effect of chunking.
            if node.op == "output":
                return
    
            if can_expand(node):
                for node2 in node.users:
                    dfs(node2)
    
                reachable_nodes.append(node)
            else:
                # If we are at leaves of expansion, only add the node if
                # it's not a view op
                if node.target != aten.view.default:
                    reachable_nodes.append(node)
    
        dfs(start_node)
    
        return list(reversed(reachable_nodes))

    @classmethod
    def collect_chunking_subgraph_nodes(cls, graph: torch.fx.Graph) -> Sequence[Node]:
        source_user = cls.collect_source_user(graph)
        argidx = eligible_source_node_op_to_idx[source_user.target]
        source_node = source_user.args[argidx]

        # TODO This check is specific to matmul/addmm. Need generalize
        # to support other kind of `source_node`
        def _can_expand(node):
            if node_is_returned(node):
                return False

            if node.target != aten.mm.default:
                return True
            
            if node == source_user:
                return True
           
            argidx = eligible_source_node_op_to_idx[source_user.target]
            source_matmul_inputs = source_user.args[argidx : argidx + 2]
            assert len(node.args) == 2
            cur_node_inputs = node.args

            for downstream_node, upstream_node in itertools.product(cur_node_inputs, source_matmul_inputs):
                if maybe_permuted(downstream_node, upstream_node):
                    # compute gradient of input/weight. Stop expanding
                    return False
            return True

        return cls._find_reachable_nodes(source_node, _can_expand)
