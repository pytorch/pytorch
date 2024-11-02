import torch
from torch.utils._pytree import tree_flatten
from torch.fx import Graph
from torch.utils._ordered_set import OrderedSet

from .collector import get_args_of_node_type, get_fake_tensor_from_node
from .chunking_subgraph import ChunkingSubgraph

class Partitioner:
    @classmethod
    def reorder_nodes(cls, graph, chunking_subgraph_nodes) -> ChunkingSubgraph:
        """
        Create a new graph to be like:
        1. all nodes run before `chunking_subgraph_nodes`
        2. all nodes in `chunking_subgraph_nodes`
        3. all nodes run after `chunking_subgraph_nodes`

        Right now, we just reorder and return a copy of the graph.
        But later we could leverage ideas from HigherOrderOp and
        embed the chunking subgraph inside the enclosing graph.
        """

        # `pre_chunking_nodes` are all nodes that only depends on
        # nodes insideo `pre_chuning_nodes`
        pre_chunking_nodes = OrderedSet()

        for node in graph.nodes:
            if node in chunking_subgraph_nodes:
                continue
            if all(arg in pre_chunking_nodes and arg not in chunking_subgraph_nodes for arg in get_args_of_node_type(node)):
                pre_chunking_nodes.add(node)

        post_chunking_nodes = []

        def _copy_node(typestr, node):
            fake_tensor = get_fake_tensor_from_node(node)
            shape = list(fake_tensor.shape) if fake_tensor is not None else "?"
            print(f" - {typestr}: {shape} {node.format_node()}")
            env[node] = new_graph.node_copy(node, lambda x: env[x]) 
            return env[node]

        # add pre_chunking_nodes
        new_graph = Graph()
        env = {}
        for node in pre_chunking_nodes:
            _copy_node("prechunking", node)

        # add nodes in the chunking subgraph
        new_chunking_subgraph_nodes = OrderedSet()
        for node in graph.nodes:
            if node in pre_chunking_nodes:
                continue
            elif node in chunking_subgraph_nodes:
                _copy_node("chunking", node)
                new_chunking_subgraph_nodes.add(env[node])
            else:
                post_chunking_nodes.append(node)

        for node in post_chunking_nodes:
            _copy_node("postchuking", node)

        assert graph._len == new_graph._len
        new_graph.eliminate_dead_code()
        new_graph.lint()

        return ChunkingSubgraph(new_graph, new_chunking_subgraph_nodes)
