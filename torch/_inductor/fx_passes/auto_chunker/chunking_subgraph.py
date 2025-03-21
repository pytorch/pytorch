from torch.utils._ordered_set import OrderedSet
from torch.fx import Node

class ChunkingSubgraph:
    def __init__(self, parent_graph, chunking_subgraph_nodes):
        self.parent_graph = parent_graph
        self.subgraph_nodes = chunking_subgraph_nodes
        self.external_nodes = OrderedSet()
        self.nodes_to_recover = self.find_nodes_to_recover(self.subgraph_nodes)
        self.source_user = next(node for node in chunking_subgraph_nodes if node.op != "placeholder")

    def add_external_node_to_chunk(self, external_node):
        self.external_nodes_to_chunk.add(external_node)

    @property
    def external_nodes_to_chunk(self):
        from .propagator import get_chunking_meta
        out = OrderedSet()
        for node in self.external_nodes:
            if get_chunking_meta(node) is not None:
                out.add(node)
        return out

    def add_external_node(self, external_node):
        self.external_nodes.add(external_node)

    @classmethod
    def find_nodes_to_recover(cls, chunking_subgraph_nodes: OrderedSet[Node]):
        """
        Nodes in chunking_subgraph_nodes that has external usage need
        to be recovered in the end.
        """
        to_recover = OrderedSet()
        for node in chunking_subgraph_nodes:
            if any(user not in chunking_subgraph_nodes for user in node.users):
                to_recover.add(node)
        return to_recover
