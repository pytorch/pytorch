from torch.utils._ordered_set import OrderedSet
from torch.fx import Node
import operator
import torch

aten = torch.ops.aten

class UnrollChunkingApplier:
    def __init__(self, chunking_subgraph, num_chunk):
        self.chunking_subgraph = chunking_subgraph
        self.graph = self.chunking_subgraph.parent_graph
        self.source_user = self.chunking_subgraph.source_user
        self.num_chunk = num_chunk

        # First index is node index,
        # Second index is chunk index.
        self.chunked_external_nodes: List[List[Node]] = []

    def chunk_external_nodes(self):
        with self.graph.inserting_before(self.source_user):
            for node_idx, external_node in enumerate(self.chunking_subgraph.external_nodes_to_chunk):
                self.chunked_external_nodes.append(self._chunk_external_node(external_node))

    def _chunk_external_node(self, external_node):
        chunk_node = self.graph.call_function(aten.chunk.default, (external_node, self.num_chunk))
        chunks = []
        for i in range(self.num_chunk):
            chunks.append(self.graph.call_function(operator.getitem, (chunk_node, i)))
        return chunks

    def create_chunked_subgraph(self):
        breakpoint()

    # TODO override tangent to be 1
    def apply(self):
        self.chunk_external_nodes()
        self.create_chunked_subgraph()
        self.recover_to_unchunked_nodes()
        breakpoint()

# We would try putting the chunked graph into a subgraph and call that
# subgraph for each chunk. Will create a separate Applier for that.
ChunkingApplier = UnrollChunkingApplier
