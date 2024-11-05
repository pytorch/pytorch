from torch.utils._ordered_set import OrderedSet
from torch.fx import Node
import operator
import torch
from torch.utils import _pytree

from .propagator import get_chunking_meta
from .collector import get_fake_tensor_from_node
from torch._dynamo.utils import detect_fake_mode
from torch.fx.passes.fake_tensor_prop import FakeTensorProp

aten = torch.ops.aten

def _factory_args(fake_tensor):
    return {
        "device": fake_tensor.device,
        "dtype": fake_tensor.dtype,
    }

def fake_tensor_prop(gm):
    inputs = []
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            fake_tensor = get_fake_tensor_from_node(node)
            if fake_tensor is not None:
                inputs.append(fake_tensor)
            else:
                inputs.append(node)

    fake_mode = detect_fake_mode(inputs)
    FakeTensorProp(gm, mode=fake_mode).propagate_dont_convert_inputs(*inputs)


class UnrollChunkingApplier:
    def __init__(self, chunking_subgraph, num_chunk):
        self.chunking_subgraph = chunking_subgraph
        self.graph = self.chunking_subgraph.parent_graph
        self.source_user = self.chunking_subgraph.source_user
        self.nodes_to_recover = self.chunking_subgraph.nodes_to_recover
        self.num_chunk = num_chunk
        self.chunk_sizes = None

        # First index is node index,
        # Second index is chunk index.
        self.chunked_external_nodes: List[List[Node]] = []
        self.chunks_for_recovering: List[List[Node]] = [
            [] for _ in range(len(self.nodes_to_recover))
        ]

    def chunk_external_nodes(self):
        with self.graph.inserting_before(self.source_user):
            for node_idx, external_node in enumerate(self.chunking_subgraph.external_nodes_to_chunk):
                self.chunked_external_nodes.append(self._chunk_external_node(external_node))

    def _chunk_external_node(self, external_node):
        chunk_node = self.graph.call_function(aten.chunk.default, (external_node, self.num_chunk))
        chunks = []
        for i in range(self.num_chunk):
            chunks.append(self.graph.call_function(operator.getitem, (chunk_node, i)))

        # let's do a small meta progation to get the size of each chunk
        if self.chunk_sizes is None:
            input_tensor = external_node.meta["val"]
            tensors = torch.chunk(input_tensor, self.num_chunk)
            self.chunk_sizes = [
                t.size(0) for t in tensors
            ]
        return chunks

    def _recreate_node_with_replacement(self, original_node, env):
        new_args, new_kwargs = _pytree.tree_map(lambda node: env.get(node, node), (original_node.args, original_node.kwargs))

        replacement = self.graph.call_function(original_node.target,
            new_args, new_kwargs)
        return replacement

    def _create_chunked_subgraph(self, chunk_id):
        env = {}

        # Fill env with chunked external nodes
        for chunks, external_node in zip(self.chunked_external_nodes, self.chunking_subgraph.external_nodes_to_chunk):
            env[external_node] = chunks[chunk_id]
        
        # Run thru the chunking_subgraph_nodes
        with self.graph.inserting_before(self.source_user):
            for original_node in self.chunking_subgraph.subgraph_nodes:
                if original_node.op == "placeholder":
                    continue

                # Chunk aten.full
                if original_node.target == aten.full.default and (meta := get_chunking_meta(original_node)) is not None and meta.chunk_dim is not None:
                    shape = list(original_node.args[0])
                    shape[meta.chunk_dim] = self.chunk_sizes[chunk_id]
                    env[original_node] = self.graph.call_function(
                        aten.full.default,
                        (shape, original_node.args[1]),
                        original_node.kwargs
                    )
                    continue

                # create the node with chunked inputs
                env[original_node] = self._recreate_node_with_replacement(original_node, env)
        # collect the chunks for recovering
        for idx, node in enumerate(self.nodes_to_recover):
            self.chunks_for_recovering[idx].append(env[node])

    def create_chunked_subgraph(self):
        for chunk_id in range(self.num_chunk):
            self._create_chunked_subgraph(chunk_id)

    def replace_tangent_to_one(self):
        tangent_node = next(iter(self.chunking_subgraph.subgraph_nodes))
        assert tangent_node.op == "placeholder" and "tangent" in tangent_node.target

        fake_tensor = tangent_node.meta["val"]

        with self.graph.inserting_before(self.source_user):
            one = self.graph.call_function(aten.full.default, (fake_tensor.shape, 1), _factory_args(fake_tensor))
        one._rename(f"tangent_overriden_as_one")
        tangent_node.replace_all_uses_with(one)

    def _recover_to_unchunked_node(self, node, chunks):
        """
        Recover the node from chunks and do the replacement.
        """

        meta = get_chunking_meta(node)
        assert meta is not None

        recovered = node

        if meta.chunk_dim is not None:
            recovered = self.graph.call_function(
                aten.cat.default,
                (chunks, meta.chunk_dim))
        elif meta.need_sum:
            recovered = chunks[0]
            for other in chunks[1:]:
                recovered = self.graph.call_function(
                    aten.add.Tensor, (recovered, other))

        # do scaling last
        if meta.scale_by is not None:
            recovered = self.graph.call_function(
                aten.mul.Tensor, (recovered, meta.scale_by)
            )

        assert recovered is not node
        node.replace_all_uses_with(recovered)

    def recover_to_unchunked_nodes(self):
        with self.graph.inserting_before(self.source_user):
            for node, chunks in zip(self.nodes_to_recover, self.chunks_for_recovering):
                self._recover_to_unchunked_node(node, chunks)

    def erase_original_nodes(self):
        # Traveral reversely to erase user first
        for node in reversed(tuple(self.chunking_subgraph.subgraph_nodes)):
            if node.op == "placeholder":
                continue

            self.graph.erase_node(node)
        
    def apply(self):
        self.replace_tangent_to_one()
        self.chunk_external_nodes()
        self.create_chunked_subgraph()
        self.recover_to_unchunked_nodes()
        self.erase_original_nodes()

# We would try putting the chunked graph into a subgraph and call that
# subgraph for each chunk. Will create a separate Applier for that.
ChunkingApplier = UnrollChunkingApplier
