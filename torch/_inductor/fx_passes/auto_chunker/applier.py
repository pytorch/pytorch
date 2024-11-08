from torch.utils._ordered_set import OrderedSet
from torch.fx import Node, Graph
import operator
import torch
import copy
from torch.utils import _pytree

from .propagator import get_chunking_meta
from .collector import get_fake_tensor_from_node
from torch._dynamo.utils import detect_fake_mode
from torch.fx.passes.fake_tensor_prop import FakeTensorProp

aten = torch.ops.aten
prims = torch.ops.prims

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

class BaseChunkingApplier:
    def __init__(self, parent_gm, chunking_subgraph, num_chunk):
        self.chunking_subgraph = chunking_subgraph
        self.num_chunk = num_chunk
        self.graph = self.chunking_subgraph.parent_graph
        self.parent_gm = parent_gm
        self.source_user = self.chunking_subgraph.source_user
        self.nodes_to_recover = self.chunking_subgraph.nodes_to_recover

        self.overriden_tangent = None

        self.chunk_sizes = None

        # First index is node index,
        # Second index is chunk index.
        self.chunked_external_nodes: List[List[Node]] = []
        self.chunks_for_recovering: Dict[Node, List[Node]] = {}
        self.accumulators: Dict[Node, Node] = {}
        for node in self.nodes_to_recover:
            meta = get_chunking_meta(node)
            assert meta
            if meta.chunk_dim is not None:
                self.chunks_for_recovering[node] = []
            else:
                # the accumulator nodes is created later
                self.accumulators[node] = None


    def replace_tangent_to_one(self):
        tangent_node = next(iter(self.chunking_subgraph.subgraph_nodes))
        assert tangent_node.op == "placeholder" and "tangent" in tangent_node.target

        fake_tensor = tangent_node.meta["val"]

        with self.graph.inserting_before(self.source_user):
            one = self.graph.call_function(aten.full.default, (fake_tensor.shape, 1), _factory_args(fake_tensor))
        one._rename(f"tangent_overriden_as_one")
        one.meta = copy.copy(tangent_node.meta)
        tangent_node.replace_all_uses_with(one)
        self.overriden_tangent = one

    def _recover_to_unchunked_node(self, node):
        """
        Recover the node from chunks and do the replacement.
        """

        meta = get_chunking_meta(node)
        assert meta is not None

        recovered = node

        if meta.chunk_dim is not None:
            chunks = self.chunks_for_recovering[node]
            recovered = self.graph.call_function(
                aten.cat.default,
                (chunks, meta.chunk_dim))
        elif meta.need_sum:
            recovered = self.accumulators[node]

        # do scaling last
        if meta.scale_by is not None:
            recovered = self.graph.call_function(
                aten.mul.Tensor, (recovered, meta.scale_by)
            )

        # convert back to the original dtype
        if meta.need_sum:
            original_dtype = node.meta["val"].dtype
            if original_dtype != torch.float32:
                recovered = self.graph.call_function(
                    prims.convert_element_type.default,
                    (recovered, original_dtype)
                )

        assert recovered is not node
        node.replace_all_uses_with(recovered)

    def recover_to_unchunked_nodes(self):
        with self.graph.inserting_before(self.source_user):
            for node in self.nodes_to_recover:
                self._recover_to_unchunked_node(node)

    def erase_original_nodes(self):
        # Traveral reversely to erase user first
        for node in reversed(tuple(self.chunking_subgraph.subgraph_nodes)):
            if node.op == "placeholder":
                continue

            self.graph.erase_node(node)

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


class SubgraphChunkingApplier(BaseChunkingApplier):
    def build_subgraph(self, chunk_size):
        new_graph = Graph()
        env = {}

        def _create_placeholder_node(external_node):
            new_node = new_graph.placeholder(external_node.name)
            fake_tensor = external_node.meta["val"]
            chunking_meta = get_chunking_meta(external_node)
            if chunking_meta and chunking_meta.chunk_dim is not None:
                # the node is chunked and we need update the 
                # fake tensor
                # TODO any better way to do this?
                new_tensor = aten.slice.Tensor(fake_tensor,
                    chunking_meta.chunk_dim,
                    0,
                    chunk_size
                )
                fake_tensor = new_tensor
            new_node.meta = {
                "val": fake_tensor
            }
            return new_node

        for node_idx, external_node in enumerate(self.chunking_subgraph.external_nodes):
            env[external_node] = _create_placeholder_node(external_node)
        env[self.overriden_tangent] = _create_placeholder_node(self.overriden_tangent)
        for _, accum in self.accumulators.items():
            env[accum] = _create_placeholder_node(accum)

        for original_node in self.chunking_subgraph.subgraph_nodes:
            if original_node.op == "placeholder":
                continue

            # Chunk aten.full
            if original_node.target == aten.full.default and (meta := get_chunking_meta(original_node)) is not None and meta.chunk_dim is not None:
                shape = list(original_node.args[0])
                shape[meta.chunk_dim] = chunk_size
                env[original_node] = new_graph.call_function(
                    aten.full.default,
                    (shape, original_node.args[1]),
                    original_node.kwargs
                )
                continue
            # Chunk aten.expand a scalar
            if original_node.target == aten.expand.default and original_node.args[0].meta["val"].numel() == 1 and (meta := get_chunking_meta(original_node)) is not None and meta.chunk_dim is not None:
                shape = list(original_node.args[1])
                shape[meta.chunk_dim] = chunk_size
                env[original_node] = new_graph.call_function(
                    aten.expand.default,
                    (env.get(original_node.args[0], original_node.args[0]), shape),
                    original_node.kwargs
                )
                continue

            # create the node with chunked inputs
            print(f"Create node in new_graph for: {original_node.format_node()}") # TODO
            env[original_node] = new_graph.node_copy(original_node, lambda x: env[x])

        # Do the accumulation inside this subgraph
        for node, accum in self.accumulators.items():
            lhs = env[node]
            rhs = env[accum]

            # add `addend` and `accum`
            add_out = new_graph.call_function(
                aten.add.Tensor, (lhs, rhs)
            )

            # override the chunk value
            env[node] = add_out

        out_values = []
        for node in self.nodes_to_recover:
            out_values.append(env[node])
        new_graph.output(tuple(out_values))
        new_graph.eliminate_dead_code()
        new_graph.lint()

        sub_gm = torch.fx._lazy_graph_module._make_graph_module(
            self.parent_gm, new_graph
        )
        print(f"sub gm:\n{sub_gm.print_readable(False)}")

        assert chunk_size not in self.chunk_size_to_gm_attr
        gm_attr = f"chunking_subgraph_{len(self.chunk_size_to_gm_attr)}"
        self.chunk_size_to_gm_attr[chunk_size] = gm_attr
        setattr(self.parent_gm, gm_attr, sub_gm)

        fake_tensor_prop(sub_gm)

        # Mark this sub graph module so we don't recursively chunking
        # it.
        sub_gm.meta["produced_by_chunker"] = True
        return sub_gm
    
    def build_subgraphs(self):
        assert self.chunk_sizes
        for chunk_size in self.chunk_sizes:
            if chunk_size not in self.chunk_size_to_gm_attr:
                self.build_subgraph(chunk_size)

    def __init__(self, parent_gm, chunking_subgraph, num_chunk):
        super().__init__(parent_gm, chunking_subgraph, num_chunk)
        self.subgraph_outputs = []

        self.chunk_size_to_gm_attr = {}

    def _call_subgraph(self, sub_gm, chunk_id):
        args = []
        
        chunks_itr = iter(self.chunked_external_nodes)
        for node in self.chunking_subgraph.external_nodes:
            if get_chunking_meta(node): # being chunked
                args.append(next(chunks_itr)[chunk_id])
            else:
                # not chunked
                args.append(node)
        args.append(self.overriden_tangent)

        for _, accum in self.accumulators.items():
            args.append(accum)

        output_node = self.graph.call_function(torch.ops.higher_order.invoke_subgraph,
                (sub_gm, None, args), {})

        output_node_dict = {}
        for i, orig_node in enumerate(self.nodes_to_recover):
            output_node_dict[orig_node] = self.graph.call_function(operator.getitem, (output_node, i))

        for orig_node, node_list in self.chunks_for_recovering.items():
            chunk = output_node_dict[orig_node]
            node_list.append(chunk)

        for orig_node in self.accumulators:
            self.accumulators[orig_node] = output_node_dict[orig_node]


    def call_subgraph_for_each_chunk(self):
        with self.graph.inserting_before(self.source_user):
            for chunk_id in range(self.num_chunk):
                chunk_size = self.chunk_sizes[chunk_id]
                sub_gm = self.graph.get_attr(self.chunk_size_to_gm_attr[chunk_size])
                self._call_subgraph(sub_gm, chunk_id)

    def create_accumulators(self):
        with self.graph.inserting_before(self.source_user):
            for node in self.accumulators:
                # use fp32 for accumulators
                fake_tensor = node.meta["val"]
                if fake_tensor.numel() == 1:
                    # This tensor maybe fused with mm and becomes a addmm
                    # if we upcast here, addmm may fail due to incompatible
                    # dtypes for input arguments.
                    override_dtype = torch.float32
                else:
                    override_dtype = fake_tensor.dtype
                    
                kwargs = _factory_args(fake_tensor)
                kwargs["dtype"] = override_dtype
                accum = self.graph.call_function(
                    aten.full.default,
                    (fake_tensor.shape, 0),
                    kwargs)
                # reuse the meta['val']
                accum.meta = {"val": fake_tensor.to(override_dtype)}
                self.accumulators[node] = accum
        

    def apply(self):
        self.replace_tangent_to_one()
        self.chunk_external_nodes()
        self.create_accumulators()
        self.build_subgraphs()

        # call subgraphs
        self.call_subgraph_for_each_chunk()

        self.recover_to_unchunked_nodes()
        self.erase_original_nodes()
        
        newgm = torch.fx._lazy_graph_module._make_graph_module(
            self.parent_gm, self.graph,
        )
        print(f"graph after calling subgraph:\n{self.graph}")
        fake_tensor_prop(newgm)
        return newgm


class UnrollChunkingApplier(BaseChunkingApplier):
    def __init__(self, parent_gm, chunking_subgraph, num_chunk):
        super().__init__(parent_gm, chunking_subgraph, num_chunk)

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
                # Chunk aten.expand a scalar
                if original_node.target == aten.expand.default and original_node.args[0].meta["val"].numel() == 1 and (meta := get_chunking_meta(original_node)) is not None and meta.chunk_dim is not None:
                    shape = list(original_node.args[1])
                    shape[meta.chunk_dim] = self.chunk_sizes[chunk_id]
                    env[original_node] = self.graph.call_function(
                        aten.expand.default,
                        (original_node.args[0], shape),
                        original_node.kwargs
                    )
                    continue

                # create the node with chunked inputs
                env[original_node] = self._recreate_node_with_replacement(original_node, env)
        # collect the chunks for recovering
        for idx, node in enumerate(self.nodes_to_recover):
            # self.chunks_for_recovering[idx].append(env[node])
            assert False

    def create_chunked_subgraph(self):
        for chunk_id in range(self.num_chunk):
            self._create_chunked_subgraph(chunk_id)

       
    def apply(self):
        self.replace_tangent_to_one()
        self.chunk_external_nodes()
        self.create_chunked_subgraph()
        self.recover_to_unchunked_nodes()
        self.erase_original_nodes()

        print(f"Graph after applying chunking:\n{self.graph}")

        newgm = torch.fx._lazy_graph_module._make_graph_module(
            self.parent_gm, self.graph,
        )
        fake_tensor_prop(newgm)
        breakpoint()
        return newgm

# We would try putting the chunked graph into a subgraph and call that
# subgraph for each chunk. Will create a separate Applier for that.
ChunkingApplier = SubgraphChunkingApplier
