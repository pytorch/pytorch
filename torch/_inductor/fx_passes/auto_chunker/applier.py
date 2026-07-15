import copy
import logging
import operator
from typing import Any, Optional

import torch
from torch import Tensor
from torch._dynamo.utils import detect_fake_mode
from torch.fx import Graph, GraphModule, Node
from torch.fx.passes.fake_tensor_prop import FakeTensorProp

from .core import get_chunking_meta, reorder_nodes
from .utils import get_args_of_node_type, get_fake_tensor_from_node_arg, is_tangent_node


log = torch._logging.getArtifactLogger(__name__, "auto_chunker")
aten = torch.ops.aten
prims = torch.ops.prims


def _factory_args(fake_tensor: Tensor) -> dict[str, Any]:
    return {
        "device": fake_tensor.device,
        "dtype": fake_tensor.dtype,
    }


def fake_tensor_prop(gm: GraphModule) -> None:
    inputs = []
    for node in gm.graph.find_nodes(op="placeholder", sort=False):
        fake_tensor = get_fake_tensor_from_node_arg(node)
        if fake_tensor is not None:
            inputs.append(fake_tensor)
        else:
            inputs.append(node)

    fake_mode = detect_fake_mode(inputs)
    FakeTensorProp(gm, mode=fake_mode).propagate_dont_convert_inputs(*inputs)


def is_chunking_subgraph_input(node: Node) -> bool:
    meta = get_chunking_meta(node)
    if meta is None or is_tangent_node(node):
        return False
    arg_nodes = get_args_of_node_type(node)
    arg_nodes_no_meta = [node for node in arg_nodes if get_chunking_meta(node) is None]
    return len(arg_nodes_no_meta) > 0 or node.op == "placeholder"


class ChunkingApplier:
    """
    A class that chunks the graph assuming chunking metadata has already
    been attached to the nodes in the chunking subgraph.
    """

    def __init__(self, parent_gm: GraphModule, num_chunk: int):
        self.gm = parent_gm
        self.parent_graph = reorder_nodes(self.gm.graph)
        # From this point on self.parent_graph is not equal to self.gm.graph
        # due to reordering. We create a new copy of the graph so it's
        # easier to fallback if chunking fails.
        self.num_chunk = num_chunk

        # tangent node to the all-one tensor
        self.overriden_tangent: dict[Node, Optional[Node]] = {}

        self.subgraph_input: list[Node] = []
        self.subgraph_body: list[Node] = []
        self.subgraph_output: list[Node] = []
        self._categorize_subgraph_nodes()

        self.chunk_sizes: Optional[list[int]] = None

        # First index is node index,
        # Second index is chunk index.
        # Node index may be different to the index for self.subgraph_input
        # since not every subgraph_input may be chunked.
        # check self.chunk_subgraph_input for more details
        self.chunked_subgraph_input: list[list[Node]] = []

        self.accumulators: dict[Node, Optional[Node]] = {}
        self.chunks_for_recovering: dict[Node, list[Node]] = {}
        for node in self.subgraph_output:
            meta = get_chunking_meta(node)
            assert meta
            if meta.chunk_dim is not None:
                self.chunks_for_recovering[node] = []
            else:
                # the accumulator nodes is created later
                self.accumulators[node] = None

        self.chunk_size_to_gm_attr: dict[int, str] = {}

    def _categorize_subgraph_nodes(self) -> None:
        """
        For each chunked node, decide if it's a input/body/output node of the
        chunking subgraph.
        """
        for node in self.parent_graph.nodes:
            meta = get_chunking_meta(node)

            # The node does not have chunking metadata, skip
            if meta is None:
                continue

            # skip tangent nodes since they are overridden as 1 and
            # reapplied later in the bwd graph
            if is_tangent_node(node):
                self.overriden_tangent[node] = None  # will update the value later
                continue

            # To check if a node is a placeholder for the chunking
            # subgraph there are 2 alternatives
            # 1. decide it's a placeholder if it's using any non-chunked node as argument
            # 2. decide it's a placeholder if it's not using any chunked nodes as argument
            # These 2 rules are almost equivalent since we don't mix
            # chunked and non-chunked arguments for a fx.Node.
            # But one subtle difference is for aten.full node.
            # We don't need to pass in aten.full as a placeholder since
            # we can just add the chunked aten.full in the subgraph
            # directly.
            # Alternative 2 does not work for this case, so we implement
            # alternative 1.

            user_nodes = node.users
            user_nodes_no_meta = [
                node for node in user_nodes if get_chunking_meta(node) is None
            ]

            if is_chunking_subgraph_input(node):
                # None of the node's arguments are chunked. It's a placeholder
                self.subgraph_input.append(node)
            elif len(user_nodes_no_meta) > 0:
                self.subgraph_output.append(node)
            else:
                self.subgraph_body.append(node)

        assert len(self.subgraph_body) > 0

    def replace_tangent_to_one(self) -> None:
        for idx, node in enumerate(self.overriden_tangent):
            assert is_tangent_node(node)

            fake_tensor = node.meta["val"]
            one = self.parent_graph.call_function(
                aten.full.default, (fake_tensor.shape, 1), _factory_args(fake_tensor)
            )

            name = "tangent_overriden_as_one_{idx}"
            one._rename(name)
            one.meta = copy.copy(node.meta)
            node.replace_all_uses_with(one)
            self.overriden_tangent[node] = one

    def chunk_subgraph_input(self) -> None:
        """
        Chunk subgraph inputs if necessary. Note that not every subgraph
        input needs to be chunked.
        E.g. the weight for matmul does not need to be chunked.
        """
        for node_idx, subgraph_input in enumerate(self.subgraph_input):
            meta = get_chunking_meta(subgraph_input)
            assert meta is not None

            # not chunked
            if meta.chunk_dim is None:
                continue

            chunk_node = self.parent_graph.call_function(
                aten.chunk.default, (subgraph_input, self.num_chunk, meta.chunk_dim)
            )
            chunks = [
                self.parent_graph.call_function(operator.getitem, (chunk_node, i))
                for i in range(self.num_chunk)
            ]

            self.chunked_subgraph_input.append(chunks)

            # let's do a small meta propagation to get the size of each chunk
            if self.chunk_sizes is None:
                input_tensor = subgraph_input.meta["val"]
                tensors = torch.chunk(input_tensor, self.num_chunk)
                self.chunk_sizes = [t.size(0) for t in tensors]

    def create_accumulators(self) -> None:
        for node in self.accumulators:
            # use fp32 for accumulators
            fake_tensor = node.meta["val"]
            if fake_tensor.numel() == 1:
                # TODO(shunting) revisit
                # This tensor may be fused with mm and becomes a addmm
                # if we upcast here, addmm may fail due to incompatible
                # dtypes for input arguments.
                override_dtype = torch.float32
            else:
                override_dtype = fake_tensor.dtype

            kwargs = _factory_args(fake_tensor)
            kwargs["dtype"] = override_dtype
            accum = self.parent_graph.call_function(
                aten.full.default, (fake_tensor.shape, 0), kwargs
            )
            # reuse the meta['val']
            accum.meta = {"val": fake_tensor.to(override_dtype)}
            self.accumulators[node] = accum

    def build_subgraph(self, chunk_size: int) -> GraphModule:
        """
        Build a subgraph for the given chunk size.
        The last chunk can be smaller and a new subgraph will be created
        to avoid involving dynamic shapes.
        """
        new_graph = Graph()
        env: dict[Node, Node] = {}

        def _create_placeholder_node(input_node: Node) -> Node:
            new_node = new_graph.placeholder(input_node.name)
            fake_tensor = input_node.meta["val"]
            chunking_meta = get_chunking_meta(input_node)
            if chunking_meta is not None and chunking_meta.chunk_dim is not None:
                # the node is chunked and we need update the
                # fake tensor
                # TODO any better way to do this?
                new_tensor = aten.slice.Tensor(
                    fake_tensor, chunking_meta.chunk_dim, 0, chunk_size
                )
                fake_tensor = new_tensor
            new_node.meta = {"val": fake_tensor}
            return new_node

        for node_idx, input_node in enumerate(self.subgraph_input):
            env[input_node] = _create_placeholder_node(input_node)

        for overriden_tangent_node in self.overriden_tangent.values():
            assert overriden_tangent_node is not None
            env[overriden_tangent_node] = _create_placeholder_node(
                overriden_tangent_node
            )

        for accum in self.accumulators.values():
            assert accum is not None
            env[accum] = _create_placeholder_node(accum)

        for original_node in self.subgraph_body + self.subgraph_output:
            assert original_node.op != "placeholder"

            # Chunk aten.full
            if (
                original_node.target == aten.full.default
                and (meta := get_chunking_meta(original_node)) is not None
                and meta.chunk_dim is not None
            ):
                shape = list(original_node.args[0])  # type: ignore[arg-type]
                shape[meta.chunk_dim] = chunk_size
                env[original_node] = new_graph.call_function(
                    aten.full.default,
                    (shape, original_node.args[1]),
                    original_node.kwargs,
                )
                continue
            # Chunk aten.expand a scalar
            if (
                original_node.target == aten.expand.default
                and isinstance(original_node.args[0], torch.fx.Node)
                and original_node.args[0].meta["val"].numel() == 1
                and (meta := get_chunking_meta(original_node)) is not None
                and meta.chunk_dim is not None
            ):
                shape = list(original_node.args[1])  # type: ignore[arg-type]
                shape[meta.chunk_dim] = chunk_size
                env[original_node] = new_graph.call_function(
                    aten.expand.default,
                    (env.get(original_node.args[0], original_node.args[0]), shape),  # type: ignore[arg-type]
                    original_node.kwargs,
                )
                continue

            # create the node with chunked inputs
            env[original_node] = new_graph.node_copy(original_node, lambda x: env[x])

        # Do the accumulation inside this subgraph
        for node, accum in self.accumulators.items():
            lhs = env[node]
            assert accum is not None
            rhs = env[accum]

            # add `addend` and `accum`
            add_out = new_graph.call_function(aten.add.Tensor, (lhs, rhs))

            # override the chunk value
            env[node] = add_out

        out_values = []
        for node in self.subgraph_output:
            out_values.append(env[node])

        new_graph.output(tuple(out_values))
        new_graph.eliminate_dead_code()
        new_graph.lint()

        sub_gm = torch.fx._lazy_graph_module._make_graph_module(self.gm, new_graph)
        fake_tensor_prop(sub_gm)
        return sub_gm

    def build_subgraphs(self) -> None:
        assert self.chunk_sizes is not None
        for chunk_size in self.chunk_sizes:
            if chunk_size in self.chunk_size_to_gm_attr:
                continue

            sub_gm = self.build_subgraph(chunk_size)
            gm_attr = f"chunking_subgraph_{len(self.chunk_size_to_gm_attr)}"
            self.chunk_size_to_gm_attr[chunk_size] = gm_attr
            setattr(self.gm, gm_attr, sub_gm)

            # Mark this sub graph module so we don't recursively chunking
            # it.
            sub_gm.meta["produced_by_chunker"] = True

    def call_subgraph_for_each_chunk(self) -> None:
        for chunk_id in range(self.num_chunk):
            assert self.chunk_sizes is not None
            chunk_size = self.chunk_sizes[chunk_id]
            subgraph_id = self.chunk_size_to_gm_attr[chunk_size]
            sub_gm = self.parent_graph.get_attr(subgraph_id)

            args = []
            chunks_iter = iter(self.chunked_subgraph_input)
            for node in self.subgraph_input:
                chunking_meta = get_chunking_meta(node)
                assert chunking_meta is not None
                if chunking_meta.chunk_dim is not None:
                    args.append(next(chunks_iter)[chunk_id])
                else:
                    # not chunked
                    args.append(node)

            args += list(self.overriden_tangent.values())  # type: ignore[arg-type]

            for accum in self.accumulators.values():
                assert accum is not None
                args.append(accum)

            output_node = self.parent_graph.call_function(
                torch.ops.higher_order.invoke_subgraph, (sub_gm, subgraph_id, *args), {}
            )

            output_node_dict = {}
            for i, orig_node in enumerate(self.subgraph_output):
                output_node_dict[orig_node] = self.parent_graph.call_function(
                    operator.getitem, (output_node, i)
                )

            for orig_node, node_list in self.chunks_for_recovering.items():
                chunk = output_node_dict[orig_node]
                node_list.append(chunk)

            for orig_node in self.accumulators:
                self.accumulators[orig_node] = output_node_dict[orig_node]

    def recover_to_unchunked_nodes(self) -> None:
        """
        Recover the node from chunks and do the replacement.
        """
        for node in self.subgraph_output:
            meta = get_chunking_meta(node)
            assert meta is not None

            recovered: torch.fx.Node = node

            if meta.chunk_dim is not None:
                chunks = self.chunks_for_recovering[node]
                recovered = self.parent_graph.call_function(
                    aten.cat.default, (chunks, meta.chunk_dim)
                )
            elif meta.need_sum:
                recovered = self.accumulators[node]  # type: ignore[assignment]

            # do scaling last
            if meta.scale_by is not None:
                recovered = self.parent_graph.call_function(
                    aten.mul.Tensor, (recovered, meta.scale_by)
                )

            # convert back to the original dtype
            if meta.need_sum:
                original_dtype = node.meta["val"].dtype
                # TODO(shunting): do we always uses a fp32 accumulator?
                if original_dtype != torch.float32:
                    recovered = self.parent_graph.call_function(
                        prims.convert_element_type.default, (recovered, original_dtype)
                    )

            assert recovered is not node
            node.replace_all_uses_with(recovered)

    def erase_original_nodes(self) -> None:
        # Traverse reversely to erase user first
        for node in reversed(tuple(self.subgraph_body + self.subgraph_output)):
            if node.op == "placeholder":
                continue

            self.parent_graph.erase_node(node)

    def apply(self) -> GraphModule:
        with self.parent_graph.inserting_before(self.subgraph_body[0]):
            self.replace_tangent_to_one()
            self.chunk_subgraph_input()
            self.create_accumulators()
            self.build_subgraphs()
            self.call_subgraph_for_each_chunk()
            self.recover_to_unchunked_nodes()
            self.erase_original_nodes()

        newgm = torch.fx._lazy_graph_module._make_graph_module(
            self.gm,
            self.parent_graph,
        )
        fake_tensor_prop(newgm)
        if log.isEnabledFor(logging.DEBUG):
            print("Graph module after chunking:")
            newgm.print_readable()
        return newgm
