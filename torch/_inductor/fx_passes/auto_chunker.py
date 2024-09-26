import itertools
from typing import Sequence, Set

import torch
from torch._inductor import config
from torch._inductor.utils import cache_on_self
from torch.utils import _pytree


aten = torch.ops.aten


def _is_permuted(use_node, source_node):
    """
    Return true if use_node is source_node or a permutation of the source_node
    """

    if use_node is source_node:
        return True

    if use_node.target == aten.permute.default and use_node.args[0] is source_node:
        return True
    return False


def _get_fake_tensor_from_node(nd: torch.fx.Node) -> torch.Tensor:
    if (
        not hasattr(nd, "meta")
        or ("val" not in nd.meta)
        or not isinstance(nd.meta["val"], torch.Tensor)
    ):
        return None
    return nd.meta["val"]


def _compute_tensor_size(*args, **kwargs):
    flat_args, _ = _pytree.tree_flatten((args, kwargs))
    tot = 0
    for arg in flat_args:
        if (fake_tensor := _get_fake_tensor_from_node(arg)) is None:
            continue
        tot += fake_tensor.numel() * fake_tensor.dtype.itemsize
    return tot


def _collect_source_nodes(graph: torch.fx.Graph) -> Sequence[torch.fx.Node]:
    source_nodes = []
    for nd in graph.nodes:
        # Only chunk a small set of source nodes like matmul for now
        if nd.op != "call_function" or nd.target != aten.mm.default:
            continue

        input_size = _compute_tensor_size(nd.args, nd.kwargs)
        output_size = _compute_tensor_size(nd)

        # We chunk the first input on the batch dimension.
        # This is the rule targeting matmul but may work good enough in general.
        if (
            _compute_tensor_size(nd.args[:1]) > config.AutoChunker.input_threshold
            and output_size / input_size > config.AutoChunker.amplify_threshold
        ):
            source_nodes.append((nd.args[0], output_size / input_size))

    source_nodes = sorted(source_nodes, key=lambda x: x[1], reverse=True)
    return tuple(map(lambda x: x[0], source_nodes))


def _collect_reachable_nodes(
    source_node: torch.fx.Node, stop_nodes: Set[torch.fx.Node]
) -> Sequence[torch.fx.Node]:
    """
    Collect the nodes reachable from the `source_node` via the use chain.
    The output is topologically sorted.
    """
    reachable_nodes = []
    seen = set()

    def dfs(nd):
        if nd in seen:
            return

        seen.add(nd)

        # no need to add the output node since the output node should not
        # see the effect of chunking.
        if nd.op == "output":
            return

        if nd not in stop_nodes:
            for nd2 in nd.users:
                dfs(nd2)

        reachable_nodes.append(nd)

    dfs(source_node)

    return list(reversed(reachable_nodes))


class AutoChunker:
    def __init__(self, gm: torch.fx.GraphModule):
        self.gm = gm
        self.source_node = None

    @property
    @cache_on_self
    def source_matmul_node(self):
        assert len(self.source_node.users) > 0
        matmul_node = list(self.source_node.users.keys())[0]
        assert matmul_node.target == aten.mm.default
        return matmul_node

    @property
    @cache_on_self
    def matmuls_in_bwd(self):
        matmuls = []
        for nd in self.node_list[self.first_bwd_node_index :]:
            if nd.target == aten.mm.default:
                matmuls.append(nd)
        return matmuls

    @property
    @cache_on_self
    def gradient_matmul_input(self):
        """
        First matmul in the backward pass that uses the second matmul input
        (maybe permuted) as argument.
        """
        assert self.source_matmul_node.target == aten.mm.default

        for mm_nd in self.matmuls_in_bwd:
            if _is_permuted(mm_nd.args[0], self.matmul_weight) or _is_permuted(
                mm_nd.args[1], self.matmul_weight
            ):
                return mm_nd

        raise RuntimeError("Can not find the node computing graident of matmul input")

    @property
    @cache_on_self
    def gradient_matmul_weight(self):
        """
        First matmul in the backward pass that uses the first matmul input
        (maybe permuted) as argument.
        """
        assert self.source_matmul_node.target == aten.mm.default

        for mm_nd in self.matmuls_in_bwd:
            if _is_permuted(mm_nd.args[0], self.matmul_input) or _is_permuted(
                mm_nd.args[1].self.matmul_input
            ):
                return mm_nd
        raise RuntimeError("Can not find the node computing graident of matmul weight")

    @property
    def matmul_input(self):
        assert self.source_matmul_node.target == aten.mm.default
        assert self.source_node is self.source_matmul_node.args[0]
        return self.source_matmul_node.args[0]

    @property
    @cache_on_self
    def matmul_weight(self):
        assert self.source_matmul_node.target == aten.mm.default
        return self.source_matmul_node.args[1]

    @property
    def last_fwd_node_index(self):
        return self.first_bwd_node_index - 1

    @property
    @cache_on_self
    def first_bwd_node_index(self):
        """
        Assume the first bwd node is the first node that accesss `gradient_output`
        """
        for idx, nd in enumerate(self.gm.graph.nodes):
            if any(
                "tangent" in x.name
                for x in _pytree.tree_flatten((nd.args, nd.kwargs))[0]
                if isinstance(x, torch.fx.Node)
            ):
                return idx
        assert False, "Can not reach here"

    @property
    def last_fwd_node(self):
        return self.node_list[self.last_fwd_node_index]

    @property
    def first_bwd_node(self):
        return self.node_list[self.first_bwd_node_index]

    @property
    @cache_on_self
    def node_list(self):
        return list(self.gm.graph.nodes)

    @property
    @cache_on_self
    def gradient_output(self):
        for nd in self.gm.graph.nodes:
            if nd.op == "placeholder" and "tangent" in nd.name:
                return nd
        assert False, "Can not reach here."

    def has_single_scalar_gradient(self):
        gradient_inputs = []
        for nd in self.gm.graph.nodes:
            if nd.op == "placeholder" and "tangent" in nd.name:
                gradient_inputs.append(nd)

        if len(gradient_inputs) != 1:
            return False

        gradient_input = gradient_inputs[0]

        if (fake_tensor := _get_fake_tensor_from_node(gradient_input)) is None:
            return False

        return fake_tensor.numel() == 1

    def chunk_batch_dimension(self):
        """
        Chunk input tensor for operations that amplifies the tensor size significantly.
        Only chunk across the batch dimension of the tensor.

        Currently only apply chunk once. But it can be enhanced to apply chunking
        multiple times.

        self.gm is transformed inplace.
        """

        if not self.has_single_scalar_gradient():
            print("The graph does not have a single scalar gradient")
            return

        gm = self.gm
        gm.print_readable()
        graph = gm.graph
        source_nodes = _collect_source_nodes(graph)
        print(f"{source_nodes=}")
        if len(source_nodes) == 0:
            return

        self.source_node = source_nodes[0]  # pick the first one

        # We should not propagate pass gradient of matmul input/weight since
        # we will 'un-chunk' them. The downstream nodes don't see the effect
        # of chunking.
        reachable_nodes = _collect_reachable_nodes(
            self.source_node, {self.gradient_matmul_input, self.gradient_matmul_weight}
        )
        print(f"{reachable_nodes=}")

        print(f"{self.gradient_output=}")
        print(f"{self.last_fwd_node_index=}")

        print(f"{self.matmul_input=}")
        print(f"{self.matmul_weight=}")
        print(f"{self.gradient_matmul_input=}")
        print(f"{self.gradient_matmul_weight=}")

        extra_reachable_nodes = []
        if self.last_fwd_node in reachable_nodes:
            # the last fwd node is affected by chunking.
            # Assume the first bwd node is for computing the gradient of the
            # last fwd node.
            # Then the first bwd node should also be affected by chunking.

            extra_reachable_nodes = _collect_reachable_nodes(
                self.first_bwd_node,
                {self.gradient_matmul_input, self.gradient_matmul_weight},
            )

        def _merge_reachable_nodes(lhs, rhs):
            """
            We want to dedup the nodes and order then as the orignal graph
            """
            node_to_idx = {nd: idx for idx, nd in enumerate(self.node_list)}
            out = set()
            for nd in itertools.chain(lhs, rhs):
                assert isinstance(nd, torch.fx.Node)
                if nd not in out:
                    out.add(nd)

            return sorted(out, key=lambda nd: node_to_idx[nd])

        nodes_to_chunk = _merge_reachable_nodes(reachable_nodes, extra_reachable_nodes)

        print(f"{nodes_to_chunk=}")
