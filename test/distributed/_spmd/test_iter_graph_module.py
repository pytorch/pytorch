# Owner(s): ["oncall: distributed"]

import copy
import itertools
import operator
from dataclasses import dataclass, field
from functools import partial
from typing import cast, Dict, List, Optional, Tuple

import torch
import torch.fx as fx
import torch.nn as nn
from torch._functorch.aot_autograd import aot_module, make_boxed_func
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.distributed._spmd.distribute import (
    _SPMD,
    DistributedGraph,
    Schema,
    TrainingPhase,
)
from torch.distributed._spmd.graph_utils import (
    CommType,
    get_comm_block_nodes,
    get_output_node,
    OP,
)
from torch.distributed._spmd.iter_graph_module import IterGraphModule
from torch.distributed._tensor import DeviceMesh, Replicate
from torch.distributed._tensor.dispatch import _CURRENT_DECOMPOSITION_TABLE
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)
from torch.utils._pytree import tree_flatten


class BoringModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ln = nn.Sequential(
            nn.Linear(20, 20),
            nn.Softmax(),
        )

    def forward(self, input):
        return self.ln(input)


class NestedBoringModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = torch.nn.Linear(20, 20)
        self.ln2 = torch.nn.Linear(20, 20)
        self.inner = BoringModel()

    def forward(self, input):
        return self.inner(self.ln2(self.ln1(input)))


@dataclass
class FusionElement:
    """
    This class tracks the nodes for a DTensor expanded communication collective
    in the graph.
    """

    size: int = 0
    shape: Optional[torch.Size] = None
    node_list: List[fx.Node] = field(default_factory=lambda: [])
    wait_node: Optional[fx.Node] = None
    comm_node: Optional[fx.Node] = None
    grad_tensor_node: Optional[fx.Node] = None


@dataclass
class GraphInfo:
    """Provides a home for global aspects of this graph.
    Currently tracks first and last node, len of the graph and
    the location and size of the global buffer node
    """

    # list of all FusionElements in the graph
    fe_list: List[FusionElement] = field(default_factory=lambda: [])
    # offset to comm node within a FusionElement sequence
    fe_offset_to_comm_node: Optional[int] = None
    # last node in graph (tail / output)
    output: Optional[fx.Node] = None
    # Map from the wait_node to
    wait_node_idx: Dict[fx.Node, int] = field(default_factory=lambda: {})
    # The gradient to index in the graph.nodes(). This index will change after
    # any transformation but we need this to get the order of the gradient.
    actual_grad_index_mapping: Dict[fx.Node, int] = field(default_factory=lambda: {})

    def update_info(self, gm: fx.GraphModule) -> "GraphInfo":
        """Get the len, input and output nodes"""
        nodelist = gm.graph.nodes
        for i, node in enumerate(nodelist):
            if node.op == OP.OUTPUT:
                for i, arg in enumerate(node.args[0]):
                    if isinstance(arg, fx.Node) and arg.name.startswith("wait_comm"):
                        self.wait_node_idx[arg] = i
        self.output = get_output_node(gm)
        return self


def _scan_graph_for_fusion_elements(
    gi: GraphInfo,
    gm: fx.GraphModule,
    comm_type: CommType = CommType.ALLREDUCE,
) -> List[FusionElement]:
    """Scan entire graph for matching sections of CommTensor style expansions
    returns list of FusionElements that match CommType"""

    element_list = []
    for node in gm.graph.nodes:
        if node.name.startswith("wait_comm"):
            comm_idx, comm_block_nodes = get_comm_block_nodes(node, comm_type)
            comm_node = comm_block_nodes[comm_idx]
            grad_node = cast(Tuple[fx.Node, ...], comm_node.args[0])[0]
            fe = FusionElement(
                node_list=comm_block_nodes[:],
                wait_node=node,
                comm_node=comm_node,
                grad_tensor_node=grad_node,
            )
            element_list.append(fe)
            # ensure we have global index to comm_node
            if not gi.fe_offset_to_comm_node:
                len_comm_section = len(fe.node_list)
                gi.fe_offset_to_comm_node = len_comm_section - comm_idx - 1
    return element_list


def _fuse_with_cat(
    gi: GraphInfo, gm: fx.GraphModule, copy_list: List[FusionElement]
) -> fx.Node:
    # Find the actual last gradient.
    all_grad_tensor_nodes = []
    for fe in copy_list:
        assert fe.grad_tensor_node is not None
        assert fe.grad_tensor_node.name.startswith("clone")
        all_grad_tensor_nodes.append(fe.grad_tensor_node)
    grad_indices_mapping = [
        gi.actual_grad_index_mapping[cast(Tuple[fx.Node], grad_tensor_node.args)[0]]
        for grad_tensor_node in all_grad_tensor_nodes
    ]
    last_grad_fe_index = grad_indices_mapping.index(max(grad_indices_mapping))
    assert copy_list[last_grad_fe_index].grad_tensor_node is not None
    last_grad_tensor_node = cast(
        fx.Node,
        cast(fx.Node, copy_list[last_grad_fe_index].grad_tensor_node).args[0],
    )

    with gm.graph.inserting_after(last_grad_tensor_node):
        cat_inputs = [
            gm.graph.call_function(
                torch.flatten,
                (cast(fx.Node, cast(fx.Node, fe.grad_tensor_node).args[0]),),
            )
            for fe in copy_list
        ]

    with gm.graph.inserting_after(cat_inputs[0]):
        cat_node = gm.graph.call_function(torch.cat, (cat_inputs,))

    assert copy_list[-1].comm_node is not None
    fused_comm_node = copy_list[-1].comm_node
    assert fused_comm_node is not None, "Pyre is not as smart as Mypy."
    gm.graph.node_update_arg(fused_comm_node, 0, [cat_node])
    # fused_comm_node.update_arg(0, [cat_node])

    # Move the fused_comm_node and its args to right after the source node
    nodes_to_move = [
        fused_comm_node,
        fused_comm_node.args[1],
        fused_comm_node.args[2],
        cat_node,
    ] + cat_inputs
    for node in nodes_to_move:
        gm.graph.node_append(last_grad_tensor_node, node)
        # last_grad_tensor_node.append(node)

    return fused_comm_node


def _scatter_results(
    gi: GraphInfo, gm: fx.GraphModule, scatter_list: List[FusionElement]
) -> List[fx.Node]:
    scatter_sizes = [fe.size for fe in scatter_list]
    assert scatter_list[-1].wait_node is not None
    wait_node = scatter_list[-1].wait_node
    with gm.graph.inserting_after(wait_node):
        scatter_node = gm.graph.call_function(
            torch.split,
            (wait_node, scatter_sizes),
        )

    grad_nodes = []
    with gm.graph.inserting_after(scatter_node):
        for idx, fe in enumerate(scatter_list):
            grad_node = gm.graph.call_function(operator.getitem, (scatter_node, idx))
            with gm.graph.inserting_after(grad_node):
                grad_nodes.append(
                    gm.graph.call_function(torch.reshape, (grad_node, fe.shape))
                )

    return grad_nodes


def _update_output_args(
    gi: GraphInfo,
    gm: fx.GraphModule,
    fe_list: List[FusionElement],
    output_args: List[fx.Node],
    grad_nodes: List[fx.Node],
) -> None:
    for fe, grad_node in zip(fe_list, grad_nodes):
        assert fe.wait_node is not None
        output_args[gi.wait_node_idx[fe.wait_node]] = grad_node


def run_fuse_communication_cat(gm: IterGraphModule, fusion_length: int) -> None:
    """
    Run fuse communication with concat.
    This implementation use concat to concat the bucketed gradients.
    """
    # First recompile to make sure we have coherent graph
    gm.recompile()

    graph_info = GraphInfo().update_info(gm)

    fe_list = _scan_graph_for_fusion_elements(
        graph_info, gm, comm_type=CommType.ALLREDUCE
    )
    graph_info.fe_list = fe_list
    assert len(graph_info.wait_node_idx) == len(fe_list), (
        "The expected wait_nodes in graph_info is different from fe_list "
        f"{len(graph_info.wait_node_idx)} {len(fe_list)}."
    )
    assert graph_info.output is not None
    new_output_args = list(cast(Tuple[fx.Node], graph_info.output.args[0]))

    # Need this mapping because the gradient may not have the same order
    # as clone.
    actual_gradients = {
        cast(Tuple[fx.Node], cast(fx.Node, fe.grad_tensor_node).args)[0]
        for fe in fe_list
    }
    for idx, node in enumerate(gm.graph.nodes):
        if node in actual_gradients:
            graph_info.actual_grad_index_mapping[node] = idx

    # Fuse every ``fusion_length`` FusionElement.
    for start in range(0, len(graph_info.fe_list), fusion_length):
        fe_list = graph_info.fe_list[start : (start + fusion_length)]
        fused_comm_node = _fuse_with_cat(graph_info, gm, fe_list)
        grad_nodes = _scatter_results(graph_info, gm, fe_list)
        _update_output_args(
            graph_info,
            gm,
            fe_list,
            new_output_args,
            grad_nodes,
        )

    # update output with the updated args
    gm.graph.erase_node(graph_info.output)
    gm.graph.output(new_output_args)
    gm.graph.lint()
    gm.graph.eliminate_dead_code()
    gm.recompile()


def distribute(
    dist_graph: DistributedGraph,
    param_schema: Schema,
    num_iters: int,
    *args: Tuple[object],
    **kwargs: Dict[str, object],
) -> nn.Module:

    spmd = _SPMD(dist_graph, param_schema, tuple())

    flat_args, _ = tree_flatten(args)
    flat_kwargs, _ = tree_flatten(kwargs)
    input_set: Set[object] = set(flat_args + flat_kwargs)

    fake_mode: FakeTensorMode = FakeTensorMode()

    # will update this to the original forward inputs
    original_inputs: List[Optional[Sequence[object]]] = [None]

    def input_to_fake(input: object) -> object:
        if not isinstance(input, torch.Tensor):
            return input
        y = fake_mode.from_tensor(input)
        if input in input_set:
            y = y.repeat(param_schema.mesh.size(0), *((1,) * (y.ndim - 1)))
        return y

    def gather_inputs_for_compilation(
        inps: Tuple[object, ...],
    ) -> Tuple[object, ...]:
        original_inputs[0] = inps
        return tuple(input_to_fake(x) for x in inps)

    def my_compile(gm: fx.GraphModule, inps: List[torch.Tensor]):
        spmd._compile(TrainingPhase.BACKWARD, gm, inps)
        igm = IterGraphModule(dist_graph.bwd_graph_modules[0])
        igm.setup(num_iters)
        run_fuse_communication_cat(igm, 2)
        for main_node, setup_node, cleanup_node in itertools.zip_longest(
            igm.main_gm.graph.nodes,
            igm.setup_gm.graph.nodes,
            igm.cleanup_gm.graph.nodes,
        ):
            assert setup_node is not None
            assert cleanup_node is not None
            assert main_node is not None

        return make_boxed_func(IterGraphModule(dist_graph.bwd_graph_modules[0]))

    compiled_m = aot_module(
        cast(nn.Module, dist_graph.orig_module),
        # partial(spmd._compile, TrainingPhase.FORWARD),
        partial(spmd._compile_wrapper, TrainingPhase.FORWARD, original_inputs),
        partial(my_compile),
        pre_compile_fn=gather_inputs_for_compilation,
        decompositions=_CURRENT_DECOMPOSITION_TABLE,
    )

    return compiled_m


class IterGraphModuleMultiGPUTest(DTensorTestBase):
    @property
    def world_size(self):
        return 2

    @with_comms
    def test_with_comm_fusion_cat(self) -> None:
        num_iters = 5
        model = NestedBoringModel().to("cuda")
        compile_m = distribute(
            DistributedGraph(model),
            Schema(
                mesh=DeviceMesh(self.device_type, torch.arange(self.world_size)),
                placements=[Replicate()],
            ),
            num_iters,
        )

        for _ in range(num_iters):
            batch = torch.randn(128, 20, device="cuda")
            output = compile_m(batch)
            output.sum().backward()


class IterGraphModuleTest(DTensorTestBase):
    @property
    def world_size(self):
        return 1

    @skip_if_lt_x_gpu(1)
    def test_basic_movement(self) -> None:
        class FakeOptimization:
            def __init__(self) -> None:
                self.all_reduce_counter = 0
                self.wait_counter = 0

            def fake_all_reduce(self, gradients: List[torch.Tensor]) -> torch.Tensor:
                self.all_reduce_counter += 1
                return torch.concat(gradients)

            def fake_wait(self, wait_tensor: torch.Tensor) -> torch.Tensor:
                self.wait_counter += 1
                return torch.clone(wait_tensor)

            def fake_comm_schedule(self, gm: IterGraphModule, move: bool):
                for node in gm.graph.nodes:
                    if node.name == "addmm_2":
                        break
                with gm.graph.inserting_after(node):
                    all_reduce_node = gm.graph.call_function(
                        self.fake_all_reduce, ([node],)
                    )
                with gm.graph.inserting_after(all_reduce_node):
                    wait_node = gm.graph.call_function(
                        self.fake_wait, (all_reduce_node,)
                    )
                for target_node in gm.graph.nodes:
                    if target_node.name == "addmm_1":
                        break
                if move:
                    gm.graph.move_to_next_iter_before([wait_node], target_node)
                # Not calling eliminate_dead_code ensures that nodes won't be
                # removed from the graph.
                gm.graph.lint()
                gm.recompile()

        def _compile_bwd(
            gm: fx.GraphModule, inps: List[torch.Tensor]
        ) -> fx.GraphModule:
            return make_boxed_func(gm)

        def _compile_fwd(
            optimization: FakeOptimization,
            num_iters: int,
            move: bool,
            gm: fx.GraphModule,
            inps: List[torch.Tensor],
        ) -> fx.GraphModule:
            igm = IterGraphModule(gm)
            igm.setup(num_iters)
            optimization.fake_comm_schedule(igm, move)
            return make_boxed_func(igm)

        num_iters = 5
        model = NestedBoringModel().to("cuda")
        model_wo_wrapped = copy.deepcopy(model)
        optim_wo_moved = FakeOptimization()
        model_wo_moved = aot_module(
            copy.deepcopy(model),
            partial(_compile_fwd, optim_wo_moved, num_iters, False),
            _compile_bwd,
        )
        optim_wi_moved = FakeOptimization()
        model_wi_moved = aot_module(
            copy.deepcopy(model),
            partial(_compile_fwd, optim_wi_moved, num_iters, True),
            _compile_bwd,
        )
        all_models = [model_wo_wrapped, model_wo_moved, model_wi_moved]

        for curr_iter in range(num_iters):
            input_ = torch.randn(128, 20, device="cuda")
            outputs = [model(input_) for model in all_models]

            # All the model outputs must be the same even if IterGraphModule is
            # applied and optimized.
            for output in outputs:
                self.assertEqual(output, outputs[0])

            if curr_iter == 0:
                self.assertEqual(optim_wo_moved.all_reduce_counter, 1)
                self.assertEqual(optim_wi_moved.all_reduce_counter, 1)
                self.assertEqual(optim_wo_moved.wait_counter, 1)
                self.assertEqual(optim_wi_moved.wait_counter, 0)
            elif curr_iter == num_iters - 1:
                self.assertEqual(optim_wo_moved.all_reduce_counter, num_iters)
                self.assertEqual(optim_wi_moved.all_reduce_counter, num_iters)
                self.assertEqual(optim_wo_moved.wait_counter, num_iters)
                self.assertEqual(optim_wi_moved.wait_counter, num_iters)
            else:
                self.assertEqual(optim_wo_moved.all_reduce_counter, curr_iter + 1)
                self.assertEqual(optim_wi_moved.all_reduce_counter, curr_iter + 1)
                self.assertEqual(optim_wo_moved.wait_counter, curr_iter + 1)
                self.assertEqual(optim_wi_moved.wait_counter, curr_iter)


if __name__ == "__main__":
    run_tests()
