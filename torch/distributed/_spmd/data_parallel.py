import operator
import os
from dataclasses import dataclass
from enum import Enum
from functools import partial

from typing import Any, Callable, cast, Dict, List, Optional, Tuple

import torch
import torch._inductor.config
import torch.distributed as dist
import torch.fx as fx
import torch.multiprocessing as mp
import torch.nn as nn

import torch.utils._pytree as pytree
from torch import SymInt
from torch._functorch.aot_autograd import make_boxed_func
from torch._functorch.compilers import aot_function
from torch.distributed._tensor import (
    DeviceMesh,
    distribute_module,
    distribute_tensor,
    DTensor,
    Replicate,
    Shard,
)

from torch.distributed._tensor._utils import compute_local_shape
from torch.distributed._tensor.op_schema import PlacementStrategy, StrategyList
from torch.distributed._tensor.placement_types import _Partial, DTensorSpec
from torch.distributed._tensor.redistribute import _redistribute_with_local_tensor
from torch.fx import GraphModule
from torch.fx.experimental.proxy_tensor import make_fx
from torch.nn.utils import stateless

# torch._inductor.config.debug = True

aten = torch.ops.aten

# from torch._subclasses import FakeTensorMode

# NOTE:
# 1. Ideally we should use FakeTensorMode, but it does not work well in practice
#    i.e. there's no deferred_init yet, so we can't materialize the fake tensor
#    easily, we choose to use CPU for SPMD partition for now.
# 2. We should use Dynamo as frontend but it currently graph break at backward call
#    so we use aot_function for now.

# graphs = []
# def custom_compiler(fx_g, _):
#     graphs.append(fx_g)
#     return fx_g

# def foo(a, b):
#     return (a.cos() + b.cos()).sum()

# aot_function(foo, custom_compiler)(torch.randn(5, 5, device='cuda', requires_grad=True), torch.randn(5, 5, device='cuda')).backward()

# import torch.fx as fx

# def join_fw_bw(fw_graph, bw_graph):
#     new_graph = fx.Graph()
#     env = {}
#     fw_outputs = []
#     for node in fw_graph.graph.nodes:
#         if node.op == 'output':
#             fw_outputs = [env[arg] for arg in node.args[0]]
#         else:
#             new_node = new_graph.node_copy(node, lambda n: env[n])
#             env[node] = new_node
#     loss, *activations = fw_outputs
#     activations = iter(activations)
#     bw_env = {}
#     for node in bw_graph.graph.nodes:
#         if node.op == 'placeholder':
#             try:
#                 bw_env[node] = next(activations)
#             except:
#                 bw_env[node] = 1
#         else:
#             new_node = new_graph.node_copy(node, lambda n: bw_env[n])
#             bw_env[node] = new_node
#     return new_graph

# out_graph = join_fw_bw(graphs[0], graphs[1])


# simple model definition
class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.net1 = nn.Linear(50, 32)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(32, 8)

    def forward(self, x):
        return torch.sigmoid(self.net2(self.relu(self.net1(x))))

    def reset_parameters(self, *args, **kwargs):
        self.net1.reset_parameters()
        self.net2.reset_parameters()


# simple train step definition, just an example
def train_step(model, loss_fn, optim, train_batch):
    # if we dont set to none, the gradients will be copied into the buffers at the end
    # that's alright but probably a perf regression
    optim.zero_grad(set_to_none=True)
    inputs, labels = train_batch

    out = model(inputs)
    loss = loss_fn(out, labels)
    loss.backward()
    # optim.step()
    return loss


class DataParallelStyle(Enum):
    DEFAULT = 0
    REPLICATE = 1
    FULLY_SHARD = 2


class NodeType(Enum):
    PARAM = 0
    ACT = 1
    GRAD = 2
    STATE = 3
    OUT = 4  # out is just for convinience to tag graph output


# special batch size variable to track batch dimension, we should use symbolic shapes to
# track the batch dimension, this needs to happen with dynamo,
batch_dim_size = 256


class BatchDimAnalyzer(object):
    def __init__(self, batch_dim_size):
        self.batch_dim_size = batch_dim_size
        self.batch_dim_map = {}

    def set_batch_dim(self, node: fx.Node, batch_dim: int):
        self.batch_dim_map[node] = batch_dim

    def get_batch_dim(self, node: fx.Node):
        if node not in self.batch_dim_map:
            raise RuntimeError(f"batch dim analysis failed on node: {node}!")
        return self.batch_dim_map[node]

    def get_batch_dim_shard_spec(self, node: fx.Node, mesh, input_full_reduction=False) -> Tuple[DTensorSpec, bool]:
        if node in self.batch_dim_map:
            node_batch_dim = self.get_batch_dim(node)
            batch_dim_shard_spec = DTensorSpec(
                mesh=mesh, placements=[Shard(node_batch_dim)]
            )
            return batch_dim_shard_spec, False

        shape = node.meta["val"].shape

        # dumb batch dim analysis
        # for reduction op that reduces over the sharded batch dim
        # we don't generate partial, but rather, we generate shard
        # This is because the intention of data parallel is to never
        # do full reduction across batch dimension, it would still
        # keep the reduction activation sharded.
        reduction_over_batch = False
        reduction_ops = [aten.sum.default, aten.mean.default]
        if node.target in reduction_ops and len(shape) == 0:
            operand = node.args[0]
            if operand in self.batch_dim_map:
                operand_batch_dim = self.get_batch_dim(operand)
                if operand_batch_dim == 0:
                    reduction_over_batch = True
                    self.set_batch_dim(node, operand_batch_dim)

            else:
                raise RuntimeError(f"batch dim analysis failed on node: {node}!")
        elif input_full_reduction:
            # the first consumer node that consumes the full reduction
            operand = node.args[0]
            assert operand in self.batch_dim_map, "input have full reduction but not in dim map!"
            self.set_batch_dim(node, self.get_batch_dim(operand))
        else:
            for i, dim_size in enumerate(shape):
                if dim_size == batch_dim_size:
                    self.set_batch_dim(node, i)

        node_batch_dim = self.get_batch_dim(node)
        batch_dim_shard_spec = DTensorSpec(mesh=mesh, placements=[Shard(node_batch_dim)])
        if reduction_over_batch or input_full_reduction:
            batch_dim_shard_spec.from_local = True

        return batch_dim_shard_spec, reduction_over_batch



def get_batch_dim_shard_spec(
    node: fx.Node, batch_dim_map, mesh, input_full_reduction=False
) -> Tuple[DTensorSpec, bool]:
    if node in batch_dim_map:
        node_batch_dim = batch_dim_map[node]
        batch_dim_shard_spec = DTensorSpec(
            mesh=mesh, placements=[Shard(node_batch_dim)]
        )
        return batch_dim_shard_spec, False

    shape = node.meta["val"].shape

    # dumb batch dim analysis
    # for reduction op that reduces over the sharded batch dim
    # we don't generate partial, but rather, we generate shard
    # This is because the intention of data parallel is to never
    # do full reduction across batch dimension, it would still
    # keep the reduction activation sharded.
    reduction_over_batch = False
    reduction_ops = [aten.sum.default, aten.mean.default]
    if node.target in reduction_ops and len(shape) == 0:
        operand = node.args[0]
        if operand in batch_dim_map:
            operand_batch_dim = batch_dim_map[operand]
            if operand_batch_dim == 0:
                reduction_over_batch = True
                batch_dim_map[node] = operand_batch_dim

        else:
            raise RuntimeError(f"batch dim analysis failed on node: {node}!")
    elif input_full_reduction:
        # the first consumer node that consumes the full reduction
        operand = node.args[0]
        assert operand in batch_dim_map, "input have full reduction but not in dim map!"
        batch_dim_map[node] = batch_dim_map[operand]
    else:
        for i, dim_size in enumerate(shape):
            if dim_size == batch_dim_size:
                batch_dim_map[node] = i

    if node not in batch_dim_map:
        raise RuntimeError(f"batch dim analysis failed on node: {node}!")

    node_batch_dim = batch_dim_map[node]
    batch_dim_shard_spec = DTensorSpec(mesh=mesh, placements=[Shard(node_batch_dim)])
    if reduction_over_batch or input_full_reduction:
        batch_dim_shard_spec.from_local = True

    return batch_dim_shard_spec, reduction_over_batch


@dataclass
class DataParallelStrategy(StrategyList):
    node_type: NodeType
    reduction_over_batch: bool

    def __init__(self, node_type, startegy_list, reduction_over_batch=False):
        super().__init__(startegy_list)
        self.node_type = node_type
        self.reduction_over_batch = reduction_over_batch

    def __str__(self) -> str:
        return f"type: {self.node_type}, {super().__str__()}"


def gen_shard_strategy(
    mesh, shard_dim, input_specs: Optional[List[DTensorSpec]] = None
):
    return PlacementStrategy(
        output_spec=DTensorSpec(mesh=mesh, placements=[Shard(shard_dim)]),
        input_specs=input_specs,
    )


def gen_replicate_strategy(mesh, input_specs: Optional[List[DTensorSpec]] = None):
    return PlacementStrategy(
        output_spec=DTensorSpec(mesh=mesh, placements=[Replicate()]),
        input_specs=input_specs,
    )


def gen_partial_strategy(mesh):
    return PlacementStrategy(
        output_spec=DTensorSpec(mesh=mesh, placements=[_Partial()]),
    )


def build_data_parallel_strategies(
    train_step_graph, num_params, num_states, mesh, batch_dim=0
) -> Dict[fx.Node, StrategyList]:
    activation_idx = num_params + num_states
    non_compute_ops = [
        torch.ops._spmd.tag_grad.default,
        aten.ones_like.default,
        aten.t.default,
        aten.view.default,
        aten.reshape.default,
        aten.detach.default,
    ]

    dp_strategy_map = {}
    batch_dim_map = {}
    placeholder_idx = 0
    num_param_grad = 0

    # print(train_step_graph.graph.nodes)

    # first we backward propagate to mark the param gradients sharding
    # with tag_grad node helps and then delete the tag_grad nodes
    for node in reversed(list(train_step_graph.graph.nodes)):
        # find a param_grad node
        if node.target == torch.ops._spmd.tag_grad.default:
            cur_node = node
            while cur_node.target in non_compute_ops:
                cur_node = cur_node.args[0]
                partial_strategy = gen_partial_strategy(mesh)
                dp_strategy_map[cur_node] = DataParallelStrategy(
                    NodeType.GRAD, [partial_strategy]
                )
            num_param_grad += 1
            # remove the tag_grad node from graph
            node.replace_all_uses_with(node.args[0])
            train_step_graph.graph.erase_node(node)

            if num_param_grad == num_params:
                # early break if we have already processed all param_grads
                break

    train_step_graph.recompile()
    train_step_graph.print_readable()

    # next we forward propagate to mark all the sharding
    for node in train_step_graph.graph.nodes:
        if node.op == "placeholder":
            if placeholder_idx < num_params:
                shard_sig = gen_shard_strategy(mesh, 0)
                replica_sig = gen_replicate_strategy(mesh)
                dp_strategy_map[node] = DataParallelStrategy(
                    NodeType.PARAM, [replica_sig, shard_sig]
                )

            elif placeholder_idx < activation_idx:
                shard_sig = gen_shard_strategy(mesh, 0)
                replica_sig = gen_replicate_strategy(mesh)
                dp_strategy_map[node] = DataParallelStrategy(
                    NodeType.STATE, [replica_sig, shard_sig]
                )
            else:
                # activation should only be sharded
                activation_batch_dim_size = node.meta["val"].shape[batch_dim]
                assert (
                    activation_batch_dim_size == batch_dim_size
                ), "input batch dim must be the dim 0"
                batch_dim_map[node] = batch_dim
                shard_sig = gen_shard_strategy(mesh, batch_dim)
                dp_strategy_map[node] = DataParallelStrategy(NodeType.ACT, [shard_sig])
            placeholder_idx += 1
        elif node.op == "call_function":
            # annotate node types for the computation graph
            # data parallel node propagation logic:
            # param (non-compute) -> out: param
            # grad (non-compute before/after) -> out: grad
            # param + activation (param must be replicate) -> out: activation
            # param + grad (param/grad share the same spec) -> out: param

            if node.target in non_compute_ops:
                input_full_reduction = False
                input_nodes = node.all_input_nodes
                assert (
                    len(input_nodes) == 1
                ), f"non-compute op only support one input now, found node: {node} with length of inputs: {len(node.args)}"
                arg_node_type = dp_strategy_map[input_nodes[0]].node_type
                if dp_strategy_map[input_nodes[0]].reduction_over_batch:
                    input_full_reduction = True

                if arg_node_type == NodeType.PARAM:
                    replica_sig = gen_replicate_strategy(mesh)
                    dp_strategy_map[node] = DataParallelStrategy(
                        NodeType.PARAM, [replica_sig]
                    )
                elif arg_node_type == NodeType.GRAD:
                    partial_sig = gen_partial_strategy(mesh)
                    dp_strategy_map[node] = DataParallelStrategy(
                        NodeType.GRAD, [partial_sig]
                    )
                elif arg_node_type == NodeType.ACT:
                    arg_node_spec, _ = get_batch_dim_shard_spec(
                        input_nodes[0], batch_dim_map, mesh
                    )
                    output_spec, _ = get_batch_dim_shard_spec(
                        node, batch_dim_map, mesh, input_full_reduction
                    )
                    shard_sig = PlacementStrategy(
                        output_spec=output_spec, input_specs=[arg_node_spec]
                    )
                    dp_strategy_map[node] = DataParallelStrategy(
                        NodeType.ACT, [shard_sig]
                    )
                else:
                    raise RuntimeError(
                        f"non compute op not supporting {arg_node_type}! "
                    )

                # finished processing this non-compute node
                continue

            # for computatation node, we need to check all the inputs
            input_args = node.all_input_nodes
            input_specs = []
            if node in dp_strategy_map:
                # found a param_grad node that already have output pre-filled spec
                # fill in the expected input specs for the pre-filled strategy
                node_type = dp_strategy_map[node].node_type
                assert node_type == NodeType.GRAD
                produce_param_grad_strat = dp_strategy_map[node].strategies
                has_activation = False
                for arg in input_args:
                    arg_node_type = dp_strategy_map[arg].node_type
                    if arg_node_type == NodeType.ACT:
                        # activation sharded
                        has_activation = True
                        act_spec, _ = get_batch_dim_shard_spec(arg, batch_dim_map, mesh)
                        input_specs.append(act_spec)

                if has_activation:
                    assert len(produce_param_grad_strat) == 1
                    produce_param_grad_strat[0].input_specs = input_specs
            else:
                has_activation = False
                has_grad = False
                for arg in input_args:
                    if dp_strategy_map[arg].node_type == NodeType.ACT:
                        has_activation = True
                        break
                    elif dp_strategy_map[arg].node_type == NodeType.GRAD:
                        has_grad = True
                        break

                if has_activation:
                    # build up the acceptable input specs for param + activation
                    for arg in input_args:
                        arg_strategies = dp_strategy_map[arg]
                        node_type = arg_strategies.node_type
                        if node_type == NodeType.ACT:
                            # activation must stay sharded
                            act_spec, _ = get_batch_dim_shard_spec(
                                arg, batch_dim_map, mesh
                            )
                            input_specs.append(act_spec)
                        else:
                            # param must be replicated
                            input_specs.append(
                                DTensorSpec(mesh=mesh, placements=[Replicate()])
                            )
                    # produce activation type sharding for output
                    output_spec, reduction_over_batch = get_batch_dim_shard_spec(
                        node, batch_dim_map, mesh
                    )

                    shard_strategy = PlacementStrategy(
                        output_spec=output_spec, input_specs=input_specs
                    )

                    dp_strategy_map[node] = DataParallelStrategy(
                        NodeType.ACT, [shard_strategy], reduction_over_batch
                    )
                elif has_grad:
                    # TODO: optimizer parts should follow the dtensor prop logic
                    # to support more general cases that allows optimizer states
                    # to have different shardings compare to the params
                    replica_strategy = gen_replicate_strategy(mesh, [])
                    shard_strategy = gen_shard_strategy(mesh, batch_dim, [])
                    for arg in input_args:
                        arg_strategies = dp_strategy_map[arg]
                        node_type = arg_strategies.node_type
                        replica_spec = DTensorSpec(mesh=mesh, placements=[Replicate()])
                        shard_spec = DTensorSpec(
                            mesh=mesh, placements=[Shard(batch_dim)]
                        )
                        replica_strategy.input_specs.append(replica_spec)
                        shard_strategy.input_specs.append(shard_spec)

                    dp_strategy_map[node] = DataParallelStrategy(
                        NodeType.PARAM, [replica_strategy, shard_strategy]
                    )

                else:
                    raise RuntimeError(f"Unrecognized node: {node}!")

        elif node.op == "output":
            dp_strategy_map[node] = DataParallelStrategy(NodeType.OUT, [None])
        else:
            raise RuntimeError(f"op code {node.op} not supported")

    for node_key, strat in dp_strategy_map.items():
        if strat.node_type != NodeType.OUT:
            print(node_key, strat)
    # print(node_key, strat)

    # TODO: for optimizer graph strategy building
    # we do: 1. make all named optim states have the same strategies as the corresponding param
    # 2. for node have param_gradient input, we generate two strategies, sharded/replicated base
    # on the parallel mode, this means any node that's not a compute op, we only accept one of the
    # two shardings as the input specs.
    # 3. in mark_data_parallel sharding, we filter out the other strategy base on the parallel mode

    # TODO: if we have full graph directly, how do we tag parameter grad and optimizer states?
    # is optimizer states an activation? or a separate tensor type?
    # NOTE: probably it should be either an paremter type or a separate tensor type, as it must follow
    # the same sharding as the corresponding parameter, but if it's a parameter type, it seems it voilates
    # the replicate assumption..
    # maybe it's a new tensor type, i.e. optim_state
    # for any optim state, we should follow dtensor rule? or we mark sharding by ourselves?

    return dp_strategy_map


def mark_data_parallel_shardings(
    train_step_graph: fx.GraphModule,
    num_parameters: int,
    num_states: int,
    dp_strategy_map: Dict[fx.Node, DataParallelStrategy],
    parallel_mode: DataParallelStyle = DataParallelStyle.FULLY_SHARD,
):
    activation_idx = num_parameters + num_states
    placeholder_idx = 0
    for node in train_step_graph.graph.nodes:
        if node.op == "placeholder":
            node_strategies = dp_strategy_map[node].strategies
            assert len(node_strategies) > 0, "node_strategies should not be empty"
            if placeholder_idx < activation_idx:
                if parallel_mode == DataParallelStyle.REPLICATE:
                    # set to replicate for replicate style
                    node_sharding = node_strategies[0]
                elif parallel_mode == DataParallelStyle.FULLY_SHARD:
                    # set to shard for fully shard style
                    node_sharding = node_strategies[1]
                elif parallel_mode == DataParallelStyle.DEFAULT:
                    # todo: add support for default mode
                    # default mode would generate either replicate or shard
                    raise NotImplementedError("default mode not implemented")
            else:
                # mark activation as sharded on batch dim
                node_sharding = node_strategies[0]

            node.meta["sharding"] = node_sharding

            placeholder_idx += 1
        elif node.op == "call_function" or node.op == "output":
            node_strategies = dp_strategy_map[node].strategies
            assert (
                len(node_strategies) <= 2
            ), "data parallel should have at most 2 strategies"
            if len(node_strategies) == 1:
                node.meta["sharding"] = node_strategies[0]
            elif len(node_strategies) == 2:
                if parallel_mode == DataParallelStyle.REPLICATE:
                    # set to replicate for replicate style
                    node.meta["sharding"] = node_strategies[0]
                elif parallel_mode == DataParallelStyle.FULLY_SHARD:
                    # set to shard for fully shard style
                    node.meta["sharding"] = node_strategies[1]
                else:
                    raise RuntimeError("default mode not supported yet!")
            elif node.op != "output":
                raise RuntimeError(
                    f"node {node} strategy length {len(node_strategies)} is not expected!"
                )
        else:
            raise RuntimeError(f"op code {node.op} not supported")

    # for node in train_step_graph.graph.nodes:
    #     if "sharding" in node.meta and node.meta["sharding"] is not None:
    #         print(
    #             node,
    #             node.meta["sharding"],
    #             node.meta["sharding"].output_spec,
    #             hasattr(node.meta["sharding"].output_spec, "from_local"),
    #         )


def to_local_shard(full_tensor: torch.Tensor, spec: DTensorSpec, from_local=False):
    local_shard = full_tensor
    # print(f">>>>> converting full tensor: {full_tensor} to local shard")
    if from_local:
        return local_shard

    for idx, placement in enumerate(spec.placements):
        if placement.is_shard():
            placement = cast(Shard, placement)
            num_chunks = spec.mesh.size(dim=idx)
            my_coord = spec.mesh.get_coordinate()[idx]
            local_shard = placement._split_tensor(
                local_shard, num_chunks, contiguous=False
            )[0][my_coord]
    # print(f">>>> converted local shard: {local_shard}")
    return local_shard


def partitioner(graph: fx.GraphModule):
    from torch.fx.passes.shape_prop import _extract_tensor_metadata

    view_ops = [aten.view.default, aten.expand.default]
    # partition the graph to distributed
    for node in graph.graph.nodes:
        # print(f">>> handling node: {node}")
        # node_strat = dp_strategy_map[node]
        node_sharding = node.meta["sharding"]

        if node.op == "placeholder":
            # print(f">>>> processing placeholder: {node}")
            out_spec = node_sharding.output_spec
            if not hasattr(out_spec, "from_local"):
                local_val = to_local_shard(node.meta["val"], out_spec)
                local_tensor_meta = _extract_tensor_metadata(local_val)
                # update metadata
                node.meta["val"] = local_val
                node.meta["tensor_meta"] = local_tensor_meta
        elif node.op == "call_function":
            out_spec = node_sharding.output_spec
            # for view related op that needs shape, adjust shape if needed
            full_tensor = node.meta["val"]
            full_shape = full_tensor.shape

            # check if there's misaligned sharding, insert reshard if there is
            expected_input_specs = node_sharding.input_specs
            for idx, input_arg in enumerate(node.all_input_nodes):
                input_arg_spec = input_arg.meta["sharding"].output_spec
                input_arg_tensor = input_arg.meta["val"]
                desired_spec = (
                    out_spec
                    if expected_input_specs is None
                    else expected_input_specs[idx]
                )
                if input_arg_spec != desired_spec:
                    # insert reshard operation
                    def reshard_fn(local_tensor: torch.Tensor) -> torch.Tensor:
                        return _redistribute_with_local_tensor(
                            local_tensor,
                            full_shape,
                            out_spec.mesh,
                            input_arg_spec.placements,
                            desired_spec.placements,
                        )

                    reshard_gm = make_fx(reshard_fn)(input_arg_tensor)
                    reshard_gm_nodes = list(reshard_gm.graph.nodes)
                    input_node = reshard_gm_nodes[0]
                    with graph.graph.inserting_before(node):
                        output_node = graph.graph.graph_copy(
                            reshard_gm.graph,
                            val_map={
                                input_node: input_arg,
                            },
                        )
                    node.replace_input_with(input_arg, output_node)
            if node.target in view_ops:
                local_shape = compute_local_shape(
                    full_shape, out_spec.mesh, out_spec.placements
                )
                node.update_arg(1, local_shape)

            # convert output to local shard
            # note that if the output is already local, we don't need to convert
            if not hasattr(out_spec, "from_local"):
                local_val = to_local_shard(full_tensor, out_spec)
                local_tensor_meta = _extract_tensor_metadata(local_val)
                # update metadata
                node.meta["val"] = local_val
                node.meta["tensor_meta"] = local_tensor_meta

        elif node.op == "output":
            # expanded_graph.output()
            break
        else:
            raise RuntimeError(f"op code {node} not supported")

    graph.graph.lint()
    graph.recompile()
    graph.print_readable()
    return graph


def partition_data_parallel(
    graph: fx.GraphModule,
    params_buffers: Dict[str, torch.Tensor],
    named_states: Dict[str, torch.Tensor],
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    mesh: DeviceMesh,
    parallel_style: DataParallelStyle,
):
    num_params_buffers = len(params_buffers)
    num_states = len(named_states)
    strategy_map = build_data_parallel_strategies(
        graph, num_params_buffers, num_states, mesh=mesh
    )

    mark_data_parallel_shardings(
        graph,
        num_parameters=num_params_buffers,
        num_states=num_states,
        dp_strategy_map=strategy_map,
        parallel_mode=parallel_style,
    )
    # really partition the weights and optim states to
    # DTensor based on the parallel style
    for param_key, param in params_buffers.items():
        if parallel_style == DataParallelStyle.REPLICATE:
            params_buffers[param_key] = distribute_tensor(param, mesh, [Replicate()])
        elif parallel_style == DataParallelStyle.FULLY_SHARD:
            params_buffers[param_key] = distribute_tensor(param, mesh, [Shard(0)])
        else:
            raise RuntimeError(f"parallel style {parallel_style} not supported yet")

    for state_key, state_val in named_states.items():
        if parallel_style == DataParallelStyle.REPLICATE:
            named_states[state_key] = distribute_tensor(state_val, mesh, [Replicate()])
        elif parallel_style == DataParallelStyle.FULLY_SHARD:
            named_states[state_key] = distribute_tensor(state_val, mesh, [Shard(0)])
        else:
            raise RuntimeError(f"parallel style {parallel_style} not supported yet")
    # for node in graph.graph.nodes:
    #     if "sharding" in node.meta:
    #         print(f">>> node: {node}, sharding: {node.meta['sharding']}")
    #     else:
    #         print(f">>> node: {node} has no sharding!!!")

    # single machine to distribute
    partitioned_graph = partitioner(graph)
    # partitioned_graph = graph

    return partitioned_graph


# def compiled_fn(
#     *args, mesh=None, parallel_mode: DataParallelStyle = DataParallelStyle.FULLY_SHARD
# ):
#     model, loss_fn, optim, train_batch = args
#     inputs, labels = train_batch

#     graphs = []

#     def fwd_bwd_compiler(fx_g, _):
#         graphs.append(fx_g)
#         return make_boxed_func(fx_g)

#     named_params = dict(model.named_parameters(remove_duplicate=False))
#     named_buffers = dict(model.named_buffers(remove_duplicate=False))
#     num_params_buffers = len(named_params) + len(named_buffers)

#     def functional_call(named_params, named_buffers, inputs, labels):
#         params_and_buffers = {**named_params, **named_buffers}
#         out = torch.func.functional_call(model, params_and_buffers, inputs)
#         loss = loss_fn(out, labels)
#         return loss

#     compiled_fn = aot_function(
#         functional_call,
#         fw_compiler=fwd_bwd_compiler,
#         num_params_buffers=num_params_buffers,
#         dynamic=False,
#     )
#     compiled_fn(named_params, named_buffers, inputs, labels).backward()

#     # build optim graph
#     from torch._dynamo.backends.common import aot_autograd

#     my_backend = aot_autograd(fw_compiler=fwd_bwd_compiler)
#     train_step_compile_fn = torch.compile(optim.step, backend=my_backend)
#     train_step_compile_fn()
#     # print(graphs)

#     print(">>> forward graph: ")
#     # print(graphs[0].print_readable())
#     print(graphs[0].graph)
#     print(">>> backward graph: ")
#     print(graphs[1].print_readable())
#     print(f">>> optim graph: ")
#     print(graphs[2].print_readable())
#     strategy_map = build_data_parallel_strategies(
#         graphs, activation_idx=num_params_buffers, mesh=mesh
#     )

#     mark_data_parallel_shardings(
#         graphs,
#         activation_idx=num_params_buffers,
#         dp_strategy_map=strategy_map,
#         parallel_mode=parallel_mode,
#     )
#     for node in graphs[0].graph.nodes:
#         node_sharding = node.meta["sharding"]
#         if node_sharding is not None:
#             print(
#                 f">>> placeholder node: {node}, target: {node.target} sharding: {node.meta['sharding'].input_specs}"
#             )

#     for node in graphs[1].graph.nodes:
#         print(
#             f">>> placeholder node: {node}, target: {node.target} sharding: {node.meta['sharding']}"
#         )
#     # print(strategy_map)
#     partitioner(graphs[0])
#     partitioner(graphs[1])
#     #  print(f">>> forward graph: ")
#     # print(graphs[0].graph)
#     # print(f">>> forward graph shape: ")
#     # for node in graphs[0].graph.nodes:
#     #     tensor_meta = node.meta.get('val', None)
#     #     if tensor_meta is not None:
#     #         shape = tensor_meta.shape
#     #     else:
#     #         shape = ""

#     #     print(f"{node}, node shape {shape}")
#     # print(f">>> backward graph: ")
#     # print(graphs[1].graph)
#     # print(f">>> backward graph shape: ")
#     # for node in graphs[1].graph.nodes:
#     #     tensor_meta = node.meta.get('val', None)
#     #     if isinstance(tensor_meta, torch.Tensor):
#     #         shape = tensor_meta.shape
#     #     else:
#     #         shape = ""
#     #     print(f"{node}, node shape: {shape}")
#     # print(f">>> optim graph: ")
#     # compiled_f = aot_function(
#     #     functional_call, fwd_bwd_compiler, num_params_buffers=num_params_buffers, dynamic=True)

#     # # compiled_fw_bwd_gm = compiled_f(named_params, named_buffers, inputs, labels)
#     # # return compiled_f
#     # # compiling
#     # compiled_f(named_params, named_buffers, inputs, labels).backward()


# # def sharded_data_parallel(full_graph: GraphModule, )
# def run_sharded_data_parallel(rank, world_size):
#     # set up world pg
#     os.environ["MASTER_ADDR"] = "localhost"
#     os.environ["MASTER_PORT"] = "12355"

#     # initialize the process group
#     dist.init_process_group("nccl", rank=rank, world_size=world_size)
#     torch.cuda.set_device(rank)
#     # create a device mesh
#     mesh = DeviceMesh("cuda", torch.arange(world_size))
#     # DeviceMesh("cuda", [1])
#     # tensor = torch.randn(3, 4)
#     # dt = DTensor.from_local(tensor, mesh, [Replicate()])

#     model = SimpleMLP()

#     LR = 0.25
#     optimizer = torch.optim.SGD(model.parameters(), lr=LR)

#     def loss_fn(out, labels):
#         return (out - labels).sum()

#     x = torch.randn(batch_dim_size, 50)
#     y = torch.randn(batch_dim_size, 8)
#     compiled_fn(
#         model,
#         loss_fn,
#         optimizer,
#         (x, y),
#         mesh=mesh,
#         parallel_mode=DataParallelStyle.FULLY_SHARD,
#     )

#     # import pdb; pdb.set_trace()
#     # mesh = DeviceMesh("cpu", torch.arange(world_size))

#     # # create and shard the model in tensor parallel fashion
#     # model = Model()
#     # input = torch.randn(10, 10)
#     # print(model(input))

#     # shutting down world pg
#     dist.destroy_process_group()


# if __name__ == "__main__":
#     world_size = 2
#     mp.spawn(
#         run_sharded_data_parallel, args=(world_size,), nprocs=world_size, join=True
#     )
