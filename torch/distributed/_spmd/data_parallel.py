import operator
from contextlib import contextmanager
from enum import Enum

from typing import Any, cast, Dict, List, Optional, Tuple

import torch

import torch.distributed.distributed_c10d as c10d
import torch.fx as fx
import torch.library
import torch.nn as nn

import torch.utils._pytree as pytree

from torch.distributed._spmd.batch_dim_utils import BatchDimAnalyzer
from torch.distributed._tensor import DeviceMesh, distribute_tensor, Replicate, Shard

from torch.distributed._tensor._utils import compute_local_shape
from torch.distributed._tensor.op_schema import (
    OpStrategy,
    PlacementStrategy,
    StrategyType,
    TupleStrategy,
)
from torch.distributed._tensor.placement_types import _Partial, DTensorSpec, Placement
from torch.distributed._tensor.redistribute import redistribute_local_tensor
from torch.fx import GraphModule
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.passes.shape_prop import _extract_tensor_metadata
from torch.nn.utils._named_member_accessor import NamedMemberAccessor

aten = torch.ops.aten

# Dummy op used by data parallel to tag gradients.
_spmd_lib_def = torch.library.Library("_spmd", "DEF")
_spmd_lib_def.define("tag_grad(Tensor self) -> Tensor")

_spmd_lib_impl = torch.library.Library("_spmd", "IMPL")
_spmd_lib_impl.impl("tag_grad", lambda x: x, "CompositeExplicitAutograd")


class DataParallelStyle(Enum):
    """This enum represents the style of the data-parallel operation.

    We have three types of Data Parallel style:
    1. DEFAULT: the default data parallel style, which is to represent a mixed
                replicate and fully shard behavior. For each parameter that is able
                to be sharded evenly, we shard it, otherwise we would replicate the
                parameter. This style avoids potential padding if the parameters
                cannot be sharded evenly, but it would generate a mixed of all_reduce
                and reduce_scatter.
    2. REPLICATE: the data parallel style that replicates all model parameters.
                  This is similar to the behavior of DistributedDataParallel.
    3. FULLY_SHARD: the data parallel style that shards all model parameters. This
                    is similar to the behavior of FullyShardedDataParallel, the
                    difference is that FullyShardedDataParallel (ZERO-3), which
                    shards the model using FlatParameter based sharding,
                    while this style shards each parameter into DTensor.
    """

    DEFAULT = 0
    REPLICATE = 1
    FULLY_SHARD = 2


class NodeType(Enum):
    """NodeType is an enum that records the type of the tensors in the graph.

    This is used to determine the data parallel strategy.
    """

    PARAM = 0
    ACT = 1
    GRAD = 2
    STATE = 3
    NON_TENSOR = 4  # NON_TENSOR is to tag non tensor node (i.e. graph output)


class DataParallelStrategy(OpStrategy):
    """DataParallelStrategy is a special case of OpStrategy that only records the "data parallel style" placement
    strategy for each fx Node.

    It takes a list of PlacementStrategy, where each PlacementStrategy describes
    one way to distribute the tensor and computation. In the DataParallel case,
    there're two possible ways to distribute the parameters:
        1. replicate the parameter over a set of devices (DDP like behavior)
        2. shard the parameter on its tensor dimension 0 over a set of devices
           (FSDP like behavior).

    In addition to the strategy list, we also need to:
    1. `node_type`: record the type of each node in the graph, so that we can
        determine how to propagate in a data parallel fashion.
    2. `reduce_over_batch` is specifically tied to data parallel as the loss
        calculation usually results in scalar tensor where it comes from a
        reduction over the batch dimension. We need to know this information
        so that we could keep the output as sharded.
    """

    def __init__(
        self,
        node_type: NodeType,
        strategy_list: List[PlacementStrategy],
        reduction_over_batch: bool = False,
    ):
        super().__init__(strategy_list)
        self.node_type = node_type
        self.reduction_over_batch = reduction_over_batch

    def __str__(self) -> str:
        return f"type: {self.node_type}, {super().__str__()}"


@contextmanager
def gradients_tagging(params: Dict[str, torch.Tensor]):
    """Tag the gradient of the parameters with a special tag, so that we can identify them during SPMD expansion.

    It's safe to trace those hooks and we would remove those nodes later.
    """
    tagging_hooks = []
    try:
        for p in params.values():
            h = p.register_hook(torch.ops._spmd.tag_grad)
            tagging_hooks.append(h)
        yield
    finally:
        # remove those hooks after tracing
        for h in tagging_hooks:
            h.remove()


def _gen_shard_strategy(
    mesh: DeviceMesh, shard_dim: int, input_specs: Optional[List[DTensorSpec]] = None
) -> PlacementStrategy:
    """Util function to generate a shard strategy on shard_dim."""
    return PlacementStrategy(
        output_specs=DTensorSpec(mesh=mesh, placements=(Shard(shard_dim),)),
        input_specs=input_specs,
    )


def _gen_replicate_strategy(
    mesh: DeviceMesh, input_specs: Optional[List[DTensorSpec]] = None
) -> PlacementStrategy:
    """Util function to generate a replicate strategy."""
    return PlacementStrategy(
        output_specs=DTensorSpec(mesh=mesh, placements=(Replicate(),)),
        input_specs=input_specs,
    )


def _gen_partial_strategy(mesh: DeviceMesh) -> PlacementStrategy:
    """Util function to generate a partial strategy."""
    # NOTE: we use AVG by default, avg reduction is needed depending on
    # the loss function, for most loss function it should do
    # gradient averaging. There might be certain cases it should
    # not do gradient averaging (i.e. sum) but it's pretty rare.
    # TODO: Only NCCL supports AVG so using backend like Gloo would
    # crash, we should figure out a way to support avg reduction
    # for non-NCCL backend
    reduce_op = c10d.ReduceOp.AVG  # type: ignore[attr-defined]
    return PlacementStrategy(
        output_specs=DTensorSpec(mesh=mesh, placements=(_Partial(reduce_op),)),
    )


def build_data_parallel_strategies(
    train_step_graph: GraphModule,
    num_params: int,
    num_states: int,
    mesh: DeviceMesh,
    batch_dim: int = 0,
) -> Dict[fx.Node, StrategyType]:
    """Loop through the train step graph and build the data parallel strategy for each fx Node."""
    activation_idx = num_params + num_states
    non_compute_ops = [
        aten.clone.default,
        aten.detach.default,
        aten.ones_like.default,
        aten.reshape.default,
        aten.t.default,
        aten.view.default,
        torch.ops._spmd.tag_grad.default,
        operator.getitem,
    ]

    tuple_strategy_ops = [aten._fused_adam.default]

    dp_strategy_map: Dict[fx.Node, StrategyType] = {}
    batch_dim_analyzer = BatchDimAnalyzer(batch_dim)
    placeholder_idx = 0
    num_param_grad = 0

    # first we backward propagate to mark the param gradients sharding
    # with tag_grad node helps and then delete the tag_grad nodes
    for node in reversed(list(train_step_graph.graph.nodes)):
        # find a param_grad node via the tagging
        if node.target == torch.ops._spmd.tag_grad.default:
            cur_node = node
            while cur_node.target in non_compute_ops:
                cur_node = cur_node.args[0]
                partial_strategy = _gen_partial_strategy(mesh)
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

    # next we forward propagate to mark all the sharding
    for node in train_step_graph.graph.nodes:
        if node.op == "placeholder":
            if "val" not in node.meta:
                # NOTE: There're certain cases where the placeholder nodes do
                # not have real tensor values:
                # 1. optimizer states can be None sometimes, i.e. SGD with
                #    no momentum, optimizer states populate `momentum` state
                #    as None, the full graph we get from `compile` would have
                #    None as the placeholder value
                # 2. function args might not only contain params or activations,
                #    but also contain other non-tensor inputs, i.e. the model
                #    and optimizer instances baked in as a placeholder, there might
                #    also be some scalar argument which is not a tensor
                #
                # For the above cases, we create a NON_TENSOR stratgy so that we
                # know it's not a tensor and we don't need to shard it
                dp_strategy_map[node] = DataParallelStrategy(NodeType.NON_TENSOR, [])

            elif placeholder_idx < num_params:
                # during compilation there's an assumption that the first num_params
                # placeholders should be parameters
                shard_strategy = _gen_shard_strategy(mesh, 0)
                replica_strategy = _gen_replicate_strategy(mesh)
                dp_strategy_map[node] = DataParallelStrategy(
                    NodeType.PARAM, [replica_strategy, shard_strategy]
                )

            elif placeholder_idx < activation_idx:
                # optimizer states follow the same strategy as
                # the corresponding parameters
                replica_strategy = _gen_replicate_strategy(mesh)
                shard_strategy = _gen_shard_strategy(mesh, 0)

                dp_strategy_map[node] = DataParallelStrategy(
                    NodeType.STATE, [replica_strategy, shard_strategy]
                )
            else:
                activation_batch_dim_size = node.meta["val"].shape[batch_dim]
                # find the first activation node and use its batch dim size
                if batch_dim_analyzer.batch_dim_size == -1:
                    batch_dim_analyzer.init_batch_dim_size(activation_batch_dim_size)

                batch_dim_analyzer.set_batch_dim(node, batch_dim)
                shard_strategy = _gen_shard_strategy(mesh, batch_dim)
                dp_strategy_map[node] = DataParallelStrategy(
                    NodeType.ACT, [shard_strategy]
                )
            placeholder_idx += 1
        elif node.op == "call_function":
            # Annotate node types for the computation graph
            # Data Parallel node propagation logic:
            # param (non-compute) -> out: param
            # grad (non-compute before/after) -> out: grad
            # state -> output: state
            #
            # param + activation (param must be replicate, act be sharded) -> out: activation
            # param/state + grad (param/state/grad be the same spec) -> out: param/state
            # param + state -> out: param

            if node.target in non_compute_ops:
                # At this point, we should have removed all the `tag_grad` nodes in the graph
                assert node.target != torch.ops._spmd.tag_grad.default

                input_nodes = node.all_input_nodes
                assert (
                    len(input_nodes) == 1
                ), f"non-compute op only support one input now, found node: {node} with length of inputs: {len(node.args)}"
                arg_strategy = dp_strategy_map[input_nodes[0]]

                if node.target == operator.getitem:
                    # for getitem call, just forward the strategy from the input
                    getitem_idx = node.args[1]
                    if isinstance(arg_strategy, TupleStrategy):
                        # for tuple strategy, we need to get the child strategy from the tuple
                        dp_strategy_map[node] = arg_strategy.childs[getitem_idx]
                    else:
                        # if it's not a tuple strategy, we just forward the arg strategy
                        dp_strategy_map[node] = arg_strategy
                else:
                    assert isinstance(arg_strategy, DataParallelStrategy)
                    arg_node_type = arg_strategy.node_type
                    if arg_node_type == NodeType.PARAM:
                        replica_strategy = _gen_replicate_strategy(mesh)
                        dp_strategy_map[node] = DataParallelStrategy(
                            NodeType.PARAM, [replica_strategy]
                        )
                    elif arg_node_type == NodeType.GRAD:
                        partial_sig = _gen_partial_strategy(mesh)
                        dp_strategy_map[node] = DataParallelStrategy(
                            NodeType.GRAD, [partial_sig]
                        )
                    elif arg_node_type == NodeType.ACT:
                        arg_node_spec = batch_dim_analyzer.compute_act_spec(
                            input_nodes[0], mesh
                        )

                        output_spec = batch_dim_analyzer.compute_act_spec(node, mesh)

                        shard_strategy = PlacementStrategy(
                            output_specs=output_spec, input_specs=[arg_node_spec]
                        )
                        dp_strategy_map[node] = DataParallelStrategy(
                            NodeType.ACT, [shard_strategy]
                        )
                    else:
                        raise RuntimeError(
                            f"non compute op not supporting {arg_node_type}! "
                        )

                # finished processing this non-compute node
                continue

            # for computatation nodes, we need to check all the inputs
            input_args = node.all_input_nodes
            input_specs = []
            if node in dp_strategy_map:
                # found a param_grad node that already have output pre-filled spec
                # fill in the expected input specs for the pre-filled strategy
                node_strategy = dp_strategy_map[node]
                assert isinstance(node_strategy, DataParallelStrategy)
                node_type = node_strategy.node_type
                assert node_type == NodeType.GRAD
                produce_param_grad_strat = node_strategy.strategies
                has_activation = False
                for arg in input_args:
                    arg_strategy = dp_strategy_map[arg]
                    assert isinstance(arg_strategy, DataParallelStrategy)
                    arg_node_type = arg_strategy.node_type
                    if arg_node_type == NodeType.ACT:
                        # activation sharded
                        has_activation = True
                        act_spec = batch_dim_analyzer.compute_act_spec(arg, mesh)

                        input_specs.append(act_spec)

                if has_activation:
                    assert len(produce_param_grad_strat) == 1
                    produce_param_grad_strat[0].input_specs = input_specs
            elif node.target in tuple_strategy_ops:
                # ops that need to build tuple strategy instead of normal strategy
                # This should happen rarely and only needed when we need to generate
                # different node strategy for multiple outputs (i.e. fused_adam op)
                # TODO: Currently this specializes to fused optimizer ops, but we need
                # to see how to generalize this strategy building logic
                output_strategy_len = len(node.args) - 1
                tuple_strategies = []
                for i in range(output_strategy_len):
                    if not isinstance(node.args[i], list):
                        raise RuntimeError(
                            f"Expecting list as arg to build Tuple Strategy, but found type {type(node.args[i])}!"
                        )
                    # for list/tuple arg, use the first one to find out the node type
                    if len(node.args[i]) > 0:
                        arg_strategy = dp_strategy_map[node.args[i][0]]
                        assert isinstance(arg_strategy, DataParallelStrategy)
                        assert arg_strategy.node_type in [
                            NodeType.PARAM,
                            NodeType.GRAD,
                            NodeType.STATE,
                        ], "Expecting param/grad/state as arg to build Tuple Strategy!"
                        replica_strategy = _gen_replicate_strategy(mesh)
                        shard_strategy = _gen_shard_strategy(mesh, shard_dim=0)
                        out_node_strategy: StrategyType = DataParallelStrategy(
                            arg_strategy.node_type, [replica_strategy, shard_strategy]
                        )

                        tuple_strategies.append(out_node_strategy)

                output_tuple_strategy = TupleStrategy(tuple(tuple_strategies))
                dp_strategy_map[node] = output_tuple_strategy
            else:
                # NOTE: This is the common region for all regular computation ops

                input_node_types = [
                    cast(DataParallelStrategy, dp_strategy_map[arg]).node_type
                    for arg in input_args
                    if isinstance(dp_strategy_map[arg], DataParallelStrategy)
                ]
                if NodeType.GRAD in input_node_types:
                    # param/state + grad, build up acceptable strategy
                    # the strategy should be the same for all the inputs/outputs
                    # TODO: optimizer parts should follow the dtensor prop logic
                    # to support more general cases that allows optimizer states
                    # to have different shardings compare to the params
                    replica_strategy = _gen_replicate_strategy(mesh)
                    shard_strategy = _gen_shard_strategy(mesh, shard_dim=0)
                    output_node_type = NodeType.PARAM

                    non_grad_types = [t for t in input_node_types if t != NodeType.GRAD]

                    output_node_type = non_grad_types[0]
                    for non_grad_type in non_grad_types:
                        assert (
                            non_grad_type == output_node_type
                        ), f"Found more than one non grad types! Expect {output_node_type} but found {non_grad_type}!"
                    assert output_node_type in [
                        NodeType.PARAM,
                        NodeType.STATE,
                    ], f"Expecting output node type to be either state or param, but found {output_node_type}!"

                    dp_strategy_map[node] = DataParallelStrategy(
                        output_node_type, [replica_strategy, shard_strategy]
                    )
                elif NodeType.STATE in input_node_types:
                    # either param + state or state + state
                    replica_strategy = _gen_replicate_strategy(mesh)
                    shard_strategy = _gen_shard_strategy(mesh, shard_dim=0)
                    output_node_type = (
                        NodeType.PARAM
                        if NodeType.PARAM in input_node_types
                        else NodeType.STATE
                    )

                    dp_strategy_map[node] = DataParallelStrategy(
                        output_node_type, [replica_strategy, shard_strategy]
                    )
                elif NodeType.PARAM in input_node_types:
                    if NodeType.ACT in input_node_types:
                        # param + activation, build up acceptable strategy
                        # param must be replicated, activation must be sharded
                        for arg in input_args:
                            arg_strategy = dp_strategy_map[arg]
                            assert isinstance(arg_strategy, DataParallelStrategy)
                            node_type = arg_strategy.node_type
                            if node_type == NodeType.ACT:
                                # compute activation spec
                                act_spec = batch_dim_analyzer.compute_act_spec(
                                    arg, mesh
                                )

                                input_specs.append(act_spec)
                            elif node_type == NodeType.PARAM:
                                # param must be replicated
                                input_specs.append(
                                    DTensorSpec(mesh=mesh, placements=(Replicate(),))
                                )
                            else:
                                raise RuntimeError(
                                    f"Expecting node with parameter and activation, but found {input_node_types}! "
                                )
                        # produce activation type sharding for output
                        output_spec = batch_dim_analyzer.compute_act_spec(node, mesh)

                        act_strategy = PlacementStrategy(
                            output_specs=output_spec, input_specs=input_specs
                        )

                        dp_strategy_map[node] = DataParallelStrategy(
                            NodeType.ACT, [act_strategy]
                        )
                    else:
                        # If inputs only have parameters, the
                        # strategy of this node should follow input
                        dp_strategy_map[node] = dp_strategy_map[input_args[0]]
                else:
                    # If input nodes does not have PARAM/GRAD/STATE, then
                    # it should be a pure activation computation, it should
                    # produce activation output.
                    # Activations are usually sharded unless model creates
                    # new tensors during computation, which depend on whether
                    # the new tensor associate with a batch dim or not, it could
                    # be shard/replicate/partial, batch dim analyzer should tell
                    # us the correct sharding.
                    for arg in input_args:
                        arg_strategy = dp_strategy_map[arg]
                        assert isinstance(arg_strategy, DataParallelStrategy)
                        input_spec = batch_dim_analyzer.compute_act_spec(arg, mesh)

                        input_specs.append(input_spec)

                    act_spec = batch_dim_analyzer.compute_act_spec(node, mesh)
                    op_strategy = PlacementStrategy(
                        output_specs=act_spec, input_specs=input_specs
                    )
                    dp_strategy_map[node] = DataParallelStrategy(
                        NodeType.ACT, [op_strategy]
                    )

        elif node.op == "output":
            dp_strategy_map[node] = DataParallelStrategy(NodeType.NON_TENSOR, [])
        else:
            raise RuntimeError(f"op code {node.op} not supported")

    return dp_strategy_map  # type: ignore[return-value]


def mark_data_parallel_shardings(
    train_step_graph: GraphModule,
    num_parameters: int,
    num_states: int,
    dp_strategy_map: Dict[fx.Node, StrategyType],
    parallel_mode: DataParallelStyle = DataParallelStyle.FULLY_SHARD,
) -> None:
    """Mark the sharding for the nodes in the train_step_graph."""
    activation_idx = num_parameters + num_states
    placeholder_idx = 0
    for node in train_step_graph.graph.nodes:
        node_strategy = dp_strategy_map[node]
        if node.op == "placeholder":
            assert isinstance(node_strategy, DataParallelStrategy)
            node_type = node_strategy.node_type
            node_strategies = node_strategy.strategies
            if node_type == NodeType.NON_TENSOR:
                # set node sharding to None
                node_sharding = None
            elif placeholder_idx < activation_idx:
                assert len(node_strategies) > 0, "node_strategies should not be empty"
                if parallel_mode == DataParallelStyle.REPLICATE:
                    # set to replicate for replicate style
                    node_sharding = node_strategies[0]
                elif parallel_mode == DataParallelStyle.FULLY_SHARD:
                    # set to shard for fully shard style
                    if len(node_strategies) == 1:
                        # only one strategy, use that instead
                        # i.e. optimizer state steps can only be replicate
                        node_sharding = node_strategies[0]
                    else:
                        # use the full sharding strategy
                        node_sharding = node_strategies[1]
                elif parallel_mode == DataParallelStyle.DEFAULT:
                    # TODO: add support for default mode
                    # default mode would generate either replicate or shard
                    raise NotImplementedError("default mode not implemented")
            else:
                assert len(node_strategies) > 0, "node_strategies should not be empty"
                # mark activation as sharded on batch dim
                node_sharding = node_strategies[0]

            node.meta["sharding"] = node_sharding  # type: ignore[possibly-undefined]

            placeholder_idx += 1
        elif node.op == "call_function":
            if isinstance(node_strategy, TupleStrategy):
                # For tuple strategy in the data parallel mode, it should have the same strategy
                # for all tuple elements, assert that then use the first element's strategy as sharding
                first_strategy = cast(DataParallelStrategy, node_strategy.childs[0])
                for child_strategy in node_strategy.childs:
                    assert isinstance(child_strategy, DataParallelStrategy)
                    assert child_strategy.strategies == first_strategy.strategies

                node_strategies = first_strategy.strategies
            else:
                assert isinstance(node_strategy, DataParallelStrategy)
                node_strategies = node_strategy.strategies

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
            else:
                raise RuntimeError(
                    f"node {node} strategy length {len(node_strategies)} is not expected!"
                )
        elif node.op == "output":
            assert (
                isinstance(node_strategy, DataParallelStrategy)
                and node_strategy.node_type == NodeType.NON_TENSOR
            ), "output node should not be tensor"
            node.meta["sharding"] = None
        else:
            raise RuntimeError(f"op code {node.op} not supported")


def _partition_val(val: Any, spec: DTensorSpec) -> Any:
    """Util function to convert a full tensor val to its local component."""
    if isinstance(val, torch.Tensor):
        local_shard = val
        if val.ndim == 0:
            # If it's already a scalar tensor, it is already local, we don't
            # need to do anything
            return local_shard

        for idx, placement in enumerate(spec.placements):
            if placement.is_shard():
                placement = cast(Shard, placement)
                num_chunks = spec.mesh.size(mesh_dim=idx)
                my_coord = spec.mesh.get_coordinate()
                assert my_coord is not None, "current rank not in mesh!"
                my_coord_on_mesh_dim = my_coord[idx]
                local_shard = placement._split_tensor(
                    local_shard, num_chunks, with_padding=False, contiguous=False
                )[0][my_coord_on_mesh_dim]
        return local_shard
    elif isinstance(val, (tuple, list)):
        return val.__class__(_partition_val(v, spec) for v in val)
    else:
        raise RuntimeError(f"val type {type(val)} not supported")


def partitioner(graph: GraphModule) -> GraphModule:
    """Graph partitioner that partitions the single device graph to distributed graph."""
    shape_adjustment_ops = {
        aten._unsafe_view.default: 1,
        aten.expand.default: 1,
        aten.new_zeros.default: 1,
        aten.ones.default: 0,
        aten.reshape.default: 1,
        aten.view.default: 1,
        aten.zeros.default: 0,
    }
    # partition the graph to distributed
    for node in graph.graph.nodes:
        node_sharding = node.meta["sharding"]
        # None sharding means this node don't need sharding
        if node_sharding is None:
            continue

        if node.op == "placeholder":
            out_spec = node_sharding.output_spec
            if not hasattr(out_spec, "from_local"):
                local_val = _partition_val(node.meta["val"], out_spec)
                # update node value
                node.meta["val"] = local_val
        elif node.op == "call_function":
            out_spec = node_sharding.output_spec

            # check if there's misaligned sharding, insert reshard if there is
            expected_input_specs = node_sharding.input_specs
            for idx, input_arg in enumerate(node.all_input_nodes):
                input_arg_sharding = input_arg.meta["sharding"]

                input_arg_spec = input_arg_sharding.output_spec
                desired_spec = (
                    out_spec
                    if expected_input_specs is None
                    else expected_input_specs[idx]
                )
                if input_arg_spec != desired_spec:
                    input_arg_spec.tensor_meta = input_arg.meta["tensor_meta"]
                    desired_spec.tensor_meta = input_arg.meta["tensor_meta"]
                    input_arg_tensor = input_arg.meta["val"]

                    # insert reshard operation
                    def reshard_fn(local_tensor: torch.Tensor) -> torch.Tensor:
                        return redistribute_local_tensor(
                            local_tensor,
                            input_arg_spec,
                            desired_spec,
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

            output_val = node.meta["val"]

            if node.target == torch.ops.aten.repeat.default:
                # for repeat op, we need to infer the repeat sizes
                assert isinstance(output_val, torch.Tensor)
                local_shape = compute_local_shape(
                    output_val.shape, out_spec.mesh, out_spec.placements
                )
                input_shape = node.args[0].meta["val"].shape

                def infer_repeat_sizes(repeated_shape, input_shape):
                    repeated_size = [1] * len(repeated_shape)
                    padded_length = len(repeated_shape) - len(input_shape)
                    for i in range(len(repeated_shape)):
                        if i < padded_length:
                            repeated_size[i] = repeated_shape[i]
                        else:
                            repeated_size[i] = (
                                repeated_shape[i] // input_shape[i - padded_length]
                            )

                    return repeated_size

                node.update_arg(1, infer_repeat_sizes(local_shape, input_shape))

            elif node.target in shape_adjustment_ops:
                # for view related op that needs shape, adjust shape to local shape if needed
                assert isinstance(output_val, torch.Tensor)
                local_shape = compute_local_shape(
                    output_val.shape, out_spec.mesh, out_spec.placements
                )
                shape_arg_num = shape_adjustment_ops[node.target]
                node.update_arg(shape_arg_num, local_shape)

            # convert output val to its local component
            node.meta["val"] = _partition_val(output_val, out_spec)

        elif node.op == "output":
            break
        else:
            raise RuntimeError(f"op code {node} not supported")

    # clean up the graph by removing sharding and partitioning related metadata
    for node in graph.graph.nodes:
        if "sharding" in node.meta:
            del node.meta["sharding"]
        if "val" in node.meta and isinstance(node.meta["val"], torch.Tensor):
            local_tensor_meta = _extract_tensor_metadata(node.meta["val"])
            node.meta["tensor_meta"] = local_tensor_meta

    graph.graph.lint()
    graph.recompile()
    return graph


def partition_data_parallel(
    graph: GraphModule,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    params_buffers: Dict[str, torch.Tensor],
    named_states: Dict[str, Any],
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    mesh: DeviceMesh,
    parallel_style: DataParallelStyle,
    input_batch_dim: int,
) -> GraphModule:
    """Partition the graph to into a data parallel graph.

    This function also shards/replicates the model parameters and optimizer states to DTensors.
    """
    num_params_buffers = len(params_buffers)
    flattened_states = pytree.tree_leaves(named_states)
    num_states = len(flattened_states)

    changed = graph.graph.eliminate_dead_code()
    if changed:
        graph.recompile()

    # 1. First build up data parallel strategies for the whole graph
    strategy_map = build_data_parallel_strategies(
        graph, num_params_buffers, num_states, mesh=mesh, batch_dim=input_batch_dim
    )

    # 2. Next we mark the data parallel strategy for each node base on
    #    the parallel_style
    mark_data_parallel_shardings(
        graph,
        num_parameters=num_params_buffers,
        num_states=num_states,
        dp_strategy_map=strategy_map,
        parallel_mode=parallel_style,
    )

    # 3. Partition the single machine graph to the distribute graph
    partitioned_graph = partitioner(graph)

    # preserve node types for the expanded graph
    for node in partitioned_graph.graph.nodes:
        if node in strategy_map:
            node_strategy = strategy_map[node]
            if isinstance(node_strategy, DataParallelStrategy):
                node.meta["node_type"] = node_strategy.node_type
            elif isinstance(node_strategy, TupleStrategy):
                node.meta["node_type"] = NodeType.NON_TENSOR
            else:
                raise RuntimeError(f"Unknown node strategy {node_strategy}")
        else:
            # if the nodes are expanded nodes (collectives), we mark them
            # the same type as the input node.
            input_node = node.all_input_nodes[0]
            node.meta["node_type"] = input_node.meta["node_type"]

    # 4. Last, inplace partition the weights and optim states to
    #    DTensors base on the parallel style
    accessor = NamedMemberAccessor(model)
    for param_key, param in params_buffers.items():
        placement: Placement = Replicate()
        if parallel_style == DataParallelStyle.FULLY_SHARD:
            placement = Shard(0)
        elif parallel_style != DataParallelStyle.REPLICATE:
            raise RuntimeError(f"parallel style {parallel_style} not supported yet")

        dtensor_param = distribute_tensor(param, mesh, [placement])
        # update re-parameterized module param dict and optim states dict to DTensor
        params_buffers[param_key] = dtensor_param.to_local()
        # update module parameters to DTensor
        accessor.set_tensor(param_key, dtensor_param)

        # update the optimizer state key and values to DTensor
        if optimizer is not None and param in optimizer.state:
            param_states = named_states[param_key]
            param_dtensor_states = {}
            for state_key, state_val in param_states.items():
                if isinstance(state_val, torch.Tensor) and state_val.ndim > 0:
                    # shard/replicate non-scalar tensors, for scalar tensor, we
                    # don't do anything
                    dtensor_state = distribute_tensor(state_val, mesh, [placement])
                    param_dtensor_states[state_key] = dtensor_state
                    param_states[state_key] = dtensor_state.to_local()
                else:
                    param_dtensor_states[state_key] = state_val

            optimizer.state.pop(param)  # type: ignore[call-overload]
            optimizer.state[dtensor_param] = param_dtensor_states  # type: ignore[index]

    return partitioned_graph
