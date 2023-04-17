import operator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum

from typing import Any, cast, Dict, List, Optional, Tuple

import torch
import torch.fx as fx
import torch.library
import torch.nn as nn

import torch.utils._pytree as pytree

from torch.distributed._tensor import DeviceMesh, distribute_tensor, Replicate, Shard

from torch.distributed._tensor._utils import compute_local_shape
from torch.distributed._tensor.op_schema import PlacementStrategy, StrategyList
from torch.distributed._tensor.placement_types import _Partial, DTensorSpec
from torch.distributed._tensor.redistribute import _redistribute_with_local_tensor
from torch.fx import GraphModule
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.passes.shape_prop import _extract_tensor_metadata
from torch.nn.utils._named_member_accessor import NamedMemberAccessor


aten = torch.ops.aten

# Dummy op used by data parallel to tag gradients.
_spmd_lib_def = torch.library.Library("_spmd", "DEF")
_spmd_lib_def.define("tag_grad(Tensor self) -> Tensor")

_spmd_lib_impl = torch.library.Library("_spmd", "IMPL")
for dispatch_key in ("CPU", "CUDA", "Meta"):
    _spmd_lib_impl.impl("tag_grad", lambda x: x, dispatch_key)


class DataParallelStyle(Enum):
    """
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
                    while this style shard each parameter into DTensor.
    """

    DEFAULT = 0
    REPLICATE = 1
    FULLY_SHARD = 2


class NodeType(Enum):
    """
    NodeType is a enum that records the type of the tensors in the graph.
    This is used to determine the data parallel strategy.
    """

    PARAM = 0
    ACT = 1
    GRAD = 2
    STATE = 3
    NON_TENSOR = 4  # NON_TENSOR is to tag non tensor node (i.e. graph output)


@dataclass
class DataParallelStrategy(StrategyList):
    """
    DataParallelStrategy is a special case of StrategyList that only records
    the "data parallel style" placement strategy for each fx Node.
    """

    node_type: NodeType
    reduction_over_batch: bool

    def __init__(
        self,
        node_type: NodeType,
        startegy_list: List[PlacementStrategy],
        reduction_over_batch: bool = False,
    ):
        super().__init__(startegy_list)
        self.node_type = node_type
        self.reduction_over_batch = reduction_over_batch

    def __str__(self) -> str:
        return f"type: {self.node_type}, {super().__str__()}"


@contextmanager
def gradients_tagging(params: Dict[str, torch.Tensor]):
    """
    This is a helper function that tags the gradient of the parameters
    with a special tag, so that we can identify them during SPMD expansion.

    It's safe to trace those hooks and we would remove those nodes later.
    """

    tagging_hooks = []
    try:
        for p in params.values():
            h = p.register_hook(lambda grad: torch.ops._spmd.tag_grad(grad))
            tagging_hooks.append(h)
        yield
    finally:
        # remove those hooks after tracing
        for h in tagging_hooks:
            h.remove()


class BatchDimAnalyzer(object):
    """
    This class is used to analyze the batch dimension of each tensor/node in the
    graph. We need to know the batch dimension of each tensor/node so that we know
    exactly the sharding layout of intermediate tensors.

    We possibly should use symbolic shapes to track the batch dimension, but this
    needs to happen with dynamo, as dynamo is the only place we can mark a single
    dimension. For now, we just use the batch dimension of the first input tensor
    as the hint to track the batch dimension of all tensors/nodes in the graph.
    """

    def __init__(self, batch_dim: int = 0) -> None:
        self.batch_dim = batch_dim

        if batch_dim != 0:
            # TODO: see if this make sense or not
            raise RuntimeError("Data Parallel only supports batch dim on dimension 0!")

        self.batch_dim_map: Dict[fx.Node, int] = {}
        # batch dim size is used to track the batch dim size of the input tensor
        self.batch_dim_size = -1

    def init_batch_dim_size(self, batch_dim_size: int) -> None:
        """
        initialize batch dim size base on the first input batch size
        """
        if self.batch_dim_size != -1 and self.batch_dim_size != batch_dim_size:
            raise RuntimeError(
                f"batch dim size is already initialized! "
                f"Found new batch size: {batch_dim_size} not "
                f"matching existing batch dim size: {self.batch_dim_size}!"
            )
        self.batch_dim_size = batch_dim_size

    def set_batch_dim(self, node: fx.Node, batch_dim: int) -> None:
        self.batch_dim_map[node] = batch_dim

    def get_batch_dim(self, node: fx.Node) -> int:
        if node not in self.batch_dim_map:
            raise RuntimeError(f"batch dim analysis failed on node: {node}!")
        return self.batch_dim_map[node]

    def get_batch_dim_shard_spec(
        self, node: fx.Node, mesh: DeviceMesh, input_full_reduction: bool = False
    ) -> Tuple[DTensorSpec, bool]:
        """
        simple batch dim analysis that analyze the batch dim of the node
        and return the corresponding shard spec
        """
        assert self.batch_dim_size != -1, "batch dim size is not initialized!"

        if node in self.batch_dim_map:
            node_batch_dim = self.get_batch_dim(node)
            batch_dim_shard_spec = DTensorSpec(
                mesh=mesh, placements=[Shard(node_batch_dim)]
            )
            return batch_dim_shard_spec, False

        shape = node.meta["val"].shape

        # for reduction op that reduces over the sharded batch dim
        # we don't generate partial, but rather, we generate shard
        # This is because the intention of data parallel is to never
        # do full reduction across batch dimension, it would still
        # keep the reduction activation sharded.
        reduction_over_batch = False
        reduction_ops = [aten.sum.default, aten.mean.default]
        if node.target in reduction_ops and len(shape) == 0:
            operand = node.all_input_nodes[0]
            if operand in self.batch_dim_map:
                operand_batch_dim = self.get_batch_dim(operand)
                if operand_batch_dim == 0:
                    reduction_over_batch = True
                    self.set_batch_dim(node, operand_batch_dim)

            else:
                raise RuntimeError(f"batch dim analysis failed on node: {node}!")
        elif input_full_reduction:
            # the first consumer node that consumes the full reduction
            operand = node.all_input_nodes[0]
            assert (
                operand in self.batch_dim_map
            ), "input have full reduction but not in dim map!"
            self.set_batch_dim(node, self.get_batch_dim(operand))
        else:
            for i, dim_size in enumerate(shape):
                if dim_size == self.batch_dim_size:
                    self.set_batch_dim(node, i)

        node_batch_dim = self.get_batch_dim(node)
        batch_dim_shard_spec = DTensorSpec(
            mesh=mesh, placements=[Shard(node_batch_dim)]
        )
        if reduction_over_batch or input_full_reduction:
            batch_dim_shard_spec.from_local = True  # type: ignore[attr-defined]

        return batch_dim_shard_spec, reduction_over_batch


def _gen_shard_strategy(
    mesh: DeviceMesh, shard_dim: int, input_specs: Optional[List[DTensorSpec]] = None
) -> PlacementStrategy:
    """
    util function to generate a shard strategy on shard_dim
    """
    return PlacementStrategy(
        output_spec=DTensorSpec(mesh=mesh, placements=[Shard(shard_dim)]),
        input_specs=input_specs,
    )


def _gen_replicate_strategy(
    mesh: DeviceMesh, input_specs: Optional[List[DTensorSpec]] = None
) -> PlacementStrategy:
    """
    util function to generate a replicate strategy
    """
    return PlacementStrategy(
        output_spec=DTensorSpec(mesh=mesh, placements=[Replicate()]),
        input_specs=input_specs,
    )


def _gen_partial_strategy(mesh: DeviceMesh) -> PlacementStrategy:
    """
    util function to generate a partial strategy
    """
    return PlacementStrategy(
        output_spec=DTensorSpec(mesh=mesh, placements=[_Partial()]),
    )


def build_data_parallel_strategies(
    train_step_graph: GraphModule,
    num_params: int,
    num_states: int,
    mesh: DeviceMesh,
    batch_dim: int = 0,
) -> Dict[fx.Node, DataParallelStrategy]:
    """
    This function loop through the train step graph and build the
    data parallel strategy for each fx Node
    """
    activation_idx = num_params + num_states
    non_compute_ops = [
        aten.ones_like.default,
        aten.t.default,
        aten.view.default,
        aten.reshape.default,
        aten.detach.default,
        aten.clone.default,
        torch.ops._spmd.tag_grad.default,
        operator.getitem,
    ]

    dp_strategy_map = {}
    batch_dim_analzer = BatchDimAnalyzer(batch_dim)
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

    train_step_graph.recompile()

    # next we forward propagate to mark all the sharding
    for node in train_step_graph.graph.nodes:
        if node.op == "placeholder":
            if "val" not in node.meta:
                # NOTE: There're certain cases where the placeholder nodes does
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
                # during compilation there's a assumption that the first num_params
                # placeholders should be parameters
                shard_strategy = _gen_shard_strategy(mesh, 0)
                replica_strategy = _gen_replicate_strategy(mesh)
                dp_strategy_map[node] = DataParallelStrategy(
                    NodeType.PARAM, [replica_strategy, shard_strategy]
                )

            elif placeholder_idx < activation_idx:
                # optimizer states follow the same strategy as
                # the corresponding parameters
                shard_strategy = _gen_shard_strategy(mesh, 0)
                replica_strategy = _gen_replicate_strategy(mesh)
                dp_strategy_map[node] = DataParallelStrategy(
                    NodeType.STATE, [replica_strategy, shard_strategy]
                )
            else:
                activation_batch_dim_size = node.meta["val"].shape[batch_dim]
                # find the first activation node and use its batch dim size
                if batch_dim_analzer.batch_dim_size == -1:
                    batch_dim_analzer.init_batch_dim_size(activation_batch_dim_size)

                batch_dim_analzer.set_batch_dim(node, batch_dim)
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
                input_nodes = node.all_input_nodes
                assert (
                    len(input_nodes) == 1
                ), f"non-compute op only support one input now, found node: {node} with length of inputs: {len(node.args)}"
                arg_node_type = dp_strategy_map[input_nodes[0]].node_type
                input_full_reduction = dp_strategy_map[
                    input_nodes[0]
                ].reduction_over_batch

                if node.target == operator.getitem:
                    # for getitem call, just forward the strategy from the input
                    dp_strategy_map[node] = dp_strategy_map[node.args[0]]
                elif arg_node_type == NodeType.PARAM:
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
                    arg_node_spec, _ = batch_dim_analzer.get_batch_dim_shard_spec(
                        input_nodes[0], mesh
                    )

                    output_spec, _ = batch_dim_analzer.get_batch_dim_shard_spec(
                        node, mesh, input_full_reduction
                    )

                    shard_strategy = PlacementStrategy(
                        output_spec=output_spec, input_specs=[arg_node_spec]
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
            input_node_types = [dp_strategy_map[arg].node_type for arg in input_args]
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
                        act_spec, _ = batch_dim_analzer.get_batch_dim_shard_spec(
                            arg, mesh
                        )

                        input_specs.append(act_spec)

                if has_activation:
                    assert len(produce_param_grad_strat) == 1
                    produce_param_grad_strat[0].input_specs = input_specs
            else:
                if NodeType.ACT in input_node_types:
                    # param + activation, build up acceptable strategy
                    # param must be replicated, activation must be sharded
                    for arg in input_args:
                        arg_strategies = dp_strategy_map[arg]
                        node_type = arg_strategies.node_type
                        if node_type == NodeType.ACT:
                            # activation must stay sharded
                            act_spec, _ = batch_dim_analzer.get_batch_dim_shard_spec(
                                arg, mesh
                            )

                            input_specs.append(act_spec)
                        elif node_type == NodeType.PARAM:
                            # param must be replicated
                            input_specs.append(
                                DTensorSpec(mesh=mesh, placements=[Replicate()])
                            )
                        else:
                            raise RuntimeError(
                                f"Expecting node with parameter and activation, but found {input_node_types}! "
                            )
                    # produce activation type sharding for output
                    (
                        output_spec,
                        reduction_over_batch,
                    ) = batch_dim_analzer.get_batch_dim_shard_spec(node, mesh)

                    shard_strategy = PlacementStrategy(
                        output_spec=output_spec, input_specs=input_specs
                    )

                    dp_strategy_map[node] = DataParallelStrategy(
                        NodeType.ACT, [shard_strategy], reduction_over_batch
                    )
                elif NodeType.GRAD in input_node_types:
                    # param/state + grad, build up acceptable strategy
                    # the strategy should be the same for all the inputs/outputs
                    # TODO: optimizer parts should follow the dtensor prop logic
                    # to support more general cases that allows optimizer states
                    # to have different shardings compare to the params
                    replica_strategy = _gen_replicate_strategy(mesh)
                    shard_strategy = _gen_shard_strategy(mesh, batch_dim)
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
                    shard_strategy = _gen_shard_strategy(mesh, batch_dim)
                    output_node_type = (
                        NodeType.PARAM
                        if NodeType.PARAM in input_node_types
                        else NodeType.STATE
                    )

                    dp_strategy_map[node] = DataParallelStrategy(
                        output_node_type, [replica_strategy, shard_strategy]
                    )
                elif NodeType.PARAM in input_node_types:
                    # at this point, inputs should only have parameters, the
                    # strategy of this node would be the same as the input
                    dp_strategy_map[node] = dp_strategy_map[input_args[0]]
                else:
                    raise RuntimeError(
                        f"Unrecognized node: {node} with input node types: {input_node_types}."
                    )

        elif node.op == "output":
            dp_strategy_map[node] = DataParallelStrategy(NodeType.NON_TENSOR, [])
        else:
            raise RuntimeError(f"op code {node.op} not supported")

    return dp_strategy_map


def mark_data_parallel_shardings(
    train_step_graph: GraphModule,
    num_parameters: int,
    num_states: int,
    dp_strategy_map: Dict[fx.Node, DataParallelStrategy],
    parallel_mode: DataParallelStyle = DataParallelStyle.FULLY_SHARD,
) -> None:
    """
    This function marks the sharding for the nodes in the train_step_graph
    """
    activation_idx = num_parameters + num_states
    placeholder_idx = 0
    for node in train_step_graph.graph.nodes:
        node_type = dp_strategy_map[node].node_type
        if node.op == "placeholder":
            node_strategies = dp_strategy_map[node].strategies
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
                    node_sharding = node_strategies[1]
                elif parallel_mode == DataParallelStyle.DEFAULT:
                    # todo: add support for default mode
                    # default mode would generate either replicate or shard
                    raise NotImplementedError("default mode not implemented")
            else:
                assert len(node_strategies) > 0, "node_strategies should not be empty"
                # mark activation as sharded on batch dim
                node_sharding = node_strategies[0]

            node.meta["sharding"] = node_sharding

            placeholder_idx += 1
        elif node.op == "call_function":
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
            else:
                raise RuntimeError(
                    f"node {node} strategy length {len(node_strategies)} is not expected!"
                )
        elif node.op == "output":
            assert node_type == NodeType.NON_TENSOR, "output node should not be tensor"
            node.meta["sharding"] = None
        else:
            raise RuntimeError(f"op code {node.op} not supported")


def _to_local_shard(
    full_tensor: torch.Tensor, spec: DTensorSpec, from_local: bool = False
) -> torch.Tensor:
    """
    util function to convert a full tensor to its local component
    """
    local_shard = full_tensor
    if from_local:
        # flag indicate the tensor is already local, we don't need to do anything
        return local_shard

    for idx, placement in enumerate(spec.placements):
        if placement.is_shard():
            placement = cast(Shard, placement)
            num_chunks = spec.mesh.size(dim=idx)
            my_coord = spec.mesh.get_coordinate()
            assert my_coord is not None, "current rank not in mesh!"
            my_coord_on_mesh_dim = my_coord[idx]
            local_shard = placement._split_tensor(
                local_shard, num_chunks, contiguous=False
            )[0][my_coord_on_mesh_dim]
    return local_shard


def partitioner(graph: GraphModule) -> GraphModule:
    """
    Graph partitioner that partitions the single device graph
    to distributed graph
    """

    shape_adjustment_ops = [
        aten.view.default,
        aten.reshape.default,
        aten.expand.default,
    ]
    # partition the graph to distributed
    for node in graph.graph.nodes:
        node_sharding = node.meta["sharding"]
        # None sharding means this node don't need sharding
        if node_sharding is None:
            continue

        print(f">>>> processing node: {node}")

        if node.op == "placeholder":
            out_spec = node_sharding.output_spec
            if not hasattr(out_spec, "from_local"):
                local_val = _to_local_shard(node.meta["val"], out_spec)
                # local_tensor_meta = _extract_tensor_metadata(local_val)
                # update metadata
                node.meta["val"] = local_val
                # node.meta["tensor_meta"] = local_tensor_meta
        elif node.op == "call_function":
            out_spec = node_sharding.output_spec
            if "val" not in node.meta:
                print(f">>>> node: {node} has no val, meta: {node.meta}")

            output_val = node.meta["val"]

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
                    input_full_shape = input_arg.meta["tensor_meta"].shape

                    # insert reshard operation
                    def reshard_fn(local_tensor: torch.Tensor) -> torch.Tensor:
                        return _redistribute_with_local_tensor(
                            local_tensor,
                            input_full_shape,
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
            if node.target in shape_adjustment_ops:
                # for view related op that needs shape, adjust shape if needed
                assert isinstance(output_val, torch.Tensor)
                local_shape = compute_local_shape(
                    output_val.shape, out_spec.mesh, out_spec.placements
                )
                node.update_arg(1, local_shape)

            # convert output tensor to local shard
            # note that if the output is already local, we don't need to convert
            if isinstance(output_val, list):
                new_local_list = []
                for tensor_val in output_val:
                    local_val = _to_local_shard(tensor_val, out_spec)
                    new_local_list.append(local_val)
                node.meta["val"] = new_local_list
            else:
                if isinstance(output_val, torch.Tensor):
                    if not hasattr(out_spec, "from_local"):
                        local_val = _to_local_shard(output_val, out_spec)
                        node.meta["val"] = local_val
                else:
                    raise RuntimeError(f"Output type {type(output_val)} not supported!")

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
) -> GraphModule:
    """
    The entry point function to partition the graph to data parallel
    graph, it also shard/replicate the model parameters and optimizer
    states to DTensors.
    """
    num_params_buffers = len(params_buffers)
    flattened_states = pytree.tree_flatten(named_states)[0]
    num_states = len(flattened_states)
    # 1. First build up data parallel strategies for the whole graph
    strategy_map = build_data_parallel_strategies(
        graph, num_params_buffers, num_states, mesh=mesh
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

    # 4. Last, inplace partition the weights and optim states to
    #    DTensors base on the parallel style
    accessor = NamedMemberAccessor(model)
    for param_key, param in params_buffers.items():
        if parallel_style == DataParallelStyle.REPLICATE:
            placement = Replicate()
        elif parallel_style == DataParallelStyle.FULLY_SHARD:
            placement = Shard(0)
        else:
            raise RuntimeError(f"parallel style {parallel_style} not supported yet")

        dtensor_param = distribute_tensor(param, mesh, [placement])
        # update re-parameterized module param dict and optim states dict to DTensor
        params_buffers[param_key] = dtensor_param.to_local()
        # update module parameters to DTensor
        accessor.set_tensor(param_key, dtensor_param)

        # update the optimizer state key and values to DTensor
        if param_key in named_states:
            assert optimizer is not None, "Can't find optimizer!"
            assert (
                param in optimizer.state
            ), f"param {param_key} not in optimizer state!"
            param_states = named_states[param_key]
            param_dtensor_states = {}
            for state_key, state_val in param_states.items():
                if isinstance(state_val, torch.Tensor):
                    dtensor_state = distribute_tensor(state_val, mesh, [placement])
                    param_dtensor_states[state_key] = dtensor_state
                    param_states[state_key] = dtensor_state.to_local()
                else:
                    param_dtensor_states[state_key] = state_val

            del optimizer.state[param]
            optimizer.state[dtensor_param] = param_dtensor_states

    return partitioned_graph
