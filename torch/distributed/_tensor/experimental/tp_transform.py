import copy
import operator
from typing import Any, cast, Dict, List, Optional, Sequence, Set, Tuple

import torch
from torch.distributed._tensor import DeviceMesh, distribute_tensor, DTensor
from torch.distributed._tensor.op_schema import (
    DTensorSpec,
    OpSchema,
    OutputSharding,
    OutputSpecType,
    PlacementStrategy,
)
from torch.distributed._tensor.placement_types import (
    Placement,
    Replicate,
    Shard,
    TensorMeta,
)
from torch.distributed._tensor.redistribute import redistribute_local_tensor
from torch.export import ExportedProgram
from torch.export.exported_program import ExportGraphSignature
from torch.fx import GraphModule
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.node import Node
from torch.fx.passes.infra.pass_base import PassBase, PassResult
from torch.fx.passes.shape_prop import _extract_tensor_metadata


aten = torch.ops.aten


def tensor_parallel_transformation(
    exported_program: ExportedProgram,
    rank: int,
    world_size: int,
) -> ExportedProgram:
    """
    The entry point function to perform graph transformations on an exported program
    to transform a single-device graph into a tensor parallel graph.

    .. warning::
        This API is experimental and subject to change.
    """
    return exported_program._transform(
        TensorParallelTransformPass(
            rank,
            world_size,
            exported_program.state_dict,
            exported_program.graph_signature,
        )
    )


class TensorParallelTransformPass(PassBase):
    """
    This pass is responsible for transforming a single-device graph into a tensor parallel
    graph. It will mark the placement strategy of each node in the graph,
    partition the graph into distributed graph, then shard the parameters/buffers accordingly.
    """

    def __init__(
        self,
        rank: int,
        world_size: int,
        state_dict: Dict[str, torch.Tensor],
        graph_signiture: ExportGraphSignature,
    ) -> None:
        super().__init__()
        self.rank = rank
        self.mesh = DeviceMesh("cuda", torch.arange(world_size))
        self.state_dict: Dict[str, torch.Tensor] = state_dict
        self.graph_signiture = graph_signiture

    def call(self, graph_module) -> PassResult:
        gm = copy.deepcopy(graph_module)

        placement_strategies = _mark_sharding(gm, self.graph_signiture, self.mesh)
        _partitioner(gm)
        _shard_state_dict(
            self.state_dict, placement_strategies, self.graph_signiture, self.mesh
        )
        return PassResult(gm, True)


def _mark_tensor_parallel_shardings(
    gm: GraphModule,
    graph_signiture: ExportGraphSignature,
    mesh: DeviceMesh,
) -> Dict[Node, PlacementStrategy]:
    """
    Mark the placement strategies of the parameter and buffer placeholder nodes.
    """
    placement_strategies: Dict[Node, PlacementStrategy] = {}
    num_params_and_buffers = len(graph_signiture.inputs_to_parameters) + len(
        graph_signiture.inputs_to_buffers
    )
    placeholder_idx: int = 0
    linear_node_names: Set[str] = set()
    last_linear_params: Set[Node] = set()
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            if placeholder_idx < num_params_and_buffers:
                # For a pair of linear layers, shard the first weight colwise and the second weight rowwise.
                linear_name, source_fn_class = node.meta["source_fn_stack"][0]
                if source_fn_class == torch.nn.Linear:
                    linear_node_names.add(linear_name)
                    placement: Placement
                    if len(linear_node_names) % 2 == 1:
                        placement = Shard(0)
                        if len(last_linear_params) == 2:
                            last_linear_params.clear()
                        last_linear_params.add(node)
                    else:
                        placement = (
                            Replicate()
                            if len(node.meta["tensor_meta"].shape) == 1
                            else Shard(1)
                        )

                    placement_strategies[node] = _create_placement_strategy(
                        node,
                        mesh,
                        placements=(placement,),
                    )
                placeholder_idx += 1
            else:
                placement_strategies[node] = _create_placement_strategy(
                    node,
                    mesh,
                    placements=(Replicate(),),
                )
    # If there are odd number of linear layers, no need to shard the params of the last layer.
    if len(linear_node_names) % 2 == 1:
        assert (
            len(last_linear_params) == 2
        ), f"There should be exactly two params for the last linear layer. but got {last_linear_params=}"
        for param_node in last_linear_params:
            placement_strategies[param_node].output_spec.placements = (Replicate(),)

    return placement_strategies


def _mark_sharding(
    gm: GraphModule, graph_signiture: ExportGraphSignature, mesh: DeviceMesh
) -> Dict[Node, PlacementStrategy]:
    """
    Mark the sharding strategy for each node in the graph module.
    """
    placement_strategies: Dict[
        Node, PlacementStrategy
    ] = _mark_tensor_parallel_shardings(gm, graph_signiture, mesh)

    for node in gm.graph.nodes:
        if node.op == "placeholder":
            if node not in placement_strategies:
                placement_strategies[node] = _create_placement_strategy(
                    node, mesh, placements=(Replicate(),)
                )
            node.meta["sharding"] = placement_strategies[node]
        elif node.op == "call_function":
            if node.target == operator.getitem:
                input_nodes = node.all_input_nodes
                assert (
                    len(input_nodes) == 1
                ), f"non-compute op only support one input now, found node: {node} with length of inputs: {len(node.args)}"
                arg_strategy = placement_strategies[input_nodes[0]]
                placement_strategies[node] = _create_placement_strategy(
                    node,
                    mesh,
                    placements=arg_strategy.output_spec.placements,
                    input_specs=_get_input_node_specs(node, placement_strategies),
                )
                node.meta["sharding"] = placement_strategies[node]
            else:
                op_schema = _get_op_schema(node, placement_strategies)

                # get DTensor specs for inputs and outputs
                output_sharding = DTensor._propagator.propagate_op_sharding(
                    op_schema,
                )

                placement_strategies[node] = PlacementStrategy(
                    output_spec=_get_output_spec_from_output_sharding(output_sharding),
                    input_specs=output_sharding.schema_suggestions[0].args_spec
                    if output_sharding.schema_suggestions is not None
                    else _get_input_node_specs(node, placement_strategies),
                )
                node.meta["sharding"] = placement_strategies[node]
        elif node.op == "output":
            node.meta["sharding"] = None
        else:
            raise RuntimeError(f"op code {node.op} not supported")
    return placement_strategies


def _get_output_spec_from_output_sharding(
    output_sharding: OutputSharding,
) -> DTensorSpec:
    """
    Util function to extract output spec from output sharding.
    """
    if isinstance(output_sharding.output_spec, DTensorSpec):
        return output_sharding.output_spec
    else:
        # For ops that return multiple outputs, the outputs should have the same output spec
        assert isinstance(output_sharding.output_spec, Sequence)
        output_spec_set = set(output_sharding.output_spec)
        assert len(output_spec_set) == 1
        assert output_sharding.output_spec[0] is not None
        return output_sharding.output_spec[0]


def _create_placement_strategy(
    node: Node,
    mesh: DeviceMesh,
    placements: Tuple[Placement, ...],
    input_specs: Optional[Sequence[DTensorSpec]] = None,
) -> PlacementStrategy:
    """
    Util function to construct a placement strategy for a given node.
    """
    placement = PlacementStrategy(
        input_specs=input_specs,
        output_spec=DTensorSpec(
            mesh=mesh,
            placements=placements,
        ),
    )
    _populate_tensor_meta(node, placement.output_spec)
    return placement


def _populate_tensor_meta(node: Node, output_spec: OutputSpecType) -> None:
    """
    Util function to populate tensor meta of output_spec based on node metadata.
    """
    if isinstance(node.meta["val"], Sequence):
        assert isinstance(output_spec, Sequence)
        for spec, fake_tensor in zip(output_spec, node.meta["val"]):
            assert spec is not None
            spec.tensor_meta = TensorMeta(
                shape=fake_tensor.shape,
                stride=fake_tensor.stride(),
                dtype=fake_tensor.dtype,
            )
    else:
        assert isinstance(output_spec, DTensorSpec)
        output_spec.tensor_meta = TensorMeta(
            shape=node.meta["val"].shape,
            stride=node.meta["val"].stride(),
            dtype=node.meta["val"].dtype,
        )


def _partitioner(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    """
    Graph partitioner that partitions the single device graph
    to distributed graph
    """
    for node in gm.graph.nodes:
        node_sharding = node.meta["sharding"]
        if node.op == "placeholder":
            out_spec = node_sharding.output_spec
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
                    _insert_reshard_gm(
                        gm, node, input_arg, input_arg_spec, desired_spec
                    )
            # convert output val to its local component
            output_val = node.meta["val"]
            node.meta["val"] = _partition_val(output_val, out_spec)
        elif node.op == "output":
            for input_arg in node.all_input_nodes:
                # input args of output should be Replicate, otherwise redistribution is needed.
                input_args_to_check: Sequence[Node] = (
                    input_arg if isinstance(input_arg, Sequence) else [input_arg]
                )
                for arg in input_args_to_check:
                    arg_sharding = arg.meta["sharding"]
                    arg_spec = arg_sharding.output_spec
                    desired_spec = copy.copy(arg_spec)
                    desired_spec.placements = (Replicate(),)
                    if arg_spec != desired_spec:
                        _insert_reshard_gm(gm, node, arg, arg_spec, desired_spec)
        else:
            raise RuntimeError(f"op code {node} not supported")

    _clean_up_graph_metadata(gm)
    gm.graph.lint()
    gm.recompile()
    return gm


def _partition_val(val: Any, spec: DTensorSpec) -> Any:
    """
    util function to convert a full tensor val to its local component
    """
    if isinstance(val, torch.Tensor):
        local_shard = val
        if val.ndim == 0:
            # If it's already a scalar tensor, it is already local, we don't
            # need to do anything
            return local_shard

        for idx, placement in enumerate(spec.placements):
            if placement.is_shard():
                placement = cast(Shard, placement)
                num_chunks = spec.mesh.size(dim=idx)
                my_coord = spec.mesh.get_coordinate()
                assert my_coord is not None, "current rank not in mesh!"
                my_coord_on_mesh_dim = my_coord[idx]
                local_shard = placement._split_tensor(
                    local_shard, num_chunks, with_padding=False, contiguous=True
                )[0][my_coord_on_mesh_dim]
        return local_shard
    elif isinstance(val, (list, tuple)):
        return val.__class__(_partition_val(v, spec) for v in val)
    else:
        raise RuntimeError(f"val type {type(val)} not supported")


def _insert_reshard_gm(
    gm: torch.fx.GraphModule,
    node: Node,
    input_arg: Node,
    input_arg_spec: DTensorSpec,
    desired_spec: DTensorSpec,
) -> None:
    """
    Transform the graph for tensor redistribution.
    """
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
    with gm.graph.inserting_before(node):
        output_node = gm.graph.graph_copy(
            reshard_gm.graph,
            val_map={
                input_node: input_arg,
            },
        )
    node.replace_input_with(input_arg, output_node)


def _clean_up_graph_metadata(gm: torch.fx.GraphModule) -> None:
    """
    Clean up the graph by removing sharding and partitioning related metadata
    """
    for node in gm.graph.nodes:
        if "sharding" in node.meta:
            del node.meta["sharding"]
        if "val" in node.meta and isinstance(node.meta["val"], torch.Tensor):
            local_tensor_meta = _extract_tensor_metadata(node.meta["val"])
            node.meta["tensor_meta"] = local_tensor_meta


def _get_input_node_specs(
    node: Node, placement_strategies: Dict[Node, PlacementStrategy]
) -> Tuple[DTensorSpec, ...]:
    """
    Get the input specs of a node.
    """
    input_specs_list: List[DTensorSpec] = []
    for input_arg in node.all_input_nodes:
        if input_arg in placement_strategies:
            input_specs_list.append(placement_strategies[input_arg].output_spec)
        else:
            raise ValueError(f"{input_arg} does not have output_spec populated.")
    return tuple(input_specs_list)


def _get_op_schema(
    node: Node, placement_strategies: Dict[Node, PlacementStrategy]
) -> OpSchema:
    """
    Util function to construct the operator schema of a node.
    """
    args_schema_list: List[object] = []
    for arg in node.args:
        if isinstance(arg, Node):
            assert (
                arg in placement_strategies
            ), f"{arg} does not have output_spec populated."
            args_schema_list.append(placement_strategies[arg].output_spec)
        else:
            # Appending the arg as it is for non-tensor arguments. E.g.: int/float/tuple
            args_schema_list.append(arg)

    op_schema = OpSchema(
        op=cast(torch._ops.OpOverload, node.target),
        args_schema=tuple(args_schema_list),
        kwargs_schema={},
    )
    return op_schema


def _shard_state_dict(
    state_dict: Dict[str, torch.Tensor],
    placement_strategies: Dict[Node, PlacementStrategy],
    graph_signiture: ExportGraphSignature,
    mesh: DeviceMesh,
) -> None:
    """
    Inplace partition the weights based on the placement strategy
    """
    for node, placement_strategy in placement_strategies.items():
        if node.op != "placeholder":
            continue
        if node.name in graph_signiture.inputs_to_parameters:
            fqn = graph_signiture.inputs_to_parameters[node.name]
        elif node.name in graph_signiture.inputs_to_buffers:
            fqn = graph_signiture.inputs_to_buffers[node.name]
        else:
            continue
        assert fqn in state_dict, f"{fqn} not found in state dict: {state_dict.keys()}"

        original_param = state_dict[fqn]
        dtensor_param = distribute_tensor(
            original_param,
            mesh,
            placement_strategy.output_spec.placements,
        )
        local_param = dtensor_param.to_local()
        state_dict[fqn] = (
            torch.nn.Parameter(local_param)
            if isinstance(original_param, torch.nn.Parameter)
            else local_param
        )
