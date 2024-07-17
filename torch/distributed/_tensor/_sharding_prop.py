# mypy: allow-untyped-defs
from functools import lru_cache
from itertools import chain, product
from typing import Callable, cast, Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch._ops import OpOverload
from torch._subclasses import FakeTensorMode
from torch.distributed._tensor._op_schema import (
    OpInfo,
    OpSchema,
    OpStrategy,
    OutputSharding,
    OutputSpecType,
    PlacementList,
    PlacementStrategy,
    RuntimeSchemaInfo,
    StrategyType,
    TupleStrategy,
)
from torch.distributed._tensor._utils import (
    compute_local_shape,
    compute_local_stride,
    try_find_mesh_from_args,
)
from torch.distributed._tensor.placement_types import (
    DTensorSpec,
    Replicate,
    Shard,
    TensorMeta,
)
from torch.distributed.device_mesh import DeviceMesh


aten = torch.ops.aten


def _length(obj) -> int:
    if obj is None:
        return 0
    if not isinstance(obj, Sequence):
        return 1
    return len(obj)


class ShardingPropagator:
    def __init__(self) -> None:
        self.op_to_rules: Dict[OpOverload, Callable[[OpSchema], OutputSharding]] = {}
        self.op_strategy_funcs: Dict[
            OpOverload,
            Callable[[DeviceMesh, OpSchema], StrategyType],
        ] = {}
        # op map to save static argnum to decide to reuse sharding prop cache or re-run sharding prop
        self.op_to_schema_info: Dict[OpOverload, RuntimeSchemaInfo] = {}
        self.propagate_op_sharding = lru_cache(None)(self.propagate_op_sharding_non_cached)  # type: ignore[method-assign]
        # op map to save indices of shape (and stride) args which may need to be modified in sharding prop
        self.op_to_shape_and_stride_idx: Dict[
            OpOverload, Union[int, Tuple[int, int]]
        ] = {
            # new factory ops
            aten.new_empty.default: 1,
            aten.new_full.default: 1,
            aten.new_ones.default: 1,
            aten.new_zeros.default: 1,
            aten.new_empty_strided.default: (1, 2),
            # view ops
            aten.expand.default: 1,
            aten.reshape.default: 1,
            aten.view.default: 1,
            aten._unsafe_view.default: 1,
        }
        # op map to save decomposition tables so that the op strategy can be generated from decomposition
        self.op_to_decompositions: Dict[OpOverload, Dict[OpOverload, Callable]] = {}

    def register_sharding_prop_rule(
        self,
        op_overload: OpOverload,
        rule_func: Callable[[OpSchema], OutputSharding],
        schema_info: Optional[RuntimeSchemaInfo] = None,
    ):
        """
        Register a sharding propagation rule for an operator.
        """
        self.op_to_rules[op_overload] = rule_func
        if schema_info is not None:
            self.op_to_schema_info[op_overload] = schema_info

    def register_op_strategy(
        self,
        op_overload: OpOverload,
        strategy_func: Callable[[DeviceMesh, OpSchema], StrategyType],
        schema_info: Optional[RuntimeSchemaInfo] = None,
    ):
        """
        Register a sharding strategy generator for an operator.
        """
        self.op_strategy_funcs[op_overload] = strategy_func
        if schema_info is not None:
            self.op_to_schema_info[op_overload] = schema_info

    def register_op_decomposition(
        self,
        op_overload: OpOverload,
        decomposition_table: Dict[OpOverload, Callable],
        schema_info: Optional[RuntimeSchemaInfo] = None,
    ) -> None:
        """
        Register a decomposition table for an operator.
        """
        self.op_to_decompositions[op_overload] = decomposition_table
        if schema_info is not None:
            self.op_to_schema_info[op_overload] = schema_info

    @lru_cache  # noqa: B019
    def _propagate_tensor_meta(
        self, op_schema: OpSchema
    ) -> Union[None, TensorMeta, Sequence[Optional[TensorMeta]]]:
        """
        Propagate the tensor metadata, it could either return a TensorMeta
        or a list/tuple of TensorMetas
        """
        if op_schema.op == aten.equal.default:
            # data dependent ops can't be used for fake propagation
            return None

        # NOTE: We must call the tracing in fake tensor mode so that it
        # avoids materializing memory
        with FakeTensorMode():
            fake_args = op_schema.gen_fake_args()
            fake_kwargs = op_schema.gen_fake_kwargs()
            fake_out = op_schema.op(*fake_args, **fake_kwargs)

        if isinstance(fake_out, torch.Tensor):
            return TensorMeta(
                shape=fake_out.shape, stride=fake_out.stride(), dtype=fake_out.dtype
            )

        elif isinstance(fake_out, (tuple, list)):
            tensor_meta_list: List[Optional[TensorMeta]] = []
            for fake_out_item in fake_out:
                if isinstance(fake_out_item, torch.Tensor):
                    tensor_meta_list.append(
                        TensorMeta(
                            shape=fake_out_item.shape,
                            stride=fake_out_item.stride(),
                            dtype=fake_out_item.dtype,
                        )
                    )
                else:
                    tensor_meta_list.append(None)
            return (
                tuple(tensor_meta_list)
                if isinstance(fake_out, tuple)
                else tensor_meta_list
            )
        else:
            # if fake is not a tensor or tuple of tensor, return as none
            return None

    def _wrap_output_spec_tensor_meta(
        self,
        op: OpOverload,
        output_specs: OutputSpecType,
        output_tensor_meta: Union[None, TensorMeta, Sequence[Optional[TensorMeta]]],
    ) -> None:
        """
        Wrap the output_specs with the tensor metadata from the output.
        """

        if isinstance(output_specs, DTensorSpec):
            if not isinstance(output_tensor_meta, TensorMeta):
                # Either error due to ShardingPropagator or due to incorrect OutputSpec
                if not isinstance(output_tensor_meta, (tuple, list)):
                    raise ValueError(
                        "ShardingPropagator error: output does not have an associated TensorMeta"
                    )
                raise ValueError(
                    f"For the op {op.name()}, `output_specs` has 1 output which does not equal the "
                    f"number of op outputs: {len(output_tensor_meta)}."
                )
            output_specs.tensor_meta = output_tensor_meta
        elif isinstance(output_specs, (tuple, list)):
            if not isinstance(output_tensor_meta, (tuple, list)) or len(
                output_specs
            ) != len(output_tensor_meta):
                raise ValueError(
                    f"For the op {op.name()}, `output_specs` has {len(output_specs)} outputs which does not equal the "
                    f"number of op outputs {_length(output_tensor_meta)}."
                )
            for i, spec in enumerate(output_specs):
                if isinstance(spec, DTensorSpec):
                    output_tensor_meta_i = output_tensor_meta[i]
                    if not isinstance(output_tensor_meta_i, TensorMeta):
                        raise ValueError(
                            f"ShardingPropagator error: output {i} does not have an associated TensorMeta"
                        )
                    spec.tensor_meta = output_tensor_meta_i

    def propagate(self, op_info: OpInfo) -> None:
        # We cannot use an lru cache if we know that inputs will have dynamic shapes,
        # because SymInts are not hashable.
        # This is generally ok because this only happens during tracing in torch.compile,
        # and tracing does not need to be as fast as eagermode DTensor usages.
        if op_info.schema.has_symints:
            output_sharding = self.propagate_op_sharding_non_cached(op_info.schema)
        else:
            output_sharding = self.propagate_op_sharding(op_info.schema)
        op_info.output_sharding = output_sharding

    def propagate_op_sharding_non_cached(self, op_schema: OpSchema) -> OutputSharding:
        """
        Propagate the sharding for an operator given the op_schema.
        """
        # special case op, we don't need to propagate for local
        # scalar. TODO: figure out a better way to handle this
        if op_schema.op is aten._local_scalar_dense.default:
            return OutputSharding(None, op_schema)

        out_tensor_meta = self._propagate_tensor_meta(op_schema)

        if (
            op_schema.op in self.op_strategy_funcs
            or op_schema.op in self.op_to_decompositions
        ):
            # generate op strategy for the op
            mesh = try_find_mesh_from_args(op_schema.op, op_schema.args_schema)

            if op_schema.op in self.op_strategy_funcs:
                strategy_schema = self._spec_schema_to_strategy_schema(op_schema)
                op_strategy = self.op_strategy_funcs[op_schema.op](
                    mesh, strategy_schema
                )
            else:
                decomposition_table = self.op_to_decompositions[op_schema.op]
                op_strategy = self._generate_op_strategy_from_decomposition(
                    mesh, op_schema, decomposition_table
                )

            if isinstance(op_strategy, OpStrategy):
                # single op strategy
                output_strategy = self._select_strategy(op_strategy)

                # check if we need to redistribute the input
                needs_redistribute = False
                expected_input_specs = []

                # in case where the op does not specify input_specs and output_specs
                # is a DTensorSpec, we use output_specs as the spec for each DTensor
                # input arg.
                if output_strategy.input_specs is None:
                    assert isinstance(output_strategy.output_specs, DTensorSpec)

                for idx, input_spec in enumerate(op_schema.args_spec):
                    desired_spec = (
                        output_strategy.output_spec
                        if output_strategy.input_specs is None
                        else output_strategy.input_specs[idx]
                    )
                    expected_input_specs.append(
                        desired_spec.shallow_copy_with_tensor_meta(
                            input_spec.tensor_meta
                        )
                    )
                    if input_spec.placements != desired_spec.placements:
                        needs_redistribute = True

                suggestion_schema = None
                if needs_redistribute:
                    suggestion_schema = OpSchema(
                        op_schema.op, tuple(expected_input_specs), {}
                    )
                    suggestion_schema._inplace_rewrap_schema_suggestion(op_schema)

                # shape and stride args need to be modified for
                # view ops and new factory ops, potentially
                if op_schema.op in self.op_to_shape_and_stride_idx:
                    assert isinstance(output_strategy.output_spec, DTensorSpec)
                    # It happens when the output has the same shape as the input
                    # and the input placements are not all Replicate().
                    if output_strategy.output_spec.is_sharded():
                        schema = suggestion_schema or op_schema
                        assert isinstance(out_tensor_meta, TensorMeta)
                        suggestion_schema = self._adjust_shape_and_stride_args(
                            out_tensor_meta, schema, output_strategy.output_spec, mesh
                        )
                        needs_redistribute = True

                # construct output spec for the op
                if op_schema.return_type_tuple_tensor_like():
                    # for ops that return multiple tensors and the output_specs is not
                    # a tuple, we use a tuple of that single output spec as the new
                    # output_specs
                    output_specs: OutputSpecType = output_strategy.output_specs
                    if isinstance(output_specs, DTensorSpec):
                        output_specs = tuple(
                            [
                                # create a new DTensorSpec with the same placement as the
                                # output_specs in output_strategy
                                DTensorSpec(
                                    mesh=output_specs.mesh,
                                    placements=output_specs.placements,
                                    tensor_meta=output_specs.tensor_meta,
                                )
                                for _ in range(len(op_schema.op._schema.returns))
                            ]
                        )
                elif op_schema.return_type_tensor():
                    output_specs = output_strategy.output_specs
                else:
                    output_specs = None

                output_sharding = OutputSharding(
                    output_specs,
                    suggestion_schema,
                    needs_redistribute=needs_redistribute,
                )
            elif isinstance(op_strategy, TupleStrategy):
                # tuple strategy output sharding processing
                # runtime selected placement strategy for each TupleStrategy input arg
                selected_strategies: List[PlacementStrategy] = []
                out_spec_list: List[DTensorSpec] = []
                for strategy in op_strategy.childs:
                    assert isinstance(strategy, OpStrategy)
                    selected_strategy = self._select_strategy(strategy)
                    selected_strategies.append(selected_strategy)
                    out_spec_list.append(selected_strategy.output_spec)

                needs_redistribute = False
                suggestion_args: List[object] = []
                tensor_or_list_tensor_arg_idx = 0

                for arg in op_schema.args_schema:
                    if (
                        arg
                        and isinstance(arg, (list, tuple))
                        and isinstance(arg[0], DTensorSpec)
                    ):
                        expected_input_spec_list: List[DTensorSpec] = []
                        for idx, arg_spec in enumerate(arg):
                            expected_input_spec = selected_strategies[idx].input_spec(
                                tensor_or_list_tensor_arg_idx
                            )
                            expected_input_spec = (
                                expected_input_spec.shallow_copy_with_tensor_meta(
                                    arg_spec.tensor_meta
                                )
                            )
                            if arg_spec.placements != expected_input_spec.placements:
                                needs_redistribute = True
                            expected_input_spec_list.append(expected_input_spec)
                        suggestion_args.append(
                            tuple(expected_input_spec_list)
                            if isinstance(arg, tuple)
                            else expected_input_spec_list
                        )
                        tensor_or_list_tensor_arg_idx += 1

                    elif isinstance(arg, DTensorSpec):
                        expected_input_spec = selected_strategies[0].input_spec(
                            tensor_or_list_tensor_arg_idx
                        )
                        expected_input_spec = (
                            expected_input_spec.shallow_copy_with_tensor_meta(
                                arg.tensor_meta
                            )
                        )
                        if arg.placements != expected_input_spec.placements:
                            needs_redistribute = True
                        suggestion_args.append(expected_input_spec)
                        tensor_or_list_tensor_arg_idx += 1
                    else:
                        suggestion_args.append(arg)

                suggestion_schema = None
                if needs_redistribute:
                    suggestion_schema = OpSchema(
                        op_schema.op, tuple(suggestion_args), op_schema.kwargs_schema
                    )

                output_sharding = OutputSharding(
                    tuple(out_spec_list) if out_tensor_meta is not None else None,
                    suggestion_schema,
                    needs_redistribute=needs_redistribute,
                )
            else:
                raise ValueError("Unsupported op strategy type")

            # associate the output sharding with the output tensor metadata
            self._wrap_output_spec_tensor_meta(
                op_schema.op, output_sharding.output_spec, out_tensor_meta
            )
            return output_sharding
        elif op_schema.op in self.op_to_rules:
            # propagate the sharding with rule
            sharding_prop_func = self.op_to_rules[op_schema.op]

            # step 1. there's sharding propagation rule, run
            # sharding propagation to get the output sharding
            try:
                output_sharding = sharding_prop_func(op_schema)
            except NotImplementedError as e:
                raise e
            except Exception as e:
                raise RuntimeError(
                    f"Sharding propagation failed on op {op_schema}.\n" f"Error: {e}"
                ) from e

            # step 2. if can't get output_spec from sharding
            # propagation (i.e. no rules apply for input
            # placements), we return the output sharding
            # with schema suggestions, which can be used to
            # decide how to do redistribute on inputs
            if output_sharding.output_spec is None:
                if output_sharding.redistribute_schema is None:
                    raise RuntimeError(
                        f"Sharding propagation failed on op {op_schema}!"
                    )
                else:
                    # we do auto redistribute on inputs if necessary
                    # run sharding propagation again with suggested schema
                    propagation_res = sharding_prop_func(
                        output_sharding.redistribute_schema
                    )
                    # we set the output sharding with the new propagation result
                    # so that dispatching know both output_spec and redistribute_schema
                    # exist, which indicates a reshard is needed
                    output_sharding.output_spec = propagation_res.output_spec
                    output_sharding.needs_redistribute = True

            # associate the output sharding with the output tensor metadata
            self._wrap_output_spec_tensor_meta(
                op_schema.op, output_sharding.output_spec, out_tensor_meta
            )

            return output_sharding
        else:
            raise NotImplementedError(
                f"Operator {op_schema.op} does not have a sharding strategy registered."
            )

    def _spec_to_strategy(self, spec: object) -> object:
        if isinstance(spec, DTensorSpec):
            return OpStrategy([PlacementStrategy(spec)])
        elif (
            isinstance(spec, (list, tuple))
            and len(spec) > 0
            and isinstance(spec[0], DTensorSpec)
        ):
            # tensor list create tuple strategy
            tuple_strategy = [self._spec_to_strategy(s) for s in spec]
            tuple_strategy = cast(Sequence[StrategyType], tuple_strategy)
            return TupleStrategy(
                tuple(tuple_strategy) if isinstance(spec, tuple) else tuple_strategy
            )
        else:
            return spec

    def _spec_schema_to_strategy_schema(self, op_schema: OpSchema) -> OpSchema:
        # swap the args spec with args strategies
        args_op_strategy = [self._spec_to_strategy(i) for i in op_schema.args_schema]
        kwargs_op_strategy = {
            k: self._spec_to_strategy(v) for k, v in op_schema.kwargs_schema.items()
        }

        # construct a new OpSchema on args for strategy based propagation
        return OpSchema(
            op=op_schema.op,
            args_schema=tuple(args_op_strategy),
            kwargs_schema=kwargs_op_strategy,
        )

    def _select_strategy(self, strategy: OpStrategy) -> PlacementStrategy:
        if len(strategy.strategies) == 1:
            # short cut with only one possible strategy
            return strategy.strategies[0]

        strategy_costs: List[float] = []
        for strtg in strategy.strategies:
            assert (
                strtg.redistribute_cost is not None
            ), "must set redistribute cost for each strategy!"
            redistribute_cost = sum(chain.from_iterable(strtg.redistribute_cost))
            strategy_costs.append(redistribute_cost)

        # for eager execution, we just select the one with the minimal redistribute cost
        return strategy.strategies[strategy_costs.index(min(strategy_costs))]

    def _adjust_shape_and_stride_args(
        self,
        out_tensor_meta: TensorMeta,
        schema: OpSchema,
        spec: DTensorSpec,
        mesh: DeviceMesh,
    ) -> OpSchema:
        shape_stride_idx = self.op_to_shape_and_stride_idx[schema.op]
        if isinstance(shape_stride_idx, tuple):
            shape_idx, stride_idx = shape_stride_idx
        else:
            shape_idx = shape_stride_idx
            stride_idx = None

        expected_input_schema = list(schema.args_schema)
        # adjust shape to be the same as that of the _local_tensor
        # of the DTensor input arg at index 0, which is inferred
        expected_input_schema[shape_idx] = compute_local_shape(
            out_tensor_meta.shape, mesh, spec.placements
        )

        # adjust the stride arg for aten.new_empty_strided.default
        if stride_idx:
            expected_input_schema[stride_idx] = compute_local_stride(
                out_tensor_meta.stride, mesh, spec.placements
            )

        return OpSchema(schema.op, tuple(expected_input_schema), schema.kwargs_schema)

    def _prepare_op_graph(
        self,
        op_schema: OpSchema,
        decomposition_table: Dict[OpOverload, Callable],
    ) -> Optional[torch.fx.GraphModule]:
        from torch.fx.experimental.proxy_tensor import get_isolated_graphmodule

        # call tracing in fake tensor mode to avoid materializing memory
        with FakeTensorMode():
            fake_args = op_schema.gen_fake_args()
            fake_kwargs = op_schema.gen_fake_kwargs()
            gm = get_isolated_graphmodule(
                op_schema.op,
                fake_args,
                fake_kwargs,
                decomposition_table=decomposition_table,
            )

        return gm

    def _propagate_sharding_through_graph(self, gm, mesh, input_specs):
        # 1. for each call_function, generate an OpSchema and run sharding prop
        # 2. filter out those which needs_redistribute
        node_to_spec = {}
        placeholder_idx = 0
        for node in gm.graph.nodes:
            if node.op == "placeholder":
                node_to_spec[node] = input_specs[placeholder_idx]
                placeholder_idx += 1
            elif node.op == "call_function":
                if (
                    not isinstance(node.target, OpOverload)
                    or node.target not in self.op_strategy_funcs
                ):
                    raise NotImplementedError(
                        "Generating sharding strategy via decomposition failed "
                        f"because {node.target} does not have a sharding strategy registered."
                    )

                # generate node output spec via sharding propagation
                node_input_arg_strategies = tuple(
                    self._spec_to_strategy(node_to_spec.get(arg, arg))
                    for arg in node.args
                )
                node_input_kwarg_strategies = {
                    k: self._spec_to_strategy(node_to_spec.get(v, v))
                    for k, v in node.kwargs.items()
                }
                node_input_strategy_schema: OpSchema = OpSchema(
                    op=node.target,
                    args_schema=node_input_arg_strategies,
                    kwargs_schema=node_input_kwarg_strategies,
                )

                node_output_strategy = self.op_strategy_funcs[node.target](
                    mesh, node_input_strategy_schema
                )
                assert isinstance(
                    node_output_strategy, OpStrategy
                ), "TupleStrategy is not supported in decomposed sharding propagation"

                # select the first PlacementStrategy with needs_redistribute=False
                # NOTE: there should be only one such PlacementStrategy; revisit if we find exceptions
                node_output_spec = None
                needs_redistribute = False
                for strtg in node_output_strategy.strategies:
                    if strtg.input_specs is None:
                        assert isinstance(strtg.output_specs, DTensorSpec)
                    for idx, input_strtg in enumerate(
                        node_input_strategy_schema.args_strategy
                    ):
                        desired_spec = (
                            strtg.output_spec
                            if strtg.input_specs is None
                            else strtg.input_specs[idx]
                        )
                        if (
                            input_strtg.strategies[0].output_spec.placements
                            != desired_spec.placements
                        ):
                            needs_redistribute = True
                            break
                    if not needs_redistribute:
                        node_output_spec = strtg.output_spec
                        break

                if node_output_spec is None:
                    return None

                node_output_tensor_meta = TensorMeta(
                    shape=node.meta["tensor_meta"].shape,
                    stride=node.meta["tensor_meta"].stride,
                    dtype=node.meta["tensor_meta"].dtype,
                )
                self._wrap_output_spec_tensor_meta(
                    node.target, node_output_spec, node_output_tensor_meta
                )

                node_to_spec[node] = node_output_spec
            elif node.op == "output":
                output_node = node.args[0]
                graph_output_specs = [node_to_spec[node] for node in output_node]
                return graph_output_specs
            else:
                raise NotImplementedError(f"Unsupported node type: {node.op}")

    def _generate_op_strategy_from_decomposition(
        self, mesh, op_schema, decomposition_table
    ):
        # TODO(lty): expand_to_full_mesh_op_strategy hit circular import if put outside
        from torch.distributed._tensor.ops.utils import expand_to_full_mesh_op_strategy
        from torch.utils._pytree import tree_flatten

        # generate all possible placements for each DTensor input
        all_possible_schema = []
        flat_args_schema, _ = tree_flatten(
            [op_schema.args_schema, op_schema.kwargs_schema]
        )
        for arg_spec in flat_args_schema:
            if isinstance(arg_spec, DTensorSpec):
                possible_placements = [Replicate()] + [
                    Shard(i) for i in range(arg_spec.ndim)
                ]
                possible_arg_specs = tuple(
                    DTensorSpec(mesh, (p,), arg_spec.tensor_meta)
                    for p in possible_placements
                )
                all_possible_schema.append(possible_arg_specs)
            else:
                all_possible_schema.append((arg_spec,))

        op_gm = self._prepare_op_graph(op_schema, decomposition_table)

        single_mesh_dim_strategies: List[PlacementList] = []
        # for each possible input placement combination, run sharding propagation through the graph
        for graph_input_specs in product(*all_possible_schema):
            graph_output_specs = self._propagate_sharding_through_graph(
                op_gm, mesh, graph_input_specs
            )
            if graph_output_specs is not None:
                input_placements: PlacementList = [
                    item.placements[0]
                    for item in graph_input_specs
                    if isinstance(item, DTensorSpec)
                ]
                output_placements: PlacementList = [
                    item.placements[0]
                    for item in graph_output_specs
                    if isinstance(item, DTensorSpec)
                ]
                output_input_placements: PlacementList = (
                    output_placements + input_placements
                )
                single_mesh_dim_strategies.append(output_input_placements)

        strategy_schema = self._spec_schema_to_strategy_schema(op_schema)

        assert (
            len(single_mesh_dim_strategies) > 0
        ), f"No valid strategy found for {op_schema.op} via decomposition!"
        return expand_to_full_mesh_op_strategy(
            mesh,
            strategy_schema,
            single_mesh_dim_strategies,
            input_index=len(op_schema.op._schema.returns),
        )
