from typing import Callable, Dict, Optional, Tuple

import torch
import torch.distributed._tensor.api as dtensor
from torch._ops import OpOverload
from torch._subclasses import FakeTensorMode
from torch.distributed._tensor.device_mesh import DeviceMesh
from torch.distributed._tensor.op_schema import (
    DTensorSpec,
    OpSchema,
    OpStrategy,
    OutputSharding,
    OutputSpecType,
    PlacementStrategy,
    StrategyType,
)
from torch.fx import Node
from torch.fx.experimental.proxy_tensor import get_isolated_graphmodule
from torch.utils._pytree import tree_flatten, tree_map

"""
Print information on ops input shape and sharding for debugging purposes.
"""
_DEBUG_VERBOSE = False


def unwrap_schema(e: object) -> object:
    return e._spec if isinstance(e, dtensor.DTensor) else e


class ShardingPropagator:
    def __init__(self) -> None:
        self.op_to_rules: Dict[OpOverload, Callable[[OpSchema], OutputSharding]] = {}
        self.op_strategy_funcs: Dict[
            OpOverload,
            Callable[[Node, DeviceMesh, Dict[Node, StrategyType]], StrategyType],
        ] = {}

    def register_sharding_prop_rule(
        self, op_overload: OpOverload, rule_func: Callable[[OpSchema], OutputSharding]
    ):
        """
        Register a sharding propagation rule for an operator.
        """
        self.op_to_rules[op_overload] = rule_func

    def register_op_strategy(
        self,
        op_overload: OpOverload,
        rule_func: Callable[[Node, DeviceMesh, Dict[Node, StrategyType]], StrategyType],
    ):
        """
        Register a sharding strategy generator for an operator.
        """
        self.op_strategy_funcs[op_overload] = rule_func

    def prepare_op_schema(
        self, op_call: OpOverload, args: Tuple[object, ...], kwargs: Dict[str, object]
    ) -> OpSchema:
        """
        This unwrap the args/kwargs DTensor to DTensorSpec and pack them
        into an OpSchema for sharding propagation usage.
        """
        args_schema = tree_map(unwrap_schema, args)
        kwargs_schema = tree_map(unwrap_schema, kwargs)

        op_schema = OpSchema(op_call._schema, args_schema, kwargs_schema)

        if _DEBUG_VERBOSE and torch.distributed.get_rank() == 0:
            print(f"OpSchema({op_schema})")
            local_shapes = tree_map(
                lambda t: t.to_local().shape
                if isinstance(t, dtensor.DTensor)
                else None,
                args,
            )
            print(f"    local shapes: {local_shapes}")

        return op_schema

    def propagate(self, op_overload: OpOverload, op_schema: OpSchema) -> OutputSharding:
        if op_overload in self.op_strategy_funcs:
            # generate op strategy for the op, this is done by propagating
            # the sharding in the graph.
            op_gm = self._prepare_op_graph(op_overload, op_schema)
            if op_gm is None:
                return OutputSharding(None, [op_schema])

            flat_args_sharding, _ = tree_flatten(
                [op_schema.args_schema, op_schema.kwargs_schema]
            )
            node_to_strategy: Dict[Node, StrategyType] = {}
            output_node = None
            out_node_strategy = None
            mesh = flat_args_sharding[0].mesh
            placeholder_idx = 0
            for node in op_gm.graph.nodes:
                if node.op == "placeholder":
                    # set sharding to placeholders if it's Node
                    if isinstance(flat_args_sharding[placeholder_idx], DTensorSpec):
                        strategy = PlacementStrategy(
                            flat_args_sharding[placeholder_idx]
                        )
                        # for eager execution, inputs only have one fixed sharding
                        node_to_strategy[node] = OpStrategy([strategy])
                    placeholder_idx += 1
                elif node.op == "call_function":
                    if isinstance(node.target, OpOverload):
                        op_strategy_func = self.op_strategy_funcs[op_overload]
                        out_strategies = op_strategy_func(node, mesh, node_to_strategy)
                        node_to_strategy[node] = out_strategies
                    else:
                        raise NotImplementedError(
                            f"Unsupported function: {node.target}"
                        )
                elif node.op == "output":
                    output_node = node.args[0]
                    out_node_strategy = node_to_strategy[output_node[0]]
                else:
                    raise NotImplementedError(f"Unsupported node type: {node.op}")

            # NOTE: This had the assumption we only have one call_function op in the
            # op graph, we need to harden this logic when there're decomposed ops.
            assert isinstance(out_node_strategy, OpStrategy)
            # we take the first strategy for now
            # TODO: add a min cost selection logic
            output_strategy = out_node_strategy.strategies[0]
            needs_redistribute = False
            expected_input_specs = []
            for idx, input_spec in enumerate(op_schema.args_spec):
                desired_spec = (
                    output_strategy.output_spec
                    if output_strategy.input_specs is None
                    else output_strategy.input_specs[idx]
                )
                expected_input_specs.append(desired_spec)
                if input_spec != desired_spec:
                    needs_redistribute = True

            if needs_redistribute:
                suggestion_schema = OpSchema(
                    op_schema.func_schema, tuple(expected_input_specs), {}
                )
                suggestion_schema._inplace_rewrap_schema_suggestion(op_schema)
            else:
                suggestion_schema = op_schema

            output_sharding = OutputSharding(
                output_strategy.output_spec,
                [suggestion_schema],
            )
            if output_node is not None:
                self._wrap_output_spec_meta(output_sharding.output_spec, output_node)
            return output_sharding

        elif op_overload in self.op_to_rules:
            return self.propagate_op_sharding(op_overload, op_schema)
        else:
            raise NotImplementedError(
                f"Operator {op_overload} does not have a sharding strategy registered."
            )

    def _wrap_output_spec_meta(
        self, output_spec: OutputSpecType, output_nodes: Node
    ) -> None:
        """
        Wrap the output_spec with the metadata from the output node.
        """
        if output_spec is not None:
            assert isinstance(output_nodes, (tuple, list))
            if isinstance(output_spec, DTensorSpec):
                output_spec.tensor_meta = output_nodes[0].meta["tensor_meta"]
            elif isinstance(output_spec, (tuple, list)):
                for i, spec in enumerate(output_spec):
                    if isinstance(spec, DTensorSpec):
                        spec.tensor_meta = output_nodes[i].meta["tensor_meta"]

    def propagate_op_sharding(
        self, op_overload: OpOverload, op_schema: OpSchema
    ) -> OutputSharding:
        """
        Propagate the sharding for an operator given the op_schema.
        """
        # first we propagate the tensor metadata
        output_node = None
        op_gm = self._prepare_op_graph(op_overload, op_schema)
        if op_gm is not None:
            for node in op_gm.graph.nodes:
                if node.op == "output":
                    output_node = node.args[0]

        # then we propagate the sharding
        sharding_prop_func = self.op_to_rules[op_overload]

        # step 1. there's sharding propagation rule, run
        # sharding propagation to get the output sharding
        try:
            output_sharding = sharding_prop_func(op_schema)
        except NotImplementedError as e:
            raise e
        except Exception as e:
            raise RuntimeError(
                f"Sharding propagation failed on op {op_overload}.\n"
                f"Input schema: {op_schema}.\n"
                f"Error: {e}"
            ) from e

        # step 2. if can't get output_spec from sharding
        # propagation (i.e. no rules apply for input
        # placements), we return the output sharding
        # with schema suggestions, which can be used to
        # decide how to do redistribute on inputs
        if output_sharding.output_spec is None:
            if output_sharding.schema_suggestions is None:
                if output_sharding.failed_reason is not None:
                    raise RuntimeError(
                        f"Sharding propagation failed on op {op_overload}!"
                        f"Input schema: {op_schema}."
                        f"Failed reason: {output_sharding.failed_reason}"
                    )
                else:
                    # if both output spec and schema suggestions are None, it
                    # means the operator return a non-tensor (scalar) value,
                    # in this case we just return the suggestion with the original
                    # input schema
                    output_sharding.schema_suggestions = [op_schema]
            else:
                # we do auto redistribute on inputs if necessary
                # to get an eligible input, which we will pick a
                # schema suggestion base on the redistribute cost.
                # For now we simply pick the first suggestion.
                suggested_input_schema = output_sharding.schema_suggestions[0]
                # run sharding propagation again with suggested schema
                propagation_res = sharding_prop_func(suggested_input_schema)
                # we set the output sharding with the new propagation result
                # so that dispatching know both output_spec and schema_suggestions
                # exist, which indicates a reshard is needed
                output_sharding.output_spec = propagation_res.output_spec
        else:
            # if sharding propagation succeed, we set the schema suggestion to
            # the default op_schema, which indicates no reshard is needed
            output_sharding.schema_suggestions = [op_schema]

        # associate the output sharding with the output metadata
        if output_node is not None:
            self._wrap_output_spec_meta(output_sharding.output_spec, output_node)

        return output_sharding

    def _prepare_op_graph(
        self,
        op_overload: OpOverload,
        op_schema: OpSchema,
    ) -> Optional[torch.fx.GraphModule]:
        # prepare the op graph for sharding propagation
        # special case op list, we don't need to propagate for local
        # scalar. TODO: figure out a better way to handle this
        skip_prop_list = [
            torch.ops.aten._local_scalar_dense.default,
            torch.ops.aten.equal.default,
            torch.ops.aten.is_same_size.default,
        ]
        if op_overload in skip_prop_list:
            return None

        # NOTE: We must call the tracing in fake tensor mode so that it
        # avoids materializing memory
        with FakeTensorMode():
            fake_args = op_schema.gen_fake_args()
            fake_kwargs = op_schema.gen_fake_kwargs()
            g = get_isolated_graphmodule(op_overload, fake_args, fake_kwargs)

        return g


class _CachingPropagator(ShardingPropagator):
    """
    A sharding propagator that caches the propagation results.
    This is currently experimental for Tensor Parallel usage.
    """

    def __init__(self, propagator: ShardingPropagator) -> None:
        super().__init__()
        self.op_to_rules = propagator.op_to_rules
        self.op_strategy_funcs = propagator.op_strategy_funcs

        # cache table for sharding propagation results, we might need to
        # limit the size of the cache table in the future
        self.cached_prop_results: Dict[OpSchema, OutputSharding] = {}

    def propagate(self, op_overload: OpOverload, op_schema: OpSchema) -> OutputSharding:
        """
        Propagate the sharding for an operator given the op_schema.
        Cache the propagation results to avoid running propagation again.
        """
        if op_schema in self.cached_prop_results:
            return self.cached_prop_results[op_schema]
        else:
            # call DTensor's propagate_op_sharding to get the prop result
            output_sharding = super().propagate(op_overload, op_schema)
            # update cached table
            self.cached_prop_results[op_schema] = output_sharding
            return output_sharding
