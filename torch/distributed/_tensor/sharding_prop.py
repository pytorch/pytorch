import operator
from typing import Callable, Dict, Tuple, List, Optional, cast

import torch
import torch.distributed._tensor.api as dtensor
from torch._ops import OpOverload
from torch._subclasses import FakeTensorMode
from torch.distributed._tensor.op_schema import DTensorSpec, OpSchema, OutputSharding
from torch.fx import Node
from torch.fx.experimental.proxy_tensor import get_isolated_graphmodule
from torch.utils._pytree import tree_map_only, tree_flatten

"""
Print information on ops input shape and sharding for debugging purposes.
"""
_DEBUG_VERBOSE = False


def unwrap_spec(e: "dtensor.DTensor") -> DTensorSpec:
    return e._spec

def unwrap_spec_from_node(n: Node) -> DTensorSpec:
    spec = n.meta["sharding"]
    spec.tensor_meta = n.meta["tensor_meta"]
    return spec

class ShardingPropagator(object):
    def __init__(self) -> None:
        self.op_to_rules: Dict[OpOverload, Callable[[OpSchema], OutputSharding]] = {}

    def register_sharding_prop_rule(
        self, op_overload: OpOverload, rule_func: Callable[[OpSchema], OutputSharding]
    ):
        """
        Register a sharding propagation rule for an operator.
        """
        self.op_to_rules[op_overload] = rule_func

    def prepare_op_schema(
        self, op_call: OpOverload, args: Tuple[object, ...], kwargs: Dict[str, object]
    ) -> OpSchema:
        """
        This unwrap the args/kwargs DTensor to DTensorSpec and pack them
        into an OpSchema for sharding propagation usage.
        """
        args_schema = tree_map_only(dtensor.DTensor, unwrap_spec, args)
        kwargs_schema = tree_map_only(dtensor.DTensor, unwrap_spec, kwargs)

        op_schema = OpSchema(op_call._schema, args_schema, kwargs_schema)

        return op_schema

    def propagate(self, op_call: OpOverload, op_schema: OpSchema) -> OutputSharding:
        """
        Propagate the sharding for an operator given the args/kwargs.
        """
        # special case op list, we don't need to propagate for local
        # scalar. TODO: figure out a better way to handle this
        skip_prop_list = [
            torch.ops.aten._local_scalar_dense.default,
            torch.ops.aten.equal.default,
        ]
        if op_call in skip_prop_list:
            return OutputSharding(None, schema_suggestions=[op_schema])

        args_schema = op_schema.args_schema
        kwargs_schema = op_schema.kwargs_schema

        # prepare a fake graph to run the propagation
        with FakeTensorMode():
            fake_args = op_schema.gen_fake_args()
            fake_kwargs = op_schema.gen_fake_kwargs()
            op_graph = get_isolated_graphmodule(op_call, fake_args, fake_kwargs)

        if _DEBUG_VERBOSE and torch.distributed.get_rank() == 0:
            print(f"OpSchema({op_schema.func_schema})")

        # flatten the args schema/kwarg schema to feed into the graph
        flat_args_sharding, _ = tree_flatten([args_schema, kwargs_schema])

        return self.run_graph(op_graph.graph, flat_args_sharding)


    def run_graph(self, op_graph: torch.fx.Graph, flat_args_sharding):
        """
        Run the sharding propagation on the op_graph.
        """
        # NOTE: we assume the first few nodes are all placeholders
        placeholder_idx = 0
        schema_suggestions = None
        for node in op_graph.nodes:
            if node.op == "placeholder":
                # set sharding to placeholders if it's Node
                if isinstance(flat_args_sharding[placeholder_idx], DTensorSpec):
                    node.meta["sharding"] = flat_args_sharding[placeholder_idx]
                placeholder_idx += 1

            elif node.op == "call_function":
                if node.target == operator.getitem:
                    list_arg = node.args[0]
                    item_idx = node.args[1]
                    node.meta["sharding"] = list_arg.meta["sharding"][item_idx]
                elif isinstance(node.target, OpOverload):
                    if schema_suggestions is None:
                        schema_suggestions = self.run_node(node)
                    else:
                        self.run_node(node)
                else:
                    raise ValueError(f"Can't propagate sharding on node target: {node.target}")
            elif node.op == "output":
                # get the sharding from the output node
                output_spec = tree_map_only(Node, unwrap_spec_from_node, node.args[0])
            else:
                raise ValueError(f"Can't propagate sharding on node type: {node.op}")

        return OutputSharding(output_spec, schema_suggestions)


    def run_node(self, op_node: Node) -> Optional[List[OpSchema]]:
        """
        Propagate the sharding for an operator given the op_schema.
        """
        op_call = cast(OpOverload, op_node.target)
        # then we propagate the sharding
        sharding_prop_func = self.op_to_rules.get(op_call, None)

        if sharding_prop_func is None:
            # step 1. If there's not even one sharding rule
            # implemented for the operator, we error out.
            raise NotImplementedError(
                f"Operator {op_call} does not have a DistributedTensor rule registered."
            )


        args_schema = tree_map_only(Node, unwrap_spec_from_node, op_node.args)
        kwargs_schema = tree_map_only(Node, unwrap_spec_from_node, op_node.kwargs)

        op_schema = OpSchema(op_call._schema, args_schema, kwargs_schema)

        # step 2. there's sharding propagation rule, run
        # sharding propagation to get the output sharding
        try:
            output_sharding = sharding_prop_func(op_schema)
        except Exception as e:
            raise RuntimeError(
                f"Sharding propagation failed on op {op_call}.\n"
                f"Input schema: {op_schema}.\n"
                f"Error: {e}"
            ) from e

        # step 3. if can't get output_spec from sharding
        # propagation (i.e. no rules apply for input
        # placements), we return the output sharding
        # with schema suggestions, which can be used to
        # decide how to do redistribute on inputs
        if output_sharding.output_spec is None:
            if output_sharding.schema_suggestions is None:
                if output_sharding.failed_reason is not None:
                    raise RuntimeError(
                        f"Sharding propagation failed on op {op_call}!"
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
                # to get an eligble input, which we will pick a
                # schema suggestion base on the redistribute cost.
                # For now we simply pick the first suggestion.
                # TODO: implement full auto distribute with a
                # simple cost estimation model
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

        # set the output sharding to the node
        op_node.meta["sharding"] = output_sharding.output_spec

        return output_sharding.schema_suggestions


class _CachingPropagator(ShardingPropagator):
    """
    A sharding propagator that caches the propagation results.
    This is currently experimental for Tensor Parallel usage.

    TODO: move this to C++ for efficient hashing
    """

    def __init__(self, op_to_rules=None) -> None:
        super().__init__()
        if op_to_rules is not None:
            self.op_to_rules = op_to_rules

        # cache table for sharding propagation results, we might need to
        # limit the size of the cache table in the future
        self.cached_prop_results: Dict[OpSchema, OutputSharding] = {}

    def propagate(
        self, op_call: OpOverload, op_schema: OpSchema
    ) -> OutputSharding:
        """
        Propagate the sharding for an operator given the op_schema.
        Cache the propagation results to avoid running propagation again.
        """
        if op_schema in self.cached_prop_results:
            return self.cached_prop_results[op_schema]
        else:
            # call DTensor's propagate to get the prop result
            output_sharding = super().propagate(op_call, op_schema)
            # update cached table
            self.cached_prop_results[op_schema] = output_sharding
            return output_sharding
