from typing import Callable, Dict, Tuple

import torch
import torch.distributed._tensor.api as dtensor
from torch._ops import OpOverload
from torch.distributed._tensor.op_schema import OpSchema, OutputSharding
from torch.utils._pytree import tree_map

"""
Print information on ops input shape and sharding for debugging purposes.
"""
_DEBUG_VERBOSE = False


def unwrap_schema(e: object) -> object:
    return e._spec if isinstance(e, dtensor.DTensor) else e


class ShardingPropagator:
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
        self,
        op_call: OpOverload,
        args: Tuple[object, ...],
        kwargs: Dict[str, object]
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
                lambda t: t.to_local().shape if isinstance(t, dtensor.DTensor) else None,
                args,
            )
            print(f"    local shapes: {local_shapes}")

        return op_schema

    def propagate_op_sharding(
        self, op_overload: OpOverload, op_schema: OpSchema
    ) -> OutputSharding:
        """
        Propagate the sharding for an operator given the op_schema.
        """
        sharding_prop_func = self.op_to_rules.get(op_overload, None)

        if sharding_prop_func is None:
            # step 1. If there's not even one sharding rule
            # implemented for the operator, we error out.
            raise NotImplementedError(
                f"Operator {op_overload} does not have a DistributedTensor rule registered."
            )

        # step 2. there's sharding propagation rule, run
        # sharding propagation to get the output sharding
        try:
            output_sharding = sharding_prop_func(op_schema)
        except Exception as e:
            raise RuntimeError(
                f"Sharding propagation failed on op {op_overload}.\n"
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
                raise RuntimeError(
                    f"Sharding propagation failed on op {op_overload}!"
                    f"Input schema: {op_schema}."
                    f"Failed reason: {output_sharding.failed_reason}"
                )
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

        return output_sharding


class _CachingPropagator(ShardingPropagator):
    """
    A sharding propagator that caches the propagation results.
    This is currently experimental for Tensor Parallel usage.
    """

    def __init__(self, op_to_rules=None) -> None:
        super().__init__()
        if op_to_rules is not None:
            self.op_to_rules = op_to_rules

        # cache table for sharding propagation results, we might need to
        # limit the size of the cache table in the future
        self.cached_prop_results: Dict[OpSchema, OutputSharding] = {}

    def propagate_op_sharding(
        self, op_overload: OpOverload, op_schema: OpSchema
    ) -> OutputSharding:
        """
        Propagate the sharding for an operator given the op_schema.
        Cache the propagation results to avoid running propagation again.
        """
        if op_schema in self.cached_prop_results:
            return self.cached_prop_results[op_schema]
        else:
            # call DTensor's propagate_op_sharding to get the prop result
            output_sharding = super().propagate_op_sharding(op_overload, op_schema)
            # update cached table
            self.cached_prop_results[op_schema] = output_sharding
            return output_sharding
