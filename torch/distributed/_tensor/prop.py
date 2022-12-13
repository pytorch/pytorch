from typing import Callable, Dict

import torch
from torch.distributed._tensor.op_schema import OpSchema, OutputSharding


class ShardingPropagator(object):
    def __init__(self) -> None:
        self.op_to_rules: Dict[str, Callable[[OpSchema], OutputSharding]] = {}

    def register_sharding_prop_rule(
        self, op_key: str, rule_func: Callable[[OpSchema], OutputSharding]
    ):
        """
        Register a sharding propagation rule for an operator.
        """
        self.op_to_rules[op_key] = rule_func

    def propagate_op_sharding(
        self, op_overload: torch._ops.OpOverload, op_schema: OpSchema
    ) -> OutputSharding:
        """
        Propagate the sharding for an operator given the op_schema.
        """
        op_key = str(op_overload)
        sharding_prop_func = self.op_to_rules.get(op_key, None)

        if sharding_prop_func is None:
            # step 1. If there's not even one sharding rule
            # implemented for the operator, we fall back to
            # local tensor compute, this is wront currently
            # we will change the behavior to reshard to full
            # replicate and do the computatation
            raise NotImplementedError(
                f"Operator {op_key} does not have a DistributedTensor rule registered."
            )

        # step 2. there's sharding propagation rule, run
        # sharding propagation to get output sharding
        try:
            output_sharding = sharding_prop_func(op_schema)
        except Exception as e:
            raise RuntimeError(
                f"Sharding propagation failed on op {op_key}.\n"
                f"Input schema: {op_schema}.\n"
                f"Error: {e}"
            ) from e

        # step 3. if can't get output_spec from sharding
        # propagation (i.e. no rules apply for input
        # placements), we return the output sharding
        # with schema suggestions, which can be used to
        # decide how to do redistribute on inputs
        if (
            output_sharding.output_spec is None
            and output_sharding.schema_suggestions is None
        ):
            raise RuntimeError(
                f"Sharding propagation failed on op {op_key}!"
                f"Input schema: {op_schema}."
                f"Failed reason: {output_sharding.failed_reason}"
            )
        else:
            return output_sharding
