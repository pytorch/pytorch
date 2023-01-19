from typing import Callable, Dict

from torch._ops import OpOverload
from torch._subclasses import FakeTensorMode
from torch.fx.experimental.proxy_tensor import make_fx
from torch.distributed._tensor.op_schema import OpSchema, OutputSharding


class ShardingPropagator(object):
    def __init__(self) -> None:
        self.op_to_rules: Dict[OpOverload, Callable[[OpSchema], OutputSharding]] = {}
        self.decomposition_table: Dict[OpOverload, Callable] = {}
        self.fake_mode = FakeTensorMode()
        # self.compiled_decomp_cache: Dict[Callable, torch.fx.Graph]

    def register_sharding_prop_rule(
        self, op_overload: OpOverload, rule_func: Callable[[OpSchema], OutputSharding]
    ):
        """
        Register a sharding propagation rule for an operator.
        """
        self.op_to_rules[op_overload] = rule_func

    def register_decomposition(
        self, op_overload: OpOverload, decomp_func: Callable
    ):
        """
        Register a decomposition for an op
        """
        self.decomposition_table[op_overload] = decomp_func

    def propagate_op_sharding(
        self, op_overload: OpOverload, op_schema: OpSchema
    ) -> OutputSharding:
        """
        Propagate the sharding for an operator given the op_schema.
        """

        # op_to_trace = self.decomposition_table[op_overload] if op_overload in self.decomposition_table else op_overload

        # with self.fake_mode:
        #     op_g = make_fx

        sharding_prop_func = self.op_to_rules.get(op_overload, None)

        if sharding_prop_func is None:
            # step 1. If there's not even one sharding rule
            # implemented for the operator, we fall back to
            # local tensor compute, this is wront currently
            # we will change the behavior to reshard to full
            # replicate and do the computatation
            raise NotImplementedError(
                f"Operator {op_overload} does not have a DistributedTensor rule registered."
            )

        # step 2. there's sharding propagation rule, run
        # sharding propagation to get output sharding
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
        if (
            output_sharding.output_spec is None
            and output_sharding.schema_suggestions is None
        ):
            raise RuntimeError(
                f"Sharding propagation failed on op {op_overload}!"
                f"Input schema: {op_schema}."
                f"Failed reason: {output_sharding.failed_reason}"
            )
        else:
            return output_sharding

    def _propagate_op_sharding_with_decomposition(self, decomp_func, op_schema: OpSchema) -> OutputSharding:
        pass
        

    def propagate_tensor_meta(self):
        pass



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
