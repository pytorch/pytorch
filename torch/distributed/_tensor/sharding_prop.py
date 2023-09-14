from functools import lru_cache
from typing import Callable, cast, Dict

import torch
from torch._ops import OpOverload
from torch._subclasses import FakeTensorMode
from torch.distributed._tensor.device_mesh import DeviceMesh
from torch.distributed._tensor.op_schema import (
    DTensorSpec,
    OpInfo,
    OpSchema,
    OpStrategy,
    OutputSharding,
    OutputSpecType,
    PlacementStrategy,
    StrategyType,
)
from torch.distributed._tensor.placement_types import TensorMeta


class ShardingPropagator:
    def __init__(self) -> None:
        self.op_to_rules: Dict[OpOverload, Callable[[OpSchema], OutputSharding]] = {}
        self.op_strategy_funcs: Dict[
            OpOverload,
            Callable[[DeviceMesh, OpSchema], StrategyType],
        ] = {}
        self.propagate_op_sharding = lru_cache(None)(self.propagate_op_sharding)  # type: ignore[method-assign]

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
        strategy_func: Callable[[DeviceMesh, OpSchema], StrategyType],
    ):
        """
        Register a sharding strategy generator for an operator.
        """
        self.op_strategy_funcs[op_overload] = strategy_func

    def _propagate_tensor_meta(self, op_schema: OpSchema) -> object:
        """
        Propagate the tensor metadata, it could either return a TensorMeta
        or a list/tuple of TensorMetas
        """
        # special case op list, we don't need to propagate for local
        # scalar. TODO: figure out a better way to handle this
        skip_prop_list = [
            torch.ops.aten._local_scalar_dense.default,
            torch.ops.aten.equal.default,
            torch.ops.aten.is_same_size.default,
        ]
        if op_schema.op in skip_prop_list:
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
            tensor_meta_list = []
            for i, fake_out_item in enumerate(fake_out):
                if isinstance(fake_out_item, torch.Tensor):
                    tensor_meta_list.append(
                        TensorMeta(
                            shape=fake_out_item.shape,
                            stride=fake_out_item.stride(),
                            dtype=fake_out_item.dtype,
                        )
                    )
            return (
                tuple(tensor_meta_list)
                if isinstance(fake_out, tuple)
                else tensor_meta_list
            )
        else:
            # if fake is not a tensor or tuple of tensor, return as none
            return None

    def _wrap_output_spec_tensor_meta(
        self, output_spec: OutputSpecType, output_tensor_meta: object
    ) -> None:
        """
        Wrap the output_spec with the tensor metadata from the output.
        """
        if output_spec is not None:
            if isinstance(output_spec, DTensorSpec):
                assert isinstance(output_tensor_meta, TensorMeta)
                output_spec.tensor_meta = output_tensor_meta
            elif isinstance(output_spec, (tuple, list)):
                for i, spec in enumerate(output_spec):
                    if isinstance(spec, DTensorSpec):
                        assert isinstance(output_tensor_meta, (tuple, list))
                        output_tensor_meta_i = output_tensor_meta[i]
                        assert isinstance(output_tensor_meta_i, TensorMeta)
                        spec.tensor_meta = output_tensor_meta_i

    def propagate(self, op_info: OpInfo) -> None:
        output_sharding = self.propagate_op_sharding(op_info.schema)
        op_info.output_sharding = output_sharding

    def propagate_op_sharding(self, op_schema: OpSchema) -> OutputSharding:
        """
        Propagate the sharding for an operator given the op_schema.
        """
        out_tensor_meta = self._propagate_tensor_meta(op_schema)
        if out_tensor_meta is None:
            return OutputSharding(None, [op_schema])

        def spec_to_strategy(spec: object) -> object:
            if isinstance(spec, DTensorSpec):
                return OpStrategy([PlacementStrategy(spec)])
            else:
                return spec

        if op_schema.op in self.op_strategy_funcs:
            # generate op strategy for the op.
            input_spec = cast(DTensorSpec, op_schema.args_schema[0])
            mesh = input_spec.mesh

            # swap the args spec with args strategies
            args_sharding_strategy = [
                spec_to_strategy(i) for i in op_schema.args_schema
            ]

            # construct a new OpSchema on args for strategy based propagation
            # TODO: op schema should contain flat sharding by default later
            strategy_schema: OpSchema = OpSchema(
                op=op_schema.op,
                args_schema=tuple(args_sharding_strategy),
                kwargs_schema=op_schema.kwargs_schema,
            )

            op_strategy = self.op_strategy_funcs[op_schema.op](mesh, strategy_schema)

            assert isinstance(op_strategy, OpStrategy)
            # we take the first strategy for now
            # TODO: add a min cost selection logic
            output_strategy = op_strategy.strategies[0]
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

            suggestion_schema = None
            if needs_redistribute:
                reshard_schema = OpSchema(op_schema.op, tuple(expected_input_specs), {})
                reshard_schema._inplace_rewrap_schema_suggestion(op_schema)
                suggestion_schema = [reshard_schema]

            output_sharding = OutputSharding(
                output_strategy.output_spec,
                suggestion_schema,
                needs_redistribute=needs_redistribute,
            )
            # associate the output sharding with the output tensor metadata
            self._wrap_output_spec_tensor_meta(
                output_sharding.output_spec, out_tensor_meta
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
                if output_sharding.schema_suggestions is None:
                    if output_sharding.failed_reason is not None:
                        raise RuntimeError(
                            f"Sharding propagation failed on op {op_schema}!"
                            f"Failed reason: {output_sharding.failed_reason}"
                        )
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
                    output_sharding.needs_redistribute = True

            # associate the output sharding with the output tensor metadata
            self._wrap_output_spec_tensor_meta(
                output_sharding.output_spec, out_tensor_meta
            )

            return output_sharding
        else:
            raise NotImplementedError(
                f"Operator {op_schema.op} does not have a sharding strategy registered."
            )
