# Copyright (c) Meta Platforms, Inc. and affiliates
# implement matrix related ops for distributed tensor
from typing import List

try:
    import numpy as np
except ModuleNotFoundError:
    np = None  # type: ignore[assignment]

import torch
from torch.distributed._tensor.op_schema import (
    OpSchema,
    OpStrategy,
    OutputSharding,
    PlacementStrategy,
    RuntimeSchemaInfo,
)
from torch.distributed._tensor.ops.utils import (
    generate_redistribute_costs,
    register_op_strategy,
    register_prop_rule,
)
from torch.distributed._tensor.placement_types import DTensorSpec, TensorMeta
from torch.distributed.device_mesh import DeviceMesh

aten = torch.ops.aten


@register_prop_rule(aten.slice_backward.default)
def slice_backward_rules(op_schema: OpSchema) -> OutputSharding:
    grad_output_spec, input_sizes, dim, start, end, step = op_schema.args_schema
    assert isinstance(grad_output_spec, DTensorSpec)
    assert isinstance(input_sizes, List)
    assert grad_output_spec.tensor_meta is not None
    grad_input_stride = list(np.cumprod(input_sizes[::-1])[:-1][::-1])
    grad_input_stride.append(1)
    dim_map = grad_output_spec.dim_map
    sums = grad_output_spec.sums

    grad_input_tensor_meta = TensorMeta(
        torch.Size(input_sizes),
        tuple(grad_input_stride),
        grad_output_spec.tensor_meta.dtype,
    )
    grad_input_spec = DTensorSpec.from_dim_map(
        grad_output_spec.mesh,
        dim_map,
        sums,
        tensor_meta=grad_input_tensor_meta,
    )

    return OutputSharding(grad_input_spec)


@register_op_strategy(
    [aten.bernoulli.default, aten.bernoulli.out, aten.bernoulli_.float],
    schema_info=RuntimeSchemaInfo(static_kwargkey=["out"]),
)
def bernoulli_strategy(mesh: DeviceMesh, op_schema: OpSchema) -> OpStrategy:
    # args: input
    # kwargs: generator, out
    assert len(op_schema.args_schema) == 1
    input_strategy = op_schema.args_schema[0]
    assert isinstance(input_strategy, OpStrategy)

    # construct output strategy
    output_strategy = OpStrategy([])
    for idx, input_placement_strategy in enumerate(input_strategy.strategies):
        input_src_spec = input_placement_strategy.output_spec

        needs_redistribute = False
        if "out" in op_schema.kwargs_schema:
            out_kwarg_strategy = op_schema.kwargs_schema["out"]
            assert isinstance(out_kwarg_strategy, OpStrategy)
            out_src_spec = out_kwarg_strategy.strategies[idx].output_spec
            if input_src_spec.placements != out_src_spec.placements:
                needs_redistribute = True

        if needs_redistribute:
            assert isinstance(out_src_spec, DTensorSpec)
            output_strategy.strategies.append(
                PlacementStrategy(
                    output_specs=out_src_spec,
                    input_specs=[out_src_spec],
                    redistribute_cost=[
                        generate_redistribute_costs(input_strategy, out_src_spec)
                    ],
                )
            )
        else:
            output_strategy.strategies.append(
                PlacementStrategy(
                    output_specs=input_src_spec,
                    input_specs=[input_src_spec],
                    redistribute_cost=[[0.0 for _ in input_strategy.strategies]],
                )
            )

    return output_strategy
