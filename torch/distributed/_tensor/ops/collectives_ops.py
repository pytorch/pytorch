# Copyright (c) Meta Platforms, Inc. and affiliates
# implement matrix related ops for distributed tensor

import torch
from torch.distributed._tensor.op_schema import OpSchema, OutputSharding
from torch.distributed._tensor.ops.utils import register_prop_rule
from torch.distributed._tensor.placement_types import DTensorSpec

_c10d_functional = torch.ops._c10d_functional


@register_prop_rule(_c10d_functional.all_to_all_single.default)
def all_to_all_single_rules(op_schema: OpSchema) -> OutputSharding:
    """
    noop pass through
    """

    (
        input_spec,
        output_sizes,
        input_sizes,
        group,
    ) = op_schema.args_schema
    assert isinstance(input_spec, DTensorSpec)
    assert input_spec.tensor_meta is not None

    return OutputSharding(
        DTensorSpec.from_dim_map(
            input_spec.mesh,
            input_spec.dim_map,
            input_spec.sums,
            tensor_meta=input_spec.tensor_meta,
        )
    )


@register_prop_rule(_c10d_functional.wait_tensor.default)
def wait_tensor_rules(op_schema: OpSchema) -> OutputSharding:
    """
    noop pass through
    """

    (input_spec,) = op_schema.args_schema
    assert isinstance(input_spec, DTensorSpec)
    assert input_spec.tensor_meta is not None

    return OutputSharding(
        DTensorSpec.from_dim_map(
            input_spec.mesh,
            input_spec.dim_map,
            input_spec.sums,
            tensor_meta=input_spec.tensor_meta,
        )
    )
