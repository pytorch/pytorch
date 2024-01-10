# Copyright (c) Meta Platforms, Inc. and affiliates
# implement matrix related ops for distributed tensor
from typing import List

import numpy as np

import torch
from torch.distributed._tensor.op_schema import OpSchema, OutputSharding
from torch.distributed._tensor.ops.utils import register_prop_rule
from torch.distributed._tensor.placement_types import DTensorSpec, TensorMeta

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


@register_prop_rule(aten.bernoulli.default)
@register_prop_rule(aten.bernoulli_.float)
def bernoulli_rules(op_schema: OpSchema) -> OutputSharding:
    input_spec = op_schema.args_schema[0]
    assert isinstance(input_spec, DTensorSpec)
    return OutputSharding(input_spec)


@register_prop_rule(aten.nll_loss_forward.default)
def nll_loss_forward_rules(op_schema: OpSchema) -> OutputSharding:
    input_spec = op_schema.args_schema[0]
    assert isinstance(input_spec, DTensorSpec)
    assert input_spec.tensor_meta is not None
    result_shape: List[int] = []
    result_stride: List[int] = []
    result_dim = 0
    total_weight_shape: List[int] = []
    total_weight_stride: List[int] = []
    total_weight_dim = 0

    result_tensor_meta = TensorMeta(
        torch.Size(result_shape),
        tuple(result_stride),
        input_spec.tensor_meta.dtype,
    )
    total_weight_tensor_meta = TensorMeta(
        torch.Size(total_weight_shape),
        tuple(result_stride),
        input_spec.tensor_meta.dtype,
    )
    result_spec = DTensorSpec.from_dim_map(
        input_spec.mesh,
        [-1 for _ in range(result_dim)],
        [],
        tensor_meta=result_tensor_meta,
    )
    total_weight_spec = DTensorSpec.from_dim_map(
        input_spec.mesh,
        [-1 for _ in range(total_weight_dim)],
        [],
        tensor_meta=total_weight_tensor_meta,
    )
    return OutputSharding([result_spec, total_weight_spec])


@register_prop_rule(aten.nll_loss_backward.default)
def nll_loss_backward_rules(op_schema: OpSchema) -> OutputSharding:
    input_spec = op_schema.args_schema[1]
    assert isinstance(input_spec, DTensorSpec)
    return OutputSharding(input_spec)
