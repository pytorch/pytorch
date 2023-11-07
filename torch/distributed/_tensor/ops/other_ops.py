# Copyright (c) Meta Platforms, Inc. and affiliates
# implement matrix related ops for distributed tensor
from typing import cast
import numpy as np

import torch
from torch.distributed._tensor.op_schema import OpSchema, OutputSharding
from torch.distributed._tensor.ops.common_rules import einop_rule, pointwise_rule
from torch.distributed._tensor.ops.utils import register_prop_rule
from torch.distributed._tensor.placement_types import DTensorSpec, TensorMeta

aten = torch.ops.aten


@register_prop_rule(aten.slice_backward.default)
def slice_backward_rules(op_schema: OpSchema) -> OutputSharding:
    grad_output_spec, input_sizes, dim, start, end, step = op_schema.args_schema
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


@register_prop_rule(aten.native_layer_norm.default)
def layer_norm_rules(op_schema: OpSchema) -> OutputSharding:
    input_spec, normalized_shape = op_schema.args_schema[0:2]

    input_shape = input_spec.tensor_meta.shape
    input_dim = len(input_shape)
    norm_dim = len(normalized_shape)
    mean_dim = input_dim - norm_dim
    mean_shape = list(input_shape[0:mean_dim]) + [1 for _ in range(norm_dim)]
    mean_stride = list(np.cumprod(mean_shape[::-1])[:-1][::-1])
    mean_stride.append(1)
    assert len(input_shape) > len(normalized_shape)
    assert list(input_shape[-norm_dim:]) == normalized_shape

    output_tensor_meta = input_spec.tensor_meta
    mean_tensor_meta = TensorMeta(
        torch.Size(mean_shape),
        tuple(mean_stride),
        input_spec.tensor_meta.dtype,
    )
    rstd_tensor_meta = TensorMeta(
        torch.Size(mean_shape),
        tuple(mean_stride),
        input_spec.tensor_meta.dtype,
    )

    output_spec = DTensorSpec.from_dim_map(
        input_spec.mesh,
        [-1 for _ in range(input_dim)],
        [],
        tensor_meta=output_tensor_meta,
    )

    mean_spec = DTensorSpec.from_dim_map(
        input_spec.mesh,
        [-1 for _ in range(mean_dim)],
        [],
        tensor_meta=mean_tensor_meta,
    )

    rstd_spec = DTensorSpec.from_dim_map(
        input_spec.mesh,
        [-1 for _ in range(mean_dim)],
        [],
        tensor_meta=rstd_tensor_meta,
    )

    return OutputSharding([output_spec, mean_spec, rstd_spec])


@register_prop_rule(aten.native_layer_norm_backward.default)
def layer_norm_backward_rules(op_schema: OpSchema) -> OutputSharding:
    input_spec = op_schema.args_schema[1]
    weight_spec = op_schema.args_schema[5]
    bias_spec = op_schema.args_schema[6]

    grad_input_spec = DTensorSpec.from_dim_map(
        input_spec.mesh,
        [-1 for _ in range(len(input_spec.tensor_meta.shape))],
        [],
        tensor_meta=input_spec.tensor_meta,
    )

    grad_weight_spec = weight_spec
    grad_bias_spec = bias_spec
    return OutputSharding([grad_input_spec, grad_weight_spec, grad_bias_spec])


@register_prop_rule(aten.bernoulli_.float)
def bernoulli_rules(op_schema: OpSchema) -> OutputSharding:
    input_spec = op_schema.args_schema[0]
    return OutputSharding(input_spec)


@register_prop_rule(aten.nll_loss_forward.default)
def nll_loss_forward_rules(op_schema: OpSchema) -> OutputSharding:
    input_spec = op_schema.args_schema[0]

    result_shape = []
    result_stride = []
    result_dim = 0
    total_weight_shape = []
    total_weight_stride = []
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
    return OutputSharding(input_spec)
