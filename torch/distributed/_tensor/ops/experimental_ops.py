# Copyright (c) Meta Platforms, Inc. and affiliates
# implement matrix related ops for distributed tensor
from typing import List

import numpy as np

import torch
from torch.distributed._tensor.op_schema import OpSchema, OutputSharding
from torch.distributed._tensor.ops.utils import register_prop_rule
from torch.distributed._tensor.placement_types import (
    _Partial,
    DTensorSpec,
    Replicate,
    TensorMeta,
)

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


@register_prop_rule(aten.native_layer_norm_backward.default)
def _prop_native_layer_norm_backward(op_schema: OpSchema) -> OutputSharding:
    (
        grad,
        input,
        normalized_shape,
        result1,
        result2,
        weight,
        bias,
        grad_input_mask,
    ) = op_schema.args_schema
    assert isinstance(grad, DTensorSpec)
    assert isinstance(grad_input_mask, (list, tuple))
    if weight is not None:
        assert isinstance(weight, DTensorSpec)
        assert all(isinstance(s, Replicate) for s in weight.placements)
    if bias is not None:
        assert isinstance(bias, DTensorSpec)
        assert all(isinstance(s, Replicate) for s in bias.placements)

    weight_grad = (
        DTensorSpec(
            mesh=weight.mesh,
            placements=tuple([_Partial()] * weight.mesh.ndim),
        )
        if weight
        else None
    )
    bias_grad = (
        DTensorSpec(
            mesh=bias.mesh,
            placements=tuple([_Partial()] * bias.mesh.ndim),
        )
        if bias
        else None
    )
    return OutputSharding(
        # NOTE: type errors below are legit. This is because DTensor currently
        # doesn't support Optional return values. Need to be fixed in DTensor repo.
        output_spec=(
            grad if grad_input_mask[0] else None,
            weight_grad if grad_input_mask[1] else None,
            bias_grad if grad_input_mask[2] else None,
        ),
    )
