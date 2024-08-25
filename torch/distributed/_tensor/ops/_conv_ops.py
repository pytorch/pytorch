# mypy: allow-untyped-decorators
# Copyright (c) Meta Platforms, Inc. and affiliates
# implement matrix related ops for distributed tensor
from typing import List

import torch
from torch.distributed._tensor._op_schema import OpSchema, OutputSharding
from torch.distributed._tensor.ops.utils import register_prop_rule
from torch.distributed._tensor.placement_types import DTensorSpec, TensorMeta


aten = torch.ops.aten


@register_prop_rule(aten.convolution.default)
def convolution_rules(op_schema: OpSchema) -> OutputSharding:
    (
        input_spec,
        weight_spec,
        bias_spec,
        stride,
        padding,
        dilation,
        *_,
    ) = op_schema.args_schema

    assert isinstance(input_spec, DTensorSpec)
    assert isinstance(weight_spec, DTensorSpec)
    assert isinstance(bias_spec, DTensorSpec)
    assert input_spec.tensor_meta is not None
    assert weight_spec.tensor_meta is not None
    in_shape = input_spec.tensor_meta.shape
    weight_shape = weight_spec.tensor_meta.shape
    assert isinstance(stride, List)
    assert isinstance(padding, List)
    assert isinstance(dilation, List)
    assert isinstance(weight_shape, torch.Size)
    N, H_in, W_in = in_shape[0], in_shape[2], in_shape[3]
    C_out = weight_shape[0]
    H_out = (H_in + 2 * padding[0] - dilation[0] * (weight_shape[2] - 1) - 1) // stride[
        0
    ] + 1
    W_out = (W_in + 2 * padding[1] - dilation[1] * (weight_shape[3] - 1) - 1) // stride[
        1
    ] + 1
    output_shape = [N, C_out, H_out, W_out]
    output_stride = (C_out * H_out * W_out, H_out * W_out, W_out, 1)
    output_dim_map = input_spec.dim_map
    pending_sums = input_spec.sums

    tensor_meta = TensorMeta(
        torch.Size(output_shape),
        output_stride,
        input_spec.tensor_meta.dtype,
    )
    return OutputSharding(
        DTensorSpec.from_dim_map(
            input_spec.mesh,
            output_dim_map,
            pending_sums,
            tensor_meta=tensor_meta,
        )
    )


@register_prop_rule(aten.convolution_backward.default)
def convolution_backward_rules(op_schema: OpSchema) -> OutputSharding:
    input_spec = op_schema.args_schema[0]
    (
        grad_output_spec,
        input_spec,
        weight_spec,
        bias_shape_opt,
        *_,
    ) = op_schema.args_schema

    assert isinstance(grad_output_spec, DTensorSpec)
    assert isinstance(input_spec, DTensorSpec)
    assert isinstance(weight_spec, DTensorSpec)
    assert isinstance(bias_shape_opt, List)
    assert input_spec.tensor_meta is not None
    weight_tensor_meta = weight_spec.tensor_meta
    bias_tensor_meta = TensorMeta(
        torch.Size(bias_shape_opt),
        (1,),
        input_spec.tensor_meta.dtype,
    )

    grad_input_spec = input_spec
    grad_weight_spec = DTensorSpec.from_dim_map(
        input_spec.mesh,
        [-1, -1, -1, -1],
        [0],
        tensor_meta=weight_tensor_meta,
    )
    grad_bias_spec = DTensorSpec.from_dim_map(
        input_spec.mesh,
        [-1],
        [0],
        tensor_meta=bias_tensor_meta,
    )
    return OutputSharding([grad_input_spec, grad_weight_spec, grad_bias_spec])
