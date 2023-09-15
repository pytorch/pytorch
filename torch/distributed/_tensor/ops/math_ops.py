# Copyright (c) Meta Platforms, Inc. and affiliates
from typing import cast, Optional, Sequence

import torch

import torch.distributed.distributed_c10d as c10d
from torch.distributed._tensor.op_schema import (
    OpSchema,
    OutputSharding,
    RuntimeSchemaInfo,
)
from torch.distributed._tensor.ops.common_rules import pointwise_rule, reduction_rule
from torch.distributed._tensor.ops.utils import (
    as_list,
    normalize_dims,
    register_prop_rule,
)
from torch.distributed._tensor.placement_types import _Partial, DTensorSpec


aten = torch.ops.aten


def _infer_reduction_dims(dims_arg: object, ndim: int) -> Optional[Sequence[int]]:
    if dims_arg is None:
        return None
    dims = cast(Sequence[int], as_list(dims_arg))
    dims = normalize_dims(dims, ndim)
    empty_dims = [[0], [-1], []]
    if ndim == 0 and dims_arg in empty_dims:
        return None
    return dims


@register_prop_rule(aten.all.default)
def default_reduction_rule(op_schema: OpSchema) -> OutputSharding:
    return reduction_rule(op_schema, reduction_linear=True)


@register_prop_rule(
    [
        aten.sum.default,
        aten.sum.dim_IntList,
    ],
    schema_info=RuntimeSchemaInfo(1),
)
def sum_rule(op_schema: OpSchema) -> OutputSharding:
    args_schema = op_schema.args_schema
    input_spec = cast(DTensorSpec, args_schema[0])
    dims = None
    if len(args_schema) > 1:
        dims = _infer_reduction_dims(args_schema[1], input_spec.ndim)

    keep_dim = len(args_schema) > 2 and bool(args_schema[2])
    return reduction_rule(
        op_schema, dims=dims, keep_dim=keep_dim, reduction_linear=True
    )


@register_prop_rule(
    [aten._log_softmax.default, aten._softmax.default], schema_info=RuntimeSchemaInfo(1)
)
def softmax_rule(op_schema: OpSchema) -> OutputSharding:
    input_spec, softmax_dim, _ = op_schema.args_schema
    input_spec = cast(DTensorSpec, input_spec)
    softmax_dim = cast(int, softmax_dim)
    dim_map = input_spec.dim_map
    if softmax_dim < len(dim_map) and dim_map[softmax_dim] >= 0:
        raise RuntimeError("Cannot run softmax on sharding dimension!")
    return OutputSharding(input_spec)


@register_prop_rule(
    [
        aten._log_softmax_backward_data.default,
        aten._softmax_backward_data.default,
    ],
    schema_info=RuntimeSchemaInfo(2),
)
def softmax_bwd_rule(op_schema: OpSchema) -> OutputSharding:
    grad_out_spec, out_spec, softmax_dim, _ = op_schema.args_schema
    grad_out_spec = cast(DTensorSpec, grad_out_spec)
    out_spec = cast(DTensorSpec, out_spec)
    softmax_dim = cast(int, softmax_dim)
    grad_out_dim_map = grad_out_spec.dim_map
    out_dim_map = out_spec.dim_map
    if softmax_dim < len(grad_out_dim_map) and (
        grad_out_dim_map[softmax_dim] >= 0 or out_dim_map[softmax_dim] >= 0
    ):
        raise RuntimeError("Cannot run _softmax_backward_data on sharding dimension!")
    return pointwise_rule(op_schema)


@register_prop_rule(
    [aten.mean.default, aten.mean.dim, aten.mean.out], schema_info=RuntimeSchemaInfo(1)
)
def mean_rule(op_schema: OpSchema) -> OutputSharding:
    args_schema = op_schema.args_schema
    input_spec = cast(DTensorSpec, args_schema[0])
    dims = None
    # if length of args > 1, we check args to find dims
    if len(args_schema) > 1:
        dims = _infer_reduction_dims(args_schema[1], input_spec.ndim)

    keep_dim = len(args_schema) > 2 and bool(args_schema[2])
    output_sharding = reduction_rule(
        op_schema, dims=dims, keep_dim=keep_dim, reduction_linear=True
    )
    if output_sharding.output_spec is not None:
        assert isinstance(output_sharding.output_spec, DTensorSpec)
        for placement in output_sharding.output_spec.placements:
            if placement.is_partial():
                partial_placement = cast(_Partial, placement)
                partial_placement.reduce_op = c10d.ReduceOp.AVG

    return output_sharding


@register_prop_rule(
    [
        aten.var.default,
        aten.var.dim,
        aten.var.out,
    ],
    schema_info=RuntimeSchemaInfo(1),
)
def var_rule(op_schema: OpSchema) -> OutputSharding:
    args_schema = op_schema.args_schema
    input_spec = cast(DTensorSpec, args_schema[0])
    dims = None
    # if length of args > 1, we check args to find dims, note that
    # var.default have unbias arg as the first argument, so we want
    # to check if it's not bool
    if len(args_schema) > 1 and not isinstance(args_schema[1], bool):
        dims = _infer_reduction_dims(args_schema[1], input_spec.ndim)

    keep_dim = len(args_schema) > 3 and bool(args_schema[3])
    return reduction_rule(
        op_schema, dims=dims, keep_dim=keep_dim, reduction_linear=False
    )


@register_prop_rule(
    [aten.var.correction, aten.var.correction_out],
    schema_info=RuntimeSchemaInfo(1, ["keepdim"]),
)
def var_correction_rule(op_schema: OpSchema) -> OutputSharding:
    args_schema = op_schema.args_schema
    input_spec = cast(DTensorSpec, args_schema[0])
    dims = None
    if len(args_schema) > 1:
        dims = _infer_reduction_dims(args_schema[1], input_spec.ndim)

    # keep_dim is a kwarg instead of arg for var.correction
    keep_dim = cast(bool, op_schema.kwargs_schema.get("keepdim", False))
    return reduction_rule(
        op_schema, dims=dims, keep_dim=keep_dim, reduction_linear=False
    )
