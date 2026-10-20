# Copyright (c) Meta Platforms, Inc. and affiliates
# implement matrix related ops for distributed tensor

from typing import Any

import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta
from torch.distributed.tensor._op_schema import (
    OpSchema,
    OutputSharding,
    RuntimeSchemaInfo,
)
from torch.distributed.tensor._ops.single_dim_strategy import (
    _ShardingPlaceholder,
    register_single_dim_strategy,
)
from torch.distributed.tensor._ops.utils import register_prop_rule
from torch.distributed.tensor.placement_types import (
    _StridedShard,
    Partial,
    Placement,
    Replicate,
    Shard,
)
from torch.fx.experimental.symbolic_shapes import guard_or_false


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
        _transposed,
        _output_padding,
        _groups,
    ) = op_schema.args_schema

    if not isinstance(input_spec, DTensorSpec):
        raise AssertionError
    if not isinstance(weight_spec, DTensorSpec):
        raise AssertionError
    # bias_spec can be None (optional parameter in aten.convolution schema)
    if bias_spec is not None:
        if not isinstance(bias_spec, DTensorSpec):
            raise AssertionError
    if input_spec.tensor_meta is None:
        raise AssertionError
    if weight_spec.tensor_meta is None:
        raise AssertionError
    in_shape = input_spec.tensor_meta.shape
    weight_shape = weight_spec.tensor_meta.shape
    if not isinstance(stride, list):
        raise AssertionError(f"stride must be list, got {type(stride)}")
    if not isinstance(padding, list):
        raise AssertionError(f"padding must be list, got {type(padding)}")
    if not isinstance(dilation, list):
        raise AssertionError(f"dilation must be list, got {type(dilation)}")
    # weight_shape might not be torch.Size in all cases (e.g., SymIntArrayRef during tracing)
    # so we don't assert its type, just use it
    out_conv_shape = [
        (d + 2 * padding[i] - dilation[i] * (weight_shape[i + 1] - 1) - 1) // stride[i]
        + 1
        for (i, d) in enumerate(in_shape[2:])
    ]
    output_shape = [in_shape[0], weight_shape[0]] + out_conv_shape
    output_stride = [1]
    for i in range(1, len(output_shape)):
        output_stride.insert(0, output_stride[0] * output_shape[-i])
    output_dim_map = input_spec.dim_map
    pending_sums = input_spec.sums

    tensor_meta = TensorMeta(
        torch.Size(output_shape),
        tuple(output_stride),
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
        _stride,
        _padding,
        _dilation,
        _transposed,
        _output_padding,
        _groups,
        _output_mask,
    ) = op_schema.args_schema

    if not isinstance(grad_output_spec, DTensorSpec):
        raise AssertionError
    if not isinstance(input_spec, DTensorSpec):
        raise AssertionError
    if not isinstance(weight_spec, DTensorSpec):
        raise AssertionError
    # bias_shape_opt can be None (optional parameter in aten.convolution_backward schema)
    if bias_shape_opt is not None:
        if not isinstance(bias_shape_opt, list):
            raise AssertionError
    if input_spec.tensor_meta is None:
        raise AssertionError
    weight_tensor_meta = weight_spec.tensor_meta

    # Only create bias_tensor_meta if bias_shape_opt is not None
    if bias_shape_opt is not None:
        bias_tensor_meta = TensorMeta(
            torch.Size(bias_shape_opt),
            (1,),
            input_spec.tensor_meta.dtype,
        )
    else:
        bias_tensor_meta = None

    grad_input_spec = input_spec
    grad_weight_spec = DTensorSpec.from_dim_map(
        input_spec.mesh,
        [-1, -1, -1, -1],
        [0],
        tensor_meta=weight_tensor_meta,
    )

    # Only create grad_bias_spec if we have bias_tensor_meta
    if bias_tensor_meta is not None:
        grad_bias_spec = DTensorSpec.from_dim_map(
            input_spec.mesh,
            [-1],
            [0],
            tensor_meta=bias_tensor_meta,
        )
    else:
        grad_bias_spec = None

    # TODO: actually the output_mask is not respected here, we should
    # set the corresponding spec to `None` if the output_mask is not `False`
    # for a certain output Tensor. This also applies to the conv handler
    # in torch/distributed/tensor/_tp_conv.py
    return OutputSharding([grad_input_spec, grad_weight_spec, grad_bias_spec])


# Single-dim strategies for autoparallel optimizer support.
# These coexist with the prop_rules above — strategies take precedence
# in the propagation path, while the prop_rules + custom handlers in
# _tp_conv.py continue to handle runtime dispatch.


def _supports_last_dim_sharding(
    input_meta: TensorMeta,
    weight_meta: TensorMeta,
    stride: list[Any],
    padding: list[Any],
    dilation: list[Any],
    transposed: bool,
) -> bool:
    ndim = len(input_meta.shape)
    return (
        ndim == len(weight_meta.shape)
        and ndim in (3, 4, 5)
        and not transposed
        and guard_or_false(padding[-1] == 0)
        and guard_or_false(dilation[-1] == 1)
        and guard_or_false(stride[-1] == weight_meta.shape[-1])
    )


def _convolution_full_mesh_strategy_filter(
    mesh: DeviceMesh,
    op_schema: OpSchema,
    input_specs: list[DTensorSpec],
    _output_specs: DTensorSpec | tuple[DTensorSpec | None, ...],
) -> bool:
    if op_schema.op == aten.convolution.default:
        input_idx, stride_idx = 0, 3
    elif op_schema.op == aten.convolution_backward.default:
        input_idx, stride_idx = 1, 4
    else:
        raise AssertionError(f"Unexpected op {op_schema.op}")

    input_spec = input_specs[input_idx]
    last_dim = len(input_spec.shape) - 1
    last_dim_mesh_dims = [
        mesh_dim
        for mesh_dim, placement in enumerate(input_spec.placements)
        if placement.is_shard(last_dim)
    ]
    if not last_dim_mesh_dims:
        return True
    # _tp_conv converts placements to dim_map, which cannot represent one tensor
    # dimension sharded over multiple mesh dimensions.
    if len(last_dim_mesh_dims) != 1:
        return False

    last_dim_mesh_dim = last_dim_mesh_dims[0]
    last_dim_placement = input_spec.placements[last_dim_mesh_dim]
    if not isinstance(last_dim_placement, Shard) or isinstance(
        last_dim_placement, _StridedShard
    ):
        return False

    stride = op_schema.args_schema[stride_idx]
    if not isinstance(stride, list):
        raise AssertionError(f"Expected stride list, got {type(stride)}")
    return guard_or_false(
        input_spec.shape[last_dim] % (stride[-1] * mesh.size(last_dim_mesh_dim)) == 0
    )


@register_single_dim_strategy(
    [aten.convolution.default],
    schema_info=RuntimeSchemaInfo(2),
    full_mesh_strategy_filter=_convolution_full_mesh_strategy_filter,
)
def convolution_single_dim_strategy(
    op: torch._ops.OpOverload,
    args_schema: tuple[Any, ...],
    kwargs_schema: dict[str, Any],
) -> list[list[Placement | _ShardingPlaceholder]]:
    input_meta = args_schema[0]
    if not isinstance(input_meta, TensorMeta):
        raise AssertionError(f"Expected TensorMeta, got {type(input_meta)}")
    weight_meta = args_schema[1]
    if not isinstance(weight_meta, TensorMeta):
        raise AssertionError(f"Expected TensorMeta, got {type(weight_meta)}")
    bias_meta = args_schema[2]
    stride, padding, dilation = args_schema[3:6]
    transposed = args_schema[6]
    # [output, input, weight, (bias)]
    rule: list[Placement | _ShardingPlaceholder] = [
        _ShardingPlaceholder(0),  # output
        _ShardingPlaceholder(0),  # input
        Replicate(),  # weight
    ]
    if bias_meta is not None:
        rule.append(Replicate())  # bias
    strategies = [rule]

    if _supports_last_dim_sharding(
        input_meta, weight_meta, stride, padding, dilation, transposed
    ):
        last_dim = len(input_meta.shape) - 1
        last_dim_rule: list[Placement | _ShardingPlaceholder] = [
            Shard(last_dim),  # output
            Shard(last_dim),  # input
            Replicate(),  # weight
        ]
        if bias_meta is not None:
            last_dim_rule.append(Replicate())  # bias
        strategies.append(last_dim_rule)

    return strategies


@register_single_dim_strategy(
    [aten.convolution_backward.default],
    schema_info=RuntimeSchemaInfo(3),
    full_mesh_strategy_filter=_convolution_full_mesh_strategy_filter,
)
def convolution_backward_single_dim_strategy(
    op: torch._ops.OpOverload,
    args_schema: tuple[Any, ...],
    kwargs_schema: dict[str, Any],
) -> list[list[Placement | _ShardingPlaceholder | None]]:
    input_meta = args_schema[1]
    if not isinstance(input_meta, TensorMeta):
        raise AssertionError(f"Expected TensorMeta, got {type(input_meta)}")
    weight_meta = args_schema[2]
    if not isinstance(weight_meta, TensorMeta):
        raise AssertionError(f"Expected TensorMeta, got {type(weight_meta)}")
    bias_sizes = args_schema[3]
    stride, padding, dilation = args_schema[4:7]
    transposed = args_schema[7]
    has_bias = bias_sizes is not None
    # outputs: [grad_input, grad_weight, grad_bias]
    # inputs: [grad_output, input, weight]
    rule: list[Placement | _ShardingPlaceholder | None] = [
        _ShardingPlaceholder(0),  # grad_input
        Partial("sum"),  # grad_weight
        Partial("sum") if has_bias else None,  # grad_bias
        _ShardingPlaceholder(0),  # grad_output
        _ShardingPlaceholder(0),  # input
        Replicate(),  # weight
    ]
    strategies = [rule]

    if _supports_last_dim_sharding(
        input_meta, weight_meta, stride, padding, dilation, transposed
    ):
        last_dim = len(input_meta.shape) - 1
        strategies.append(
            [
                Shard(last_dim),  # grad_input
                Partial("sum"),  # grad_weight
                Partial("sum") if has_bias else None,  # grad_bias
                Shard(last_dim),  # grad_output
                Shard(last_dim),  # input
                Replicate(),  # weight
            ]
        )

    return strategies
