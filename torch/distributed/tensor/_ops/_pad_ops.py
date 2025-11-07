# Copyright (c) Meta Platforms, Inc. and affiliates
# implement padding ops for distributed tensor
import torch
from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta
from torch.distributed.tensor._op_schema import OpSchema, OutputSharding
from torch.distributed.tensor._ops.utils import register_prop_rule

@register_prop_rule(
    [
        torch.ops.aten.reflection_pad2d.default,
        torch.ops.aten.replication_pad2d.default,
        torch.ops.aten.replication_pad3d.default,
        torch.ops.aten.reflection_pad3d.default,
    ]
)
def padding_rules(op_schema: OpSchema) -> OutputSharding:
    (input_spec, padding) = op_schema.args_schema
    in_shape = input_spec.tensor_meta.shape
    out_conv_shape = [
        (d + padding[2 * i] + padding[2 * i + 1]) for (i, d) in enumerate(in_shape[2:])
    ]
    output_shape = [in_shape[0], in_shape[1]] + out_conv_shape
    output_stride = [1]
    for i in range(1, len(output_shape)):
        output_stride.insert(0, output_stride[0] * output_shape[-i])
    output_dim_map = input_spec.dim_map
    tensor_meta = TensorMeta(
        torch.Size(output_shape),
        tuple(output_stride),
        input_spec.tensor_meta.dtype,
    )

    return OutputSharding(
        DTensorSpec.from_dim_map(
            input_spec.mesh,
            output_dim_map,
            input_spec.sums,
            tensor_meta=tensor_meta,
        )
    )
