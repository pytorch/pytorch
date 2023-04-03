# Copyright (c) Meta Platforms, Inc. and affiliates
# implement matrix related ops for distributed tensor

import torch

from torch.distributed._tensor.api import DTensorSpec, Replicate
from torch.distributed._tensor.op_schema import OpSchema, OutputSharding
from torch.distributed._tensor.ops.utils import register_prop_rule

aten = torch.ops.aten


# TODO: Enable BWD for embedding op.
@register_prop_rule(aten.embedding.default)
def embedding_rules(op_schema: OpSchema) -> OutputSharding:
    weight_spec, inp_spec = op_schema.args_spec
    if any(placement.is_shard(0) for placement in weight_spec.placements):
        raise NotImplementedError(
            "DTensor does not support row-wise sharded embedding operation yet!"
        )

    if all(placement.is_replicate() for placement in inp_spec.placements):
        weight_dim_map = weight_spec.dim_map
        output_dim_map = inp_spec.dim_map
        output_dim_map.append(weight_dim_map[1])
        return OutputSharding(
            output_spec=DTensorSpec.from_dim_map(inp_spec.mesh, output_dim_map, [])
        )

    return OutputSharding(
        output_spec=None,
        schema_suggestions=[
            OpSchema(
                func_schema=op_schema.func_schema,
                args_schema=(
                    weight_spec,
                    DTensorSpec(
                        mesh=inp_spec.mesh,
                        placements=[Replicate()] * len(inp_spec.placements),
                        tensor_meta=inp_spec.tensor_meta,
                    ),
                ),
                kwargs_schema=op_schema.kwargs_schema,
            )
        ],
    )


@register_prop_rule(aten.embedding_renorm_.default)
def embedding_renorm_rules(op_schema: OpSchema) -> OutputSharding:
    raise NotImplementedError(
        "DTensor does not support sharded embedding operation with max_norm yet!"
    )
