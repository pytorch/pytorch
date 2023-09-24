# Copyright (c) Meta Platforms, Inc. and affiliates
# implement matrix related ops for distributed tensor

import torch

from torch.distributed._tensor.op_schema import OpSchema, OutputSharding
from torch.distributed._tensor.ops.common_rules import einop_rule, pointwise_rule
from torch.distributed._tensor.ops.utils import register_prop_rule
from torch.distributed._tensor.placement_types import DTensorSpec, Shard, TensorMeta

aten = torch.ops.aten


def _update_schema_suggestion_for_addmm(
    output_sharding: OutputSharding,
    op_schema: OpSchema,
    pointwise_add_update: bool = True,
) -> OutputSharding:
    # schema suggestion coming from output sharding could be:
    # 1. pointwise add sharding input suggestion
    # 2. mm sharding input suggestion
    # inplace update schema suggestion to return addmm suggestion
    assert output_sharding.schema_suggestions is not None
    suggestion = output_sharding.schema_suggestions[0]
    if pointwise_add_update:
        # update with pointwise suggestion
        args_schema = (
            suggestion.args_schema[0],
            op_schema.args_schema[1],
            op_schema.args_schema[2],
        )
    else:
        # update with mm suggestion
        args_schema = (
            op_schema.args_schema[0],
            suggestion.args_schema[0],
            suggestion.args_schema[1],
        )

    output_sharding.schema_suggestions = [
        OpSchema(
            op=op_schema.op,
            args_schema=args_schema,
            kwargs_schema=op_schema.kwargs_schema,
        )
    ]
    return output_sharding


@register_prop_rule(aten.mm.default)
def mm_rules(op_schema: OpSchema) -> OutputSharding:
    return einop_rule("mk,kn->mn", op_schema, linearity=False)


@register_prop_rule(aten.addmm.default)
def addmm_rules(op_schema: OpSchema) -> OutputSharding:
    input_spec, mat1_spec, mat2_spec = op_schema.args_spec
    mm_out_sharding = mm_rules(OpSchema(op_schema.op, (mat1_spec, mat2_spec), {}))
    if mm_out_sharding.output_spec is None:
        # non-eligible input, suggest addmm input specs
        if mm_out_sharding.schema_suggestions is not None:
            # TODO: add more suggestions for resharding
            return _update_schema_suggestion_for_addmm(
                mm_out_sharding,
                op_schema,
                pointwise_add_update=False,
            )
        else:
            return OutputSharding(None)

    # run point wise rule on input + (mm_out) with linearity
    output_sharding = pointwise_rule(
        OpSchema(op_schema.op, (input_spec, mm_out_sharding.output_spec), {}),
        linearity=True,
    )
    # if propagation failed, edit the schema suggestion from pointwise rules
    # to return addmm suggestion instead as it's a chained suggestion.
    if (
        output_sharding.output_spec is None
        and output_sharding.schema_suggestions is not None
    ):
        return _update_schema_suggestion_for_addmm(output_sharding, op_schema)

    return output_sharding


@register_prop_rule(aten.linear.default)
def linear_rules(op_schema: OpSchema) -> OutputSharding:
    if len(op_schema.args_spec) < 3:
        input_spec, weight_spec = op_schema.args_spec
        bias_spec = None
    else:
        input_spec, weight_spec, bias_spec = op_schema.args_spec
    if (not isinstance(weight_spec, DTensorSpec) or len(weight_spec.placements) > 1):
        raise NotImplementedError("Only support weight sharded by one dimension.")
    input_dim = len(input_spec.tensor_meta.shape)
    t_weight_spec = DTensorSpec(
        mesh=weight_spec.mesh,
        placements=tuple(
            Shard(len(weight_spec.tensor_meta.shape) - p.dim - 1)
            for p in weight_spec.placements
        ),
        tensor_meta=TensorMeta(
            shape=(weight_spec.tensor_meta.shape[1], weight_spec.tensor_meta.shape[0]),
            stride=(weight_spec.tensor_meta.stride[1], weight_spec.tensor_meta.stride[0]),
            dtype=weight_spec.tensor_meta.dtype,
        )
    )
    alphabet = "abcdefghijlopqrstuvwxyz"
    einrule = f"m{alphabet[:input_dim - 2]}k,kn->mn"
    mm_out_sharding = einop_rule(einrule, OpSchema(op_schema.op, (input_spec, t_weight_spec), {}), linearity=False)
    if mm_out_sharding.output_spec is None:
        # non-eligible input, suggest addmm input specs
        if mm_out_sharding.schema_suggestions is not None:
            # TODO: add more suggestions for resharding
            return _update_schema_suggestion_for_addmm(
                mm_out_sharding,
                op_schema,
                pointwise_add_update=False,
            )
        else:
            return OutputSharding(None)

    if bias_spec is not None:
        # run point wise rule on input + (mm_out) with linearity
        output_sharding = pointwise_rule(
            OpSchema(op_schema.op, (bias_spec, mm_out_sharding.output_spec), {}),
            linearity=True,
        )
        # if propagation failed, edit the schema suggestion from pointwise rules
        # to return addmm suggestion instead as it's a chained suggestion.
        if (
            output_sharding.output_spec is None
            and output_sharding.schema_suggestions is not None
        ):
            return _update_schema_suggestion_for_addmm(output_sharding, op_schema)
        return output_sharding
    else:
        return mm_out_sharding


@register_prop_rule(aten.t.default)
def transpose_rule(op_schema: OpSchema) -> OutputSharding:
    return einop_rule("ij->ji", op_schema, linearity=True)


@register_prop_rule(aten.bmm.default)
def bmm_rules(op_schema: OpSchema) -> OutputSharding:
    return einop_rule("bmk,bkn->bmn", op_schema, linearity=False)


@register_prop_rule(aten.baddbmm.default)
def baddbmm_rules(op_schema: OpSchema) -> OutputSharding:
    input_spec, mat1_spec, mat2_spec = op_schema.args_spec
    bmm_output_sharding = bmm_rules(OpSchema(op_schema.op, (mat1_spec, mat2_spec), {}))
    if bmm_output_sharding.output_spec is None:
        # TODO: add more suggestions
        if bmm_output_sharding.schema_suggestions is not None:
            return _update_schema_suggestion_for_addmm(
                bmm_output_sharding,
                op_schema,
                pointwise_add_update=False,
            )
        else:
            return OutputSharding(None)

    # run point wise rule on input + (bmm_out) with linearity
    output_sharding = pointwise_rule(
        OpSchema(
            op_schema.op,
            (input_spec, bmm_output_sharding.output_spec),
            {},
        ),
        linearity=True,
    )
    # if propagation failed, edit the schema suggestion from pointwise rules
    # to return baddbmm suggestion instead as it's a chained suggestion.
    if (
        output_sharding.output_spec is None
        and output_sharding.schema_suggestions is not None
    ):
        return _update_schema_suggestion_for_addmm(output_sharding, op_schema)

    return output_sharding
