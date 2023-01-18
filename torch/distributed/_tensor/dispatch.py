# Copyright (c) Meta Platforms, Inc. and affiliates
from typing import Callable, cast, Dict, Tuple, Union, Optional

import torch

import torch.distributed._tensor.api as dtensor
from torch.distributed._tensor.op_schema import (
    ArgsType,
    KwargsType,
    OpSchema,
    OutputSpecType,
)
from torch.distributed._tensor.placement_types import DTensorSpec
from torch.distributed._tensor.prop import ShardingPropagator
from torch.distributed._tensor.redistribute import redistribute_dtensor
from torch.utils._pytree import tree_flatten, tree_map, tree_unflatten


"""
If _ENABLE_FALLBACK set to False, dispatch will fail when an op doesn't
have a sharding rule registered.
"""
_ENABLE_FALLBACK = False


"""
Print information on ops input shape and sharding for debugging purposes.
"""
_DEBUG_VERBOSE = False


def unwrap_schema(e: object) -> object:
    return e._spec if isinstance(e, dtensor.DTensor) else e


def wrap(res: object, spec: OutputSpecType) -> object:
    if isinstance(res, torch.Tensor):
        assert spec is not None and isinstance(
            spec, DTensorSpec
        ), f"output spec does not match with output! Expected DTensorSpec, got {spec}."
        return dtensor.DTensor(
            res,
            spec.mesh,
            spec.placements,
            size=spec.shape,
            requires_grad=res.requires_grad,
        )
    elif isinstance(res, list):
        assert spec is not None and isinstance(
            spec, list
        ), f"output spec does not match with output! Expected list, got {spec}."
        return list(
            dtensor.DTensor(e, s.mesh, s.placements, size=s.shape)
            for e, s in zip(res, spec)
        )
    elif isinstance(res, tuple):
        assert spec is not None and isinstance(
            spec, tuple
        ), f"output spec does not match with output! Expected tuple, got {spec}"

        # NOTE: local results might return Optional Tensor from ATen op, so we need to
        # handle that case and make sure we don't wrap None with DTensor.
        # (i.e. native_layer_norm.backward)
        return tuple(
            dtensor.DTensor(e, s.mesh, s.placements, size=s.shape)
            if e is not None and s is not None
            else None
            for e, s in zip(res, spec)
        )
    else:
        # if the res contains only non tensor values, we simply return it without rewrapping
        return res


def pack_args_kwargs_with_local_tensor(
    args: Union[ArgsType, KwargsType],
    args_schema: Union[ArgsType, KwargsType],
    redistribute_with_schema: bool = False,
) -> Union[ArgsType, KwargsType]:
    flatten_args, args_tree_spec = tree_flatten(args)
    flatten_args_schema, _ = tree_flatten(args_schema)

    for i, arg in enumerate(flatten_args):
        if isinstance(arg, dtensor.DTensor):
            if redistribute_with_schema:
                target_spec = flatten_args_schema[i]
                arg = redistribute_dtensor(
                    arg, target_spec.mesh, target_spec.placements
                )

            # reuse the schema list and update it with local tensor
            flatten_args_schema[i] = arg._local_tensor

    return tree_unflatten(flatten_args_schema, args_tree_spec)


def _reshape_alias(
    x: torch.Tensor, shape: Tuple[int, ...], strides: Tuple[int, ...]
) -> torch.Tensor:
    return torch.ops.aten.view(x, shape)


_CURRENT_DECOMPOSITION_TABLE: Dict[Callable[..., object], Callable[..., object]] = {
    torch.ops.aten._reshape_alias.default: _reshape_alias,
}


def operator_dispatch(
    op_call: torch._ops.OpOverload,
    args: Tuple[object, ...],
    kwargs: Dict[str, object],
    sharding_propagator: ShardingPropagator,
    custom_dispatch_ops: Optional[Dict[str, Callable[..., object]]] = None,
) -> object:
    # first we need to lift some private aten aliases to public calls
    if op_call in _CURRENT_DECOMPOSITION_TABLE:
        return _CURRENT_DECOMPOSITION_TABLE[op_call](*args, **kwargs)

    # STEP 0. See if threre're user defined custom aten operator
    # implementations. Custom operators take the highest priority
    if custom_dispatch_ops is not None and str(op_call) in custom_dispatch_ops:
        # dispatch to user defined custom distributed tensor ops
        return custom_dispatch_ops[str(op_call)](*args, **kwargs)

    # unwrap the args/kwargs schema
    args_schema = tree_map(unwrap_schema, args)
    kwargs_schema = tree_map(unwrap_schema, kwargs)

    op_schema = OpSchema(op_call._schema, args_schema, kwargs_schema)

    if _DEBUG_VERBOSE and torch.distributed.get_rank() == 0:
        print(f"OpSchema({op_schema})")
        local_shapes = tree_map(
            lambda t: t.to_local().shape if isinstance(t, dtensor.DTensor) else None,
            args,
        )
        print(f"    local shapes: {local_shapes}")

    output_sharding = sharding_propagator.propagate_op_sharding(op_call, op_schema)
    needs_redistribute = False
    target_schema = op_schema

    if output_sharding.output_spec is None:
        if output_sharding.schema_suggestions is None:
            raise RuntimeError(
                f"Sharding propagation failed on op {op_call}!"
                f"Input schema: {op_schema}."
                f"Failed reason: {output_sharding.failed_reason}"
            )
        else:
            needs_redistribute = True
            # we do auto redistribute on inputs if necessary
            # to get an eligble input, which we will pick a
            # target schema base on the redistribute cost.
            # For now we simply pick the first suggestion.
            # TODO: implement full auto distribute with a
            # simple cost estimation model
            target_schema = output_sharding.schema_suggestions[0]
            # run sharding propagation again with target schema
            output_sharding = sharding_propagator.propagate_op_sharding(
                op_call, target_schema
            )

    local_tensor_args = pack_args_kwargs_with_local_tensor(
        args,
        target_schema.args_schema,
        redistribute_with_schema=needs_redistribute,
    )
    local_tensor_kwargs = pack_args_kwargs_with_local_tensor(
        kwargs,
        target_schema.kwargs_schema,
        redistribute_with_schema=needs_redistribute,
    )

    # run local op computation with potentially modified args/kwargs
    local_tensor_args = cast(Tuple[object, ...], local_tensor_args)
    local_tensor_kwargs = cast(Dict[str, object], local_tensor_kwargs)
    local_results = op_call(*local_tensor_args, **local_tensor_kwargs)

    if target_schema.is_inplace:
        # inplace op should return self instead of re-wrapping
        self = cast(dtensor.DTensor, args[0])
        self._spec = cast(DTensorSpec, output_sharding.output_spec)
        return self
    elif target_schema.is_out_variant:
        # out variant could possibly have multiple out args (i.e. lu_unpack.out)
        output_specs = (
            (output_sharding.output_spec,)
            if not isinstance(output_sharding.output_spec, tuple)
            else output_sharding.output_spec
        )
        out_dts = []
        spec_idx = 0
        for arg in target_schema.func_schema.arguments:
            if arg.is_out:
                out_dt = cast(dtensor.DTensor, kwargs[arg.name])
                out_dt._spec = cast(DTensorSpec, output_specs[spec_idx])
                out_dts.append(out_dt)
                spec_idx += 1

        assert len(out_dts) >= 1, "out variant should have at least one out arg"
        return tuple(out_dts) if len(out_dts) > 1 else out_dts[0]
    else:
        return wrap(local_results, output_sharding.output_spec)
