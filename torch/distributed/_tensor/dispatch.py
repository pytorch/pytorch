# Copyright (c) Meta Platforms, Inc. and affiliates
import functools
import operator
from typing import cast, Dict, List, Optional, Sequence, Tuple

import torch

import torch.distributed as dist
import torch.distributed._tensor.api as dtensor
import torch.distributed._tensor.random as random
from torch.distributed._tensor.device_mesh import DeviceMesh
from torch.distributed._tensor.op_schema import (
    _is_inplace_op,
    _is_out_variant_op,
    OpInfo,
    OpSchema,
    OutputSharding,
    OutputSpecType,
)
from torch.distributed._tensor.placement_types import DTensorSpec, Replicate, TensorMeta
from torch.distributed._tensor.random import is_rng_supported_mesh
from torch.distributed._tensor.redistribute import redistribute_local_tensor
from torch.distributed._tensor.sharding_prop import ShardingPropagator
from torch.utils._pytree import tree_flatten, tree_unflatten


def _is_random_op(op):
    aten = torch.ops.aten
    random_ops = [
        aten.native_dropout.default,
        aten.normal_.default,
        aten.uniform_.default,
    ]
    return op in random_ops


def wrap(res: object, spec: OutputSpecType) -> object:
    def to_dt(res, spec):
        assert spec is not None and isinstance(
            spec, DTensorSpec
        ), f"output spec does not match with output! Expected DTensorSpec, got {spec}."
        assert spec.tensor_meta is not None
        return dtensor.DTensor(
            res,
            spec.mesh,
            spec.placements,
            shape=spec.tensor_meta.shape,
            dtype=spec.tensor_meta.dtype,
            requires_grad=res.requires_grad,
            stride=spec.tensor_meta.stride,
        )

    if isinstance(res, torch.Tensor):
        return to_dt(res, spec)
    elif isinstance(res, (list, tuple)):
        assert spec is not None and isinstance(
            spec, (list, tuple)
        ), f"output spec does not match with output! Expected list/tuple, got {spec}."
        res_list = []
        for e, s in zip(res, spec):
            # NOTE: local results might return Optional Tensor from ATen op, so we need
            # to handle that case and make sure we don't wrap None with DTensor.
            # (i.e. native_layer_norm.backward)
            if isinstance(e, (list, tuple)) and isinstance(s, (list, tuple)):
                res_list.append(type(e)([to_dt(ee, ss) for ee, ss in zip(e, s)]))
            elif e is not None and s is not None:
                res_list.append(to_dt(e, s))
            else:
                res_list.append(None)  # type: ignore[arg-type]

        return tuple(res_list) if isinstance(res, tuple) else res_list
    else:
        # if the res contains only non tensor values, we simply return it without rewrapping
        return res


def redistribute_local_args(
    op_info: OpInfo,
    suggested_input_schema: OpSchema,
) -> None:
    # NOTE: it's very rare that we need to reshard kwargs so we intentionally skip it

    # TODO: the op schema should probably just remain flattened so that we can avoid this tree flatten
    # Need to fix all the ops before doing this.
    if op_info.args_tree_spec is not None:
        flatten_args_schema_to_reshard = tuple(
            tree_flatten(suggested_input_schema.args_schema)[0]
        )
    else:
        flatten_args_schema_to_reshard = suggested_input_schema.args_schema

    new_local_args: List[object] = []
    for i, arg_spec in enumerate(op_info.flat_args_schema):
        reshard_arg_spec = flatten_args_schema_to_reshard[i]
        if isinstance(arg_spec, DTensorSpec):
            local_tensor = cast(torch.Tensor, op_info.local_args[i])
            if arg_spec != reshard_arg_spec:
                resharded_local_tensor = redistribute_local_tensor(
                    local_tensor, arg_spec, reshard_arg_spec
                )
                new_local_args.append(resharded_local_tensor)
            else:
                new_local_args.append(local_tensor)
        else:
            new_local_args.append(reshard_arg_spec)

    op_info.local_args = tuple(new_local_args)


def operator_dispatch(
    op_call: torch._ops.OpOverload,
    args: Tuple[object, ...],
    kwargs: Dict[str, object],
    sharding_propagator: ShardingPropagator,
) -> object:
    out, _, _ = _operator_dispatch(op_call, args, kwargs, sharding_propagator)
    return out


def _operator_dispatch(
    op_call: torch._ops.OpOverload,
    args: Tuple[object, ...],
    kwargs: Dict[str, object],
    sharding_propagator: ShardingPropagator,
) -> Tuple[object, OpSchema, OutputSharding]:
    runtime_schema_info = sharding_propagator.op_to_schema_info.get(op_call, None)

    if runtime_schema_info is not None and runtime_schema_info.needs_pytree:
        # flatten args/kwargs when necessary
        tree_args, args_spec = tree_flatten(args)
        args_list: Sequence[object] = tree_args
    else:
        args_list, args_spec = args, None

    args_schema: List[object] = []
    kwargs_schema: Dict[str, object] = {}
    local_args: List[object] = []
    local_kwargs: Dict[str, object] = {}
    mesh: Optional[DeviceMesh] = None

    for arg in args_list:
        if isinstance(arg, dtensor.DTensor):
            args_schema.append(arg._spec)
            local_args.append(arg._local_tensor)
            if mesh is not None:
                if mesh != arg.device_mesh:
                    raise NotImplementedError(
                        f"{op_call}: DTensor does not support cross-mesh operation yet!"
                    )
            else:
                mesh = arg.device_mesh
        elif isinstance(arg, torch.Tensor):
            if arg.ndim == 0 and mesh is not None:
                # scalar tensor can be safely treated as replicated
                args_schema.append(
                    DTensorSpec(
                        mesh,
                        (Replicate(),) * mesh.ndim,
                        tensor_meta=TensorMeta(
                            shape=arg.shape, stride=arg.stride(), dtype=arg.dtype
                        ),
                    )
                )
                local_args.append(arg)
            else:
                raise RuntimeError(
                    f"{op_call}: got mixed torch.Tensor and DTensor, need to convert all"
                    " torch.Tensor to DTensor before calling distributed operators!"
                )
        else:
            args_schema.append(arg)
            local_args.append(arg)

    for k, v in kwargs.items():
        if isinstance(v, dtensor.DTensor):
            kwargs_schema[k] = v._spec
            local_kwargs[k] = v._local_tensor
            if mesh is not None:
                if mesh != v.device_mesh:
                    raise NotImplementedError(
                        f"{op_call}: DTensor does not support cross-mesh operation yet!"
                    )
            else:
                mesh = v.device_mesh
        elif isinstance(v, torch.Tensor):
            raise RuntimeError(
                f"{op_call}: got mixed torch.Tensor and DTensor, need to convert all"
                " torch.Tensor to DTensor before calling distributed operators!"
            )
        else:
            kwargs_schema[k] = v
            local_kwargs[k] = v

    assert mesh is not None, "found no DeviceMesh from dtensor args!"
    op_info = OpInfo(
        mesh,
        OpSchema(
            op_call,
            tree_unflatten(args_schema, args_spec) if args_spec else tuple(args_schema),
            kwargs_schema,
            schema_info=runtime_schema_info,
        ),
        args_schema,
        tuple(local_args),
        local_kwargs,
        args_spec,
    )

    sharding_propagator.propagate(op_info)
    output_sharding = op_info.output_sharding
    assert output_sharding is not None, "output sharding should not be None"

    if mesh.get_coordinate() is None:
        # For a non-participating device, we do:
        #   1. if the return type is scalar, set the local result to None.
        #   The local results from all devices will then be all-gathered
        #   and a reduce op will be performed on the list of results
        #   with appropriate operators:
        #       for bool type, we by default use AND to reduce;
        #       we can extend for more ops if necessary.
        #   2. if the return type is Tensor or List[Tensor], return empty
        #   tensor(s) with correct dtype.
        spec = output_sharding.output_spec
        ret_list = op_info.schema.op._schema.returns

        if spec is None:
            # For a scalar return type, the non-participating device has None
            # as its local result
            local_results: object = None
        else:

            def default_tensor(spec: DTensorSpec) -> torch.Tensor:
                if spec.tensor_meta is not None:
                    shape = spec.tensor_meta.shape
                    dtype = spec.tensor_meta.dtype
                    if len(shape) == 0:
                        # scalar tensor
                        return torch.zeros((), dtype=dtype)
                    else:
                        # non-scalar tensor
                        return torch.tensor([], dtype=dtype)
                else:
                    raise RuntimeError(f"{spec} has no tensor metadata.")

            if isinstance(spec, DTensorSpec):
                # return a Tensor value
                local_results = default_tensor(spec)
            elif isinstance(spec, Sequence):
                # return a List[Tensor] value
                local_results = [
                    default_tensor(s) if s is not None else None for s in spec
                ]
                assert isinstance(local_results, List)
                if None in local_results:
                    ret_type = str(ret_list[0].type)
                    raise NotImplementedError(
                        f"return type {ret_type} in DTensor op is not supported"
                    )
    else:
        if output_sharding.needs_redistribute:
            # compute locally with redistribute first if needed
            assert output_sharding.schema_suggestions is not None
            suggested_input_schema = output_sharding.schema_suggestions[0]
            redistribute_local_args(op_info, suggested_input_schema)

        local_tensor_args = (
            tree_unflatten(cast(List[object], op_info.local_args), args_spec)
            if args_spec
            else op_info.local_args
        )

        # run local op computation with potentially modified args/kwargs
        local_tensor_args = cast(Tuple[object, ...], local_tensor_args)
        if _is_random_op(op_call) and is_rng_supported_mesh(mesh):
            if not random._rng_tracker:
                raise RuntimeError(
                    "A CudaRNGStateTracker instance must be instantiated "
                    "before executing a random op over a DTensor. "
                    "Try calling random.manual_seed() or distribute_tensor() "
                    "before executing a DTensor random op."
                )
            # For DTensor random operator, run it within a distribute region
            with random._rng_tracker._distribute_region(
                cast(DTensorSpec, args_schema[0])
            ):
                local_results = op_call(*local_tensor_args, **local_kwargs)
        else:
            local_results = op_call(*local_tensor_args, **local_kwargs)

    # communicate the result to all ranks for some operators that return scalar value
    if output_sharding.output_spec is None:
        if op_call == torch.ops.aten.equal.default:
            obj_list = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(obj_list, local_results)
            obj_list = list(filter(lambda x: x is not None, obj_list))
            # perform reduce on the collection with AND op
            local_results = functools.reduce(operator.and_, obj_list, True)

    if _is_inplace_op(op_call):
        # inplace op should return self instead of re-wrapping
        self = cast(dtensor.DTensor, args[0])
        self._spec = cast(DTensorSpec, output_sharding.output_spec)
        return self, op_info.schema, output_sharding
    elif _is_out_variant_op(op_call):
        # out variant could possibly have multiple out args (i.e. lu_unpack.out)
        output_specs = (
            (output_sharding.output_spec,)
            if not isinstance(output_sharding.output_spec, tuple)
            else output_sharding.output_spec
        )
        out_dts = []
        spec_idx = 0
        for argument in op_call._schema.arguments:
            if argument.is_out:
                out_dt = cast(dtensor.DTensor, kwargs[argument.name])
                out_dt._spec = cast(DTensorSpec, output_specs[spec_idx])
                out_dts.append(out_dt)
                spec_idx += 1

        assert len(out_dts) >= 1, "out variant should have at least one out arg"
        return (
            tuple(out_dts) if len(out_dts) > 1 else out_dts[0],
            op_info.schema,
            output_sharding,
        )
    else:
        return (
            wrap(local_results, output_sharding.output_spec),
            op_info.schema,
            output_sharding,
        )
