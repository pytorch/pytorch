# Copyright (c) Meta Platforms, Inc. and affiliates
import contextlib
import functools
import logging
import operator
import warnings
from typing import cast, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING

import torch
import torch.distributed as dist
import torch.distributed._tensor.api as dtensor
import torch.distributed._tensor.random as random
from torch.distributed._tensor._op_schema import (
    _is_inplace_op,
    _is_out_variant_op,
    OpInfo,
    OpSchema,
    OutputSpecType,
)
from torch.distributed._tensor._redistribute import redistribute_local_tensor
from torch.distributed._tensor._sharding_prop import ShardingPropagator
from torch.distributed._tensor._tp_conv import (
    convolution_backward_handler,
    convolution_handler,
)
from torch.distributed._tensor._utils import try_find_mesh_from_args
from torch.distributed._tensor.placement_types import DTensorSpec, Partial, Replicate, TensorMeta
from torch.distributed._tensor.random import is_rng_supported_mesh


if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh

try:
    from torch.utils import _cxx_pytree as pytree
except ImportError:
    from torch.utils import _pytree as pytree  # type: ignore[no-redef]

aten = torch.ops.aten
logger = logging.getLogger(__name__)


def decompose_handler(
    op_call: torch._ops.OpOverload,
    args: Tuple[object, ...],
    kwargs: Dict[str, object],
) -> object:
    """
    Decomposes a op to core ATen op, this handler is mostly here
    for inference mode usage where the ops are not core aten ops.
    """
    r = op_call.decompose(*args, **kwargs)
    if r is not NotImplemented:
        return r
    else:
        raise RuntimeError("Decomposition failed")


def is_same_size_handler(
    op_call: torch._ops.OpOverload,
    args: Tuple[object, ...],
    kwargs: Dict[str, object],
) -> bool:
    lhs = cast(torch.Tensor, args[0])
    rhs = cast(torch.Tensor, args[1])
    return lhs.shape == rhs.shape


class OpDispatcher:
    """
    Op dispatching class instance to handle args/kwargs pre-processing (un-wrapping), sharding
    propagation, redistribute local args, local compute, and post-processing (re-wrapping). It
    also handles any op specific logic if necessary.

    NOTE: Given the runtime overhead of Tensor subclass (__torch_dispatch__), the OpDispatcher
    is designed to minimize the CPU overhead by using the tricks of proper unflattening, faster
    pytree if needed, and leveraging various caching mechanisms implemented in the sharding
    propagation and redistribute modules. The CPU overhead is critical to eager mode performance,
    one need to carefully measure the CPU overhead when making significant changes to the
    OpDispatcher and ShardingPropagator.
    """

    def __init__(self) -> None:
        self.sharding_propagator = ShardingPropagator()
        self._random_ops = {
            aten.native_dropout.default,
            aten.normal_.default,
            aten.rand_like.default,
            aten.randn_like.default,
            aten.randint_like.default,
            aten.randint_like.low_dtype,
            aten.randint_like.low_dtype_out,
            aten.uniform_.default,
            aten.bernoulli.default,
            aten.bernoulli_.float,
        }
        self._custom_op_handlers = {
            aten.linear.default: decompose_handler,
            aten.is_same_size.default: is_same_size_handler,
            aten.convolution.default: convolution_handler,
            aten.convolution_backward.default: convolution_backward_handler,
        }

        # This flag is used internally to control whether we treat the torch.Tensor(non-DTensor)
        # as implicitly replicated or we throw error to user.
        # NOTE: It is EXTREMELY UNSAFE to turn this flag on by default so we intentionally leave
        # it as False by default.
        self._allow_implicit_replication = False

    def dispatch(
        self,
        op_call: torch._ops.OpOverload,
        args: Tuple[object, ...],
        kwargs: Dict[str, object],
    ) -> object:
        """
        Main dispatching logic
        """
        # operators that does not need to go through sharding propagation
        if op_call in self._custom_op_handlers:
            return self._custom_op_handlers[op_call](op_call, args, kwargs)  # type: ignore[operator]

        # extract local tensor and sharding infos to a OpInfo
        op_info = self.unwrap_to_op_info(op_call, args, kwargs)
        logger.debug("Dispatching op_call: %s", op_info.schema)

        self.sharding_propagator.propagate(op_info)
        output_sharding = op_info.output_sharding
        logger.debug("output_sharding for %s: %s", op_call, output_sharding)
        assert output_sharding is not None, "output sharding should not be None"

        mesh = op_info.mesh
        if mesh.get_coordinate() is not None:
            # computation that happens in the current rank of the mesh, normal case
            if output_sharding.needs_redistribute:
                # If sharding propagation decision needs redistribute, perform redistribute
                # on args first, which could potentially modify args (i.e. allgather certain arg)
                assert output_sharding.redistribute_schema is not None
                self.redistribute_local_args(
                    op_info, output_sharding.redistribute_schema
                )

            local_tensor_args = (
                pytree.tree_unflatten(
                    cast(List[object], op_info.local_args), op_info.args_tree_spec
                )
                if op_info.args_tree_spec
                else op_info.local_args
            )

            # run local op computation with potentially modified args/kwargs
            local_tensor_args = cast(Tuple[object, ...], local_tensor_args)
            if op_call in self._random_ops:
                if not random._rng_tracker and is_rng_supported_mesh(mesh):
                    # Default to `OffsetBasedRNGTracker` if the parallelism API
                    # did not already construct one
                    random._rng_tracker = random.OffsetBasedRNGTracker(mesh.device_type)

                first_arg, first_local_arg = cast(dtensor.DTensor, args[0]), cast(
                    torch.Tensor, local_tensor_args[0]
                )
                rng_context = (
                    random._rng_tracker._distribute_region(first_arg._spec)
                    if random._rng_tracker and not first_local_arg.is_meta
                    else contextlib.nullcontext()
                )
                # For DTensor random operator, run it within a RNGTracker context to
                # ensure the random number generator is properly distributed.
                with rng_context:
                    local_results = op_call(*local_tensor_args, **op_info.local_kwargs)
            else:
                # normal case, run local sharded op computation
                local_results = op_call(*local_tensor_args, **op_info.local_kwargs)

        else:
            # For a non-participating device (happens on rank that does not belong to
            # the device mesh), we do:
            #   1. if the return type is scalar, set the local result to None.
            #   2. if the return type is Tensor or List[Tensor], return empty
            #   tensor(s) with correct dtype.
            spec = output_sharding.output_spec
            ret_list = op_info.schema.op._schema.returns

            if spec is None:
                # For a scalar return type, the non-participating device has None
                # as its local result
                local_results = None
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

        if output_sharding.output_spec is None:
            if op_call == aten.equal.default:
                # For equal operator, The local results from all devices should be all-gathered
                # and a reduce op (AND) will be performed on the list of results to ensure SPMD
                # execution. We can extend this for more ops if necessary.
                obj_list = [None for _ in range(dist.get_world_size())]
                dist.all_gather_object(obj_list, local_results)  # type: ignore[possibly-undefined]
                obj_list = list(filter(lambda x: x is not None, obj_list))
                # perform reduce on the collection with AND op
                local_results = functools.reduce(operator.and_, obj_list, True)

        if _is_inplace_op(op_call):
            # inplace op should return self instead of re-wrapping
            if output_sharding.output_spec is not None:
                return args[0]
            else:
                if op_call == aten._amp_foreach_non_finite_check_and_unscale_.default:
                    found_inf_dtensor = dtensor.DTensor.from_local(args[1], mesh, [Partial("max")])
                    found_inf = found_inf_dtensor.max().full_tensor()
                    arg_list = list(args)
                    arg_list[1].copy_(found_inf)
                    args = tuple(arg_list)
                return None
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
            return tuple(out_dts) if len(out_dts) > 1 else out_dts[0]
        else:
            return self.wrap(local_results, output_sharding.output_spec)  # type: ignore[possibly-undefined]

    @staticmethod
    def redistribute_local_args(
        op_info: OpInfo,
        suggested_input_schema: OpSchema,
    ) -> None:
        # NOTE: it's very rare that we need to reshard kwargs so we intentionally skip it
        if op_info.args_tree_spec is not None:
            flatten_args_schema_to_reshard = tuple(
                pytree.tree_leaves(suggested_input_schema.args_schema)
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

    def unwrap_to_op_info(
        self,
        op_call: torch._ops.OpOverload,
        args: Tuple[object, ...],
        kwargs: Dict[str, object],
    ) -> OpInfo:
        # get runtime schema info to determine whether to use pytree to flatten inputs
        runtime_schema_info = self.sharding_propagator.op_to_schema_info.get(
            op_call, None
        )

        if runtime_schema_info is not None and runtime_schema_info.needs_pytree:
            # flatten args/kwargs when op says necessary
            tree_args, args_spec = pytree.tree_flatten(args)
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
                local_args.append(arg._local_tensor)
                if mesh is not None and mesh != arg.device_mesh:
                    # TODO: try replicate dtensor spec in missing dimension would work
                    # for most cases for foreach case except when the first DTensor in
                    # the list is one that also need to be replicated. We need to revisit
                    # how we want to handle this corner case. For now, this case would hit
                    # the cross mesh error even if implicit replication is turned on.
                    spec = self._try_replicate_dtensor_spec_in_missing_dim(
                        op_call, arg, mesh
                    )
                    args_schema.append(spec)
                else:
                    mesh = arg.device_mesh
                    args_schema.append(arg._spec)
            elif isinstance(arg, torch.Tensor):
                mesh = mesh or try_find_mesh_from_args(op_call, args_list)
                args_schema.append(
                    self._try_replicate_spec_for_scalar_tensor(op_call, arg, mesh)
                )
                local_args.append(arg)
            else:
                args_schema.append(arg)
                local_args.append(arg)

        for k, v in kwargs.items():
            if isinstance(v, dtensor.DTensor):
                local_kwargs[k] = v._local_tensor
                if mesh is not None and mesh != v.device_mesh:
                    spec = self._try_replicate_dtensor_spec_in_missing_dim(
                        op_call, v, mesh
                    )
                    kwargs_schema[k] = spec
                else:
                    mesh = v.device_mesh
                    kwargs_schema[k] = v._spec
            elif isinstance(v, torch.Tensor):
                mesh = mesh or try_find_mesh_from_args(op_call, args_list)
                kwargs_schema[k] = self._try_replicate_spec_for_scalar_tensor(
                    op_call, v, mesh
                )
                local_kwargs[k] = v
            else:
                kwargs_schema[k] = v
                local_kwargs[k] = v

        assert mesh is not None, f"found no DeviceMesh from dtensor args for {op_call}!"
        op_info = OpInfo(
            mesh,
            OpSchema(
                op_call,
                pytree.tree_unflatten(args_schema, args_spec)
                if args_spec
                else tuple(args_schema),
                kwargs_schema,
                schema_info=runtime_schema_info,
            ),
            args_schema,
            tuple(local_args),
            local_kwargs,
            args_spec,
        )
        return op_info

    @staticmethod
    def wrap(res: object, spec: OutputSpecType) -> object:
        if isinstance(res, torch.Tensor):
            if spec is not None:
                assert isinstance(
                    spec, DTensorSpec
                ), f"output spec does not match with output! Expected DTensorSpec, got {spec}."
                return dtensor.DTensor(res, spec, requires_grad=res.requires_grad)
            else:
                # if output does not have a DTensorSpec due to specific ops, it must be a scalar tensor
                assert res.ndim == 0, "output tensor should be scalar!"
                return res
        elif isinstance(res, (list, tuple)):
            assert spec is not None and isinstance(
                spec, (list, tuple)
            ), f"output spec does not match with output! Expected list/tuple, got {spec}."
            res_list = []
            for e, s in zip(res, spec):
                res_list.append(OpDispatcher.wrap(e, s))

            return tuple(res_list) if isinstance(res, tuple) else res_list
        else:
            # if the res contains only non tensor values (i.e. int/float/none), we simply return it
            # without rewrapping to DTensor.
            return res

    def _try_replicate_spec_for_scalar_tensor(
        self,
        op_call: torch._ops.OpOverload,
        tensor_arg: torch.Tensor,
        mesh: "DeviceMesh",
    ) -> DTensorSpec:
        # util function to produce a replicate spec for a scalar tensor arg/kwarg
        if tensor_arg.numel() == 1 and tensor_arg.ndim == 1:
            warnings.warn(
                "Found a non-scalar tensor with numel=1 and ndim!=0, "
                "we are implicitly creating a replicated DTensor for it. "
                "However, please consider changing it to a scalar tensor "
                "or explicitly create a DTensor under distributed enviroment."
            )

        if tensor_arg.numel() == 1 or self._allow_implicit_replication:
            # scalar tensor can be safely treated as replicated
            replication_spec = DTensorSpec(
                mesh,
                (Replicate(),) * mesh.ndim,
                tensor_meta=TensorMeta(
                    shape=tensor_arg.shape,
                    stride=tensor_arg.stride(),
                    dtype=tensor_arg.dtype,
                ),
            )
        else:
            raise RuntimeError(
                f"{op_call}: got mixed torch.Tensor and DTensor, need to convert all"
                " torch.Tensor to DTensor before calling distributed operators!"
            )
        return replication_spec

    def _try_replicate_dtensor_spec_in_missing_dim(
        self,
        op_call: torch._ops.OpOverload,
        dtensor_arg: "dtensor.DTensor",
        mesh: "DeviceMesh",
    ) -> DTensorSpec:
        # util function to produce a new spec for a DTensor arg/kwarg
        # that puts Replicate() placement in the missing dimension for foreach ops
        from torch.distributed.device_mesh import _mesh_resources

        cur_mesh = dtensor_arg.device_mesh
        root_mesh = _mesh_resources.get_root_mesh(cur_mesh)
        if (
            self._allow_implicit_replication
            and "foreach" in op_call.__name__
            and root_mesh == mesh
        ):
            placements = [Replicate() for _ in range(root_mesh.ndim)]
            cur_mesh_root_idx = _mesh_resources.get_root_mesh_dim(cur_mesh)
            placements[cur_mesh_root_idx] = dtensor_arg.placements[0]  # type: ignore[call-overload]
            replicate_spec = DTensorSpec(
                root_mesh,
                tuple(placements),
                tensor_meta=TensorMeta(
                    shape=dtensor_arg.shape,
                    stride=dtensor_arg.stride(),
                    dtype=dtensor_arg.dtype,
                ),
            )
        else:
            raise NotImplementedError(
                f"{op_call}: DTensor does not support cross-mesh operation yet! "
                f"Got meshes: {mesh} {cur_mesh}"
            )
        return replicate_spec
