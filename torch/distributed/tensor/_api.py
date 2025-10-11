# mypy: allow-untyped-decorators
# mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates
import inspect
import warnings
from collections.abc import Callable, Sequence
from typing import Any, cast, Optional, TypeAlias
from typing_extensions import deprecated

import torch
import torch.distributed.tensor._dispatch as op_dispatch
import torch.distributed.tensor._random as random
import torch.nn as nn
from torch._export.wrappers import mark_subclass_constructor_exportable_experimental
from torch.distributed.device_mesh import _mesh_resources, DeviceMesh
from torch.distributed.tensor._collective_utils import check_tensor_meta, mesh_broadcast
from torch.distributed.tensor._dtensor_spec import (
    DTensorSpec,
    ShardOrder,
    ShardOrderEntry,
    TensorMeta,
)
from torch.distributed.tensor._redistribute import (
    Redistribute,
    redistribute_local_tensor,
)
from torch.distributed.tensor._utils import (
    compute_global_tensor_info,
    compute_local_shape_and_global_offset,
    normalize_to_torch_size,
)
from torch.distributed.tensor.placement_types import (
    _StridedShard,
    Partial,
    Placement,
    Replicate,
    Shard,
)


__all__ = [
    "DTensor",
    "distribute_tensor",
    "distribute_module",
    "ones",
    "empty",
    "full",
    "rand",
    "randn",
    "zeros",
]

aten = torch.ops.aten


TensorShardingDict: TypeAlias = dict[int, Sequence[int | str]]


# NOTE [Autograd interaction between torch.Tensor]
#
# The autograd functions defined below are being used by the public
# facing APIs (i.e. from_local, to_local) to ensure DTensor to work
# together with torch.Tensor within the autograd engine. This
# allows DTensor to only exist on part of the module hierarchy.
#
# As an example, we have the a module that consists of submodules
# A, B, and C, the execution flow would be like:
#  input(torch.Tensor) -> Module A -> Module B -> Module C -> output (torch.Tensor)
#
# Suppose I only want to make Module B be a sharded module with
# DTensor params, the following forward/backward should work:
#
#  input(torch.Tensor) -> Module A
#       -> DTensor input (from_local) -> Sharded Module B -> DTensor output
#           -> torch.Tensor output (to_local) -> Module C
#
# So from_local/to_local must be Autograd functions.
#
class _ToTorchTensor(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        input: "DTensor",
        grad_placements: Optional[Sequence[Placement]],
    ):
        ctx.dtensor_spec = input._spec
        ctx.grad_placements = grad_placements
        local_tensor = input._local_tensor

        # We need to return a fresh Tensor object there as autograd metadata
        # will be inplaced into it. So we don't want to pollute the Tensor
        # object stored in the _local_tensor of this DTensor.
        return local_tensor.view_as(local_tensor)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        dtensor_spec = ctx.dtensor_spec
        mesh = dtensor_spec.mesh
        grad_placements = ctx.grad_placements
        dtensor_meta = dtensor_spec.tensor_meta

        _, tensor_stride = compute_global_tensor_info(
            grad_output, mesh, dtensor_spec.placements
        )
        tensor_stride = tuple(tensor_stride)
        grad_placements = grad_placements or dtensor_spec.placements
        grad_spec = DTensorSpec(
            mesh,
            grad_placements,
            tensor_meta=TensorMeta(
                shape=dtensor_meta.shape,
                stride=tensor_stride,
                dtype=dtensor_meta.dtype,
            ),
        )

        return (
            DTensor(
                grad_output,
                grad_spec,
                requires_grad=grad_output.requires_grad,
            ),
            None,
        )


class _FromTorchTensor(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,  # pyre-ignore[2]: Parameter must be annotated.
        input: torch.Tensor,
        device_mesh: DeviceMesh,
        placements: tuple[Placement, ...],
        run_check: bool,
        shape: Optional[torch.Size] = None,
        stride: Optional[tuple[int, ...]] = None,
    ) -> "DTensor":
        ctx.previous_placement = placements
        ctx.previous_device_mesh = device_mesh

        if shape and stride:
            tensor_shape, tensor_stride = shape, stride
        elif not shape and not stride:
            # if it's not by default run_check, we assume user is certain that each
            # rank has the same tensor shape, and we just use that to calculate the
            # global shape
            global_shape, global_stride = compute_global_tensor_info(
                input, device_mesh, placements
            )
            tensor_shape, tensor_stride = torch.Size(global_shape), tuple(global_stride)
        else:
            raise RuntimeError(
                f"Found shape:{shape}, stride:{stride}.",
                "Please pass both shape and stride at the same time.",
            )

        if device_mesh.get_coordinate() is None:
            # if the global rank is not participating in the device mesh, we
            # simply set the local tensor to an empty tensor
            input = input.new_empty(0, requires_grad=input.requires_grad)
        elif run_check:
            # TODO: support uneven sharding when global shape/stride not passed, by
            # building the global TensorMeta during check_tensor_meta
            check_shape_stride = not shape and not stride
            check_tensor_meta(input, check_shape_stride=check_shape_stride)
            # TODO: See if we need to make this run_check logic
            # have a corresponding backward.
            for idx, placement in enumerate(placements):
                if placement.is_replicate():
                    # broadcast rank 0 tensor to all ranks
                    # only broadcast if run_check is True
                    input = input.contiguous()
                    mesh_broadcast(input, device_mesh, mesh_dim=idx)

        dist_spec = DTensorSpec(
            device_mesh,
            placements,
            tensor_meta=TensorMeta(
                tensor_shape,
                tensor_stride,
                input.dtype,
            ),
        )

        # We want a fresh Tensor object that shares memory with the input tensor
        dist_tensor = DTensor(
            input.view_as(input),
            dist_spec,
            # requires_grad of the dist tensor depends on if input
            # requires_grad or not
            requires_grad=input.requires_grad,
        )
        return dist_tensor

    @staticmethod
    def backward(ctx, grad_output: "DTensor"):  # type: ignore[override]
        previous_placement = ctx.previous_placement
        previous_device_mesh = ctx.previous_device_mesh

        # reshard to the placement when creating DistributedTensor
        # so that the gradient layout matches, and we could return
        # local gradients directly
        if grad_output.placements != previous_placement:
            current_spec = grad_output._spec
            target_spec = DTensorSpec(
                previous_device_mesh,
                previous_placement,
                tensor_meta=grad_output._spec.tensor_meta,
            )
            local_tensor = grad_output._local_tensor
            output = redistribute_local_tensor(
                local_tensor, current_spec, target_spec, is_backward=True
            )
            # TODO: return the redistributed local tensor directly without
            # differentiable backward. see if this make sense for all cases.
            return output, None, None, None, None, None

        # TODO: backward is also differentiable now, add a test
        # to test higher level gradients.
        return grad_output.to_local(), None, None, None, None, None


def _prepare_placements_and_shard_order(
    device_mesh: DeviceMesh,
    tensor_rank: int,
    placements: Optional[Sequence[Placement]] = None,
    shard_order: Optional[TensorShardingDict] = None,
) -> tuple[
    tuple[Placement, ...],
    ShardOrder,
]:
    """
    This function places `placements` and `shard_order` in a redundant but
    canonical form, whereas in the input users can elide redundant information
    when specifying `placements`. For example, when there is never a tensor
    dimension that is sharded by multiple mesh dims, `shard_order` can be elided
    (this is traditional "PyTorch" style). Similarly, if a user specifies a
    `shard_order` that has entry for every device mesh dim, the `placements` can
    be inferred (this is traditional "JAX" style).

    You can also specify both arguments, which may be necessary in some
    situations as `placements` and `shard_order` have different expressivity.
    For example, to express that a device mesh is `Partial` or a non-standard
    sharding differently from Shard, you must use `placements` (as only it takes
    Placement). To express the order multiple shardings apply to a single tensor
    dimension, you must use `shard_order`. If you want to express both of these
    things, you will need to use both arguments.

    `placements` and `shard_order` must be consistent with each other. What this
    means is, when both are set, for each entry on `shard_order` i.e.,
    tensor_dim: mesh_dim, it must be the case that `Shard(tensor_dim)[...,
    mesh_dim, ...]` corresponds to a `placements[mesh_dim]` with the value
    `Shard(tensor_dim)`. For every entry on `placements`, if
    `placements[mesh_dim]` value is `Shard(tensor_dim)`, it must be the case
    either `shard_order` is None (no shard order was specified for `placements`,
    default ordering is fine) or `mesh_dim` in `shard_order[tensor_dim]` (it is
    not OK to omit mesh dims from an explicitly specified placement). A mesh dim
    mentioned in `shard_order` is always a Shard placement, no other placement
    types are supported.

    In the returned canonical representation, no information is omitted. In
    particular, `shard_order` is no longer a dict, it is a sparse tuple with
    each inner tuple corresponds to a tensor dimension as the first element and
    remaining elements as the indices of device mesh dimensions that this tensor
    dimension is sharded over. Check the "Returns" section for more details.

    Args:
        device_mesh (:class:`DeviceMesh`): DeviceMesh to place the tensor.
        tensor_rank (int): The rank (number of dimensions) of the tensor to be
            distributed or redistributed.
        placements (Sequence[:class:`Placement`], optional): The placements that
            describe how to place the local torch.Tensor on DeviceMesh. Must
            have the same number of elements as ``device_mesh.ndim``.
        shard_order (dict[int, Sequence[int | str]], optional):
            Specifies the mapping of tensor dimensions to the order of device mesh
            dimensions they are sharded over. It is a dictionary where keys are tensor
            dimensions and values are sequences of mesh dimensions (or mesh dimension names)
            that the tensor dimension is sharded across, in execution order.

            Internally, this is converted to a ShardOrder (tuple of ShardOrderEntry objects)
            where each ShardOrderEntry contains a tensor_dim and mesh_dims tuple.

            If not specified, a default left-to-right on the device mesh sharding
            order is used.

    Returns:
        Tuple:
            - placements (Tuple[:class:`Placement`, ...]): The computed
                placements as a tuple.
            - shard_order (ShardOrder): The computed shard order as a tuple
                of ShardOrderEntry objects. Each ShardOrderEntry contains:
                  * tensor_dim (int): The tensor dimension being sharded
                  * mesh_dims (tuple[int, ...]): The device mesh dimensions that
                    this tensor dimension is sharded across, in execution order.
                For example, ShardOrderEntry(tensor_dim=1, mesh_dims=(0, 2)) means
                tensor dimension 1 is sharded first over mesh dimension 0, then
                mesh dimension 2. If a tensor dimension is not sharded, it won't
                have a corresponding ShardOrderEntry in the tuple.

    Raises:
        ValueError: If the length of `placements` does not match
            `device_mesh.ndim`, if `shard_order` contains invalid tensor or
            mesh dimensions, if both normalized and un-normalized tensor_dim
            are specified in `shard_order`, or if a tensor_dim or mesh_dim in
            `shard_order` is out of range.
        RuntimeError: If attempting to redistribute from a non-Partial to a
            Partial placement, from one Partial type to a different Partial type,
            or use _StridedShard with `shard_order`.
        AssertionError: If a placement's shard dim normalization would result in
            a negative value, or if there is a conflict between placements and
            shard_order for sharding annotation.

    """

    def _from_dict_to_ShardOrder(
        shard_order_map: dict[int, list[int]],
    ) -> ShardOrder:
        sparse_shard_order = tuple(
            ShardOrderEntry(tensor_dim=key, mesh_dims=tuple(value))
            for key, value in sorted(shard_order_map.items())
            if value
        )
        return sparse_shard_order

    def _convert_shard_order_to_placements(
        shard_order_map: dict[int, Sequence[int]],
        device_mesh: DeviceMesh,
    ) -> tuple[Placement, ...]:
        # convert from shard_order to placements
        placements: list[Placement] = [Replicate() for _ in range(device_mesh.ndim)]
        for tensor_dim, mesh_dims in shard_order_map.items():
            for mesh_dim in mesh_dims:
                placements[mesh_dim] = Shard(tensor_dim)
        return tuple(placements)

    if placements is None and shard_order is None:
        placements = [Replicate() for _ in range(device_mesh.ndim)]

    if placements is not None and len(placements) != device_mesh.ndim:
        raise ValueError(
            f"`placements` must have the same length as `device_mesh.ndim`! "
            f"Found placements length: {len(placements)}, and device_mesh.ndim: {device_mesh.ndim}."
        )

    # normalize tensor dims in shard_order and convert str mesh dim into int
    normalized_shard_order: dict[int, list[int]] = {}
    has_stridedshard_in_placements = False

    if shard_order is not None:
        for tensor_dim, mesh_dims in shard_order.items():
            if tensor_dim < -tensor_rank or tensor_dim >= tensor_rank:
                raise ValueError(
                    f"tensor dim {tensor_dim} specified in `shard_order` is out of range for tensor_rank {tensor_rank}."
                )
            tensor_dim = tensor_dim + tensor_rank if tensor_dim < 0 else tensor_dim
            if tensor_dim in normalized_shard_order:
                raise ValueError(
                    f"both normalized tensor dim {tensor_dim} and un-normalized "
                    f"tensor dim {tensor_dim - tensor_rank}) is specified in `shard_order`!"
                )
            normalized_shard_order[tensor_dim] = []
            for mesh_dim in mesh_dims:
                if isinstance(mesh_dim, str):
                    mesh_dim = device_mesh._get_mesh_dim_by_name(mesh_dim)
                if mesh_dim < 0 or mesh_dim >= device_mesh.ndim:
                    raise IndexError(
                        f"mesh dim {mesh_dim} specified in `shard_order` is out of range "
                        f"for placements of length {device_mesh.ndim}"
                    )
                normalized_shard_order[tensor_dim].append(mesh_dim)

    # set default placements to replicated if not specified
    placement_tuple: tuple[Placement, ...]
    if placements is None:
        if shard_order is None:
            placement_tuple = tuple(Replicate() for _ in range(device_mesh.ndim))
        else:
            # convert from shard_order to placements
            placement_tuple = _convert_shard_order_to_placements(
                cast(dict[int, Sequence[int]], normalized_shard_order), device_mesh
            )
        shard_order_tuple = _from_dict_to_ShardOrder(normalized_shard_order)
    else:
        normalized_placements = list(placements)  # type: ignore[assignment]
        for i, placement in enumerate(placements):
            if placement.is_shard():
                tensor_dim = placement.dim  # type: ignore[attr-defined]
                if tensor_dim < -tensor_rank or tensor_dim >= tensor_rank:
                    raise ValueError(
                        f"tensor dim {tensor_dim} specified in `placements` is out of range for tensor_rank {tensor_rank}."
                    )
                tensor_dim = tensor_dim + tensor_rank if tensor_dim < 0 else tensor_dim

                # reconstruct `placement` object in case it is `_StridedShard` for backward compatibility
                if isinstance(placement, _StridedShard):
                    has_stridedshard_in_placements = True
                    normalized_placements[i] = _StridedShard(
                        tensor_dim, split_factor=placement.split_factor
                    )
                else:
                    normalized_placements[i] = Shard(tensor_dim)
        placement_tuple = tuple(normalized_placements)

        if shard_order is None:
            shard_order_tuple = DTensorSpec.compute_default_shard_order(placement_tuple)
        else:
            # both shard_order and placements are specified; need to validate their correctness
            if has_stridedshard_in_placements:
                # _StridedShard doesn't work with shard_order
                raise RuntimeError(
                    "Cannot specify both `placements` and `shard_order` when "
                    "`placements` contains `_StridedShard`!"
                )
            # check if user specified shard_order is valid
            placement_tuple_from_shard_order = _convert_shard_order_to_placements(
                cast(dict[int, Sequence[int]], normalized_shard_order), device_mesh
            )
            for original_mesh_dim, converted_mesh_dim in zip(
                placements, placement_tuple_from_shard_order
            ):
                if isinstance(original_mesh_dim, Shard):
                    assert (
                        isinstance(converted_mesh_dim, Shard)
                        and original_mesh_dim.dim == converted_mesh_dim.dim
                    ), (
                        f"Conflict sharding annotation for Shard {original_mesh_dim.dim} "
                        "detected between `placement` and `shard_order`."
                    )
                if isinstance(converted_mesh_dim, Shard):
                    assert isinstance(original_mesh_dim, Shard), (
                        f"Conflict sharding annotation detected for Shard {converted_mesh_dim.dim} "
                        "between `placement` and `shard_order`."
                    )
            shard_order_tuple = _from_dict_to_ShardOrder(normalized_shard_order)
    return placement_tuple, shard_order_tuple


class DTensor(torch.Tensor):
    """
    ``DTensor`` (Distributed Tensor) is a subclass of ``torch.Tensor`` that provides single-device like
    abstraction to program with multi-device ``torch.Tensor``. It describes the distributed tensor sharding
    layout (DTensor Layout) through the :class:`DeviceMesh` and following types of :class:`Placement`:

    * :class:`Shard`: Tensor sharded on the tensor dimension ``dim`` on the devices of the ``DeviceMesh`` dimension
    * :class:`Replicate`: Tensor replicated on the devices of the ``DeviceMesh`` dimension
    * :class:`Partial`: Tensor is pending reduction on the devices of the ``DeviceMesh`` dimension

    When calling PyTorch operators, ``DTensor`` overrides the PyTorch operators to perform sharded computation and issue
    communications whenever necessary. Along with the operator computation, ``DTensor`` will transform or propagate the
    placements (DTensor Layout) properly (based on the operator semantic itself) and generate new ``DTensor`` outputs.

    To ensure numerical correctness of the ``DTensor`` sharded computation when calling PyTorch operators, ``DTensor``
    requires every Tensor argument of the operator be DTensor.

    .. note:: Directly using the Tensor subclass constructor here is not the recommended way to create a ``DTensor``
        (i.e. it does not handle autograd correctly hence is not the public API). Please refer to the `create_dtensor`_
        section to see how to create a ``DTensor``.
    """

    _local_tensor: torch.Tensor
    _spec: DTensorSpec
    __slots__ = ["_local_tensor", "_spec"]

    # _op_dispatcher instance as a class attribute to handle runtime dispatching logic
    _op_dispatcher: op_dispatch.OpDispatcher = op_dispatch.OpDispatcher()

    # This implementation is just to convince mypy _spec and _local_tensor are
    # initialized; it is immediately overridden below.
    def __new__(
        cls,
        local_tensor: torch.Tensor,
        spec: DTensorSpec,
        *,
        requires_grad: bool,
    ) -> "DTensor":
        r = torch.Tensor._dtensor__new__(
            cls, local_tensor, spec, requires_grad=requires_grad
        )
        r._spec = spec
        r._local_tensor = local_tensor
        return r

    __new__ = torch.Tensor._dtensor__new__  # type: ignore[assignment] # noqa: F811

    @torch._disable_dynamo
    @mark_subclass_constructor_exportable_experimental
    def __init__(self, *args, **kwargs):
        """
        Construct a DTensor from a local tensor, device mesh, and placement and
        other tensor properties (i.e. shape, requires_grad, strides, etc).
        .. note:: This is not a public API and it's only supposed to be used by the
            operator implementations and internals. If you want to construct a
            DTensor from a local tensor, consider using ``DTensor.from_local``, if
            you want to construct a DTensor from a "global" tensor (where you
            already have tensor initialized and want to shard this tensor),
            consider using ``distribute_tensor``.
        """
        super().__init__()

    # pyre-fixme[14]: `__repr__` overrides method defined in `DTensor` inconsistently.
    # pyre-fixme[3]: Return type must be annotated.
    def __repr__(self):  # type: ignore[override]
        # TODO: consider all_gather the local tensors for better debugging
        return f"DTensor(local_tensor={self._local_tensor}, device_mesh={self._spec.mesh}, placements={self._spec.placements})"

    def __tensor_flatten__(self):
        """
        protocol to inform how to flatten a DTensor to local tensor
        for PT2 tracing
        """
        return ["_local_tensor"], (self._spec, self.requires_grad)

    @staticmethod
    def __tensor_unflatten__(inner_tensors, flatten_spec, outer_size, outer_stride):
        assert flatten_spec is not None, (
            "Expecting spec to be not None from `__tensor_flatten__` return value!"
        )
        local_tensor = inner_tensors["_local_tensor"]
        spec, requires_grad = flatten_spec
        unflatten_tensor_meta = TensorMeta(
            shape=outer_size,
            stride=outer_stride,
            dtype=spec.tensor_meta.dtype,
        )
        unflatten_spec = DTensorSpec(
            spec.mesh,
            spec.placements,
            tensor_meta=unflatten_tensor_meta,
        )
        return DTensor(
            local_tensor,
            unflatten_spec,
            requires_grad=requires_grad,
        )

    def __coerce_tangent_metadata__(self):
        if not any(isinstance(p, Partial) for p in self.placements):
            return self
        placements = [
            Replicate() if isinstance(p, Partial) else p for p in self.placements
        ]
        return self.redistribute(device_mesh=self.device_mesh, placements=placements)

    def __coerce_same_metadata_as_tangent__(self, flatten_spec, expected_type=None):
        if expected_type is not None:
            return None

        (spec, _) = flatten_spec  # Result of tensor_flatten()
        return self.redistribute(
            device_mesh=self.device_mesh,
            placements=spec.placements,
        )

    @classmethod
    @torch._disable_dynamo
    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):  # type: ignore[override]
        return DTensor._op_dispatcher.dispatch(
            func,
            args,
            kwargs or {},
        )

    # TODO(zpcore): support `shard_order` argument
    @staticmethod
    def from_local(
        local_tensor: torch.Tensor,
        device_mesh: Optional[DeviceMesh] = None,
        placements: Optional[Sequence[Placement]] = None,
        *,
        run_check: bool = False,
        shape: Optional[torch.Size] = None,
        stride: Optional[tuple[int, ...]] = None,
    ) -> "DTensor":
        """
        Create a :class:`DTensor` from a local torch.Tensor on each rank
        according to the ``device_mesh`` and ``placements`` specified.

        Args:
            local_tensor (torch.Tensor): local torch.Tensor on each rank.
            device_mesh (:class:`DeviceMesh`, optional): DeviceMesh to place the
                tensor, if not specified, must be called under a DeviceMesh
                context manager, default: None
            placements (List[:class:`Placement`], optional): the placements that
                describes how to place the local torch.Tensor on DeviceMesh, must
                have the same number of elements as ``device_mesh.ndim``.

        Keyword args:
            run_check (bool, optional): at a cost of extra communications, perform
                sanity check across ranks to check each local tensor's meta information
                to ensure correctness. If have :class:`Replicate` in ``placements``, the
                data on first rank of the device mesh dimension will be broadcasted
                to other ranks. default: False
            shape (torch.Size, optional): A List of int which specifies the size of
                DTensor which build on top of `local_tensor`. Note this needs to be
                provided if the shape of ``local_tensor`` are different across the ranks.
                If not provided, ``shape`` will be computed assuming the given distributed
                tensor is evenly sharded across ranks. default: None
            stride (tuple, optional): A List of int which specifies the stride of DTensor.
                If not provided, ``stride`` will be computed assuming the given distributed
                tensor is evenly sharded across ranks. default: None

        Returns:
            A :class:`DTensor` object

        Raises:
            RuntimeError: If both ``shape`` and ``stride`` are not provided together,
                or if the device mesh does not contain the current rank.

        .. note:: When ``run_check=False``, it is the user's responsibility to ensure the
            local tensor passed in is correct across ranks (i.e. the tensor is sharded for
            the ``Shard(dim)`` placement or replicated for the ``Replicate()`` placement).
            If not, the behavior of the created DTensor is undefined.

        .. note:: ``from_local`` is differentiable, the `requires_grad` of the created
            `DTensor` object will depend on if `local_tensor` requires_grad or not.
        """
        # `local_tensor` argument cannot be DTensor
        if isinstance(local_tensor, DTensor):
            raise RuntimeError(
                f"the local_tensor argument only accepts torch.Tensor but got {type(local_tensor)} value."
            )

        # if same shape/dtype, no need to run_check, if not, must allgather
        # the metadatas to check the size/dtype across ranks
        # There should be no data communication unless there's replication
        # strategy, where we broadcast the replication from the first rank
        # in the mesh dimension
        device_mesh = device_mesh or _mesh_resources.get_current_mesh()
        device_type = device_mesh.device_type

        # convert the local tensor to desired device base on device mesh's device_type
        if device_type != local_tensor.device.type and not local_tensor.is_meta:
            local_tensor = local_tensor.to(device_type)

        # set default placements to replicated if not specified
        if placements is None:
            placements = [Replicate() for _ in range(device_mesh.ndim)]
        else:
            placements = list(placements)
            for idx, placement in enumerate(placements):
                # normalize shard dim to be positive
                if placement.is_shard():
                    placement = cast(Shard, placement)
                    if placement.dim < 0:
                        placements[idx] = Shard(placement.dim + local_tensor.ndim)

        # `from_local` is differentiable, and the gradient of the dist tensor this function
        # created should flow back the gradients to the local_tensor, so we call an autograd
        # function to construct the dist tensor instead.
        return _FromTorchTensor.apply(  # pyre-ignore[16]: autograd func
            local_tensor,
            device_mesh,
            tuple(placements),
            run_check,
            shape,
            stride,
        )

    # TODO(zpcore): support `shard_order` argument for grad
    def to_local(
        self, *, grad_placements: Optional[Sequence[Placement]] = None
    ) -> torch.Tensor:
        """
        Get the local tensor of this DTensor on its current rank. For sharding it returns
        a local shard of the logical tensor view, for replication it returns the replica on
        its current rank.

        Keyword args:
            grad_placements (List[:class:`Placement`], optional): the placements describes
                the future layout of any gradient layout of the Tensor returned from this
                function.
                `to_local` converts DTensor to local tensor and the returned local tensor
                might not be used as the original DTensor layout later in the code. This
                argument is the hint that user can give to autograd in case the gradient
                layout of the returned tensor does not match the original DTensor layout.
                If not specified, we will assume the gradient layout remains the same
                as the original DTensor and use that for gradient computation.

        Returns:
            A :class:`torch.Tensor` or ``AsyncCollectiveTensor`` object. it represents the
            local tensor on its current rank. When an ``AsyncCollectiveTensor`` object is returned,
            it means the local tensor is not ready yet (i.e. communication is not finished). In this
            case, user needs to call ``wait`` to wait the local tensor to be ready.

        Raises:
            ValueError: If ``grad_placements`` has a different length than the device mesh dimensions.

        .. note:: ``to_local`` is differentiable, the ``requires_grad`` of the local tensor returned
            will depend on if the `DTensor` requires_grad or not.

        """
        if not torch.is_grad_enabled():
            return self._local_tensor

        if grad_placements is not None and not isinstance(grad_placements, tuple):
            grad_placements = tuple(grad_placements)
        return _ToTorchTensor.apply(
            self, grad_placements
        )  # pyre-ignore[16]: autograd func

    def redistribute(
        self,
        device_mesh: Optional[DeviceMesh] = None,
        placements: Optional[Sequence[Placement]] = None,
        shard_order: Optional[TensorShardingDict] = None,
        *,
        async_op: bool = False,
        forward_dtype: Optional[torch.dtype] = None,
        backward_dtype: Optional[torch.dtype] = None,
    ) -> "DTensor":
        """
        ``redistribute`` performs necessary collective operations that redistribute the current
        DTensor from its current placements to a new placements, or from its current DeviceMesh
        to a new DeviceMesh. i.e. we can turn a Sharded DTensor to a Replicated DTensor by
        specifying a Replicate placement for each dimension of the DeviceMesh.

        When redistributing from current to the new placements on one device mesh dimension, we
        will perform the following operations including communication collective or local operation:

        1. ``Shard(dim)`` -> ``Replicate()``: ``all_gather``
        2. ``Shard(src_dim)`` -> ``Shard(dst_dim)``: ``all_to_all``
        3. ``Replicate()`` -> ``Shard(dim)``: local chunking (i.e. ``torch.chunk``)
        4. ``Partial()`` -> ``Replicate()``: ``all_reduce``
        5. ``Partial()`` -> ``Shard(dim)``: ``reduce_scatter``


        ``redistribute`` would correctly figure out the necessary redistribute steps for DTensors
        that are created either on 1-D or N-D DeviceMesh.

        Args:
            device_mesh (:class:`DeviceMesh`, optional): DeviceMesh to place the
                DTensor. If not specified, it would use the current DTensor's DeviceMesh.
                default: None
            placements (List[:class:`Placement`], optional): the new placements that
                describes how to place the DTensor into the DeviceMesh, must
                have the same number of elements as ``device_mesh.ndim``.
                default: replicate on all mesh dimensions
            shard_order (dict[int, Sequence[int | str]], optional):
                Specifies the mapping of tensor dimensions to the order of device mesh
                dimensions they are sharded over. It is a dictionary where keys are tensor
                dimensions and values are sequences of mesh dimensions (or mesh dimension names)
                that the tensor dimension is sharded across, in execution order.

                If not specified, a default sharding order is used.

                Example: For a 3D tensor and a 4D device mesh, to shard tensor dimension 0 over
                mesh dim 1, and tensor dimension 1 over mesh dim 2 and then mesh dim 0, use
                ``shard_order = {0: [1], 1: [2, 0]}``. In this case, you can also specify the
                ``placements`` as [Shard(1), Shard(0), Shard(1), Replicate()] along with ``shard_order``.

        Keyword args:
            async_op (bool, optional): whether to perform the DTensor redistribute operation
                asynchronously or not. Default: False
            forward_dtype (torch.dtype, optional): the local tensor datatype can be converted to
                ``forward_dtype`` before redistributing the local tensor in its forward.
                The result DTensor will be in ``forward_dtype`` Default: None.
            backward_dtype (torch.dtype, optional): the local tensor datatype can be converted to
                ``backward_dtype`` before redistributing the local tensor in its backward.
                The result DTensor gradient would be converted back to the current DTensor dtype. Default: None

        Returns:
            A :class:`DTensor` object

        Raises:
            RuntimeError: If attempting to redistribute from a non-Partial to a Partial placement,
                or from one Partial type to a different Partial type.
            ValueError: If ``placements`` has a different length than the device mesh dimensions,
                if ``shard_order`` contains invalid tensor or mesh dimensions, or if both normalized and un-normalized
                tensor_dim are specified in ``shard_order``, or if a tensor_dim in ``shard_order`` is out of range.
            IndexError: If a mesh_dim specified in ``shard_order`` is out of range for the device mesh.

        .. note:: ``redistribute`` is differentiable, which means user do not need to worry about
            the backward formula of the redistribute operation.

        .. note:: ``redistribute`` currently only supports redistributing DTensor on the same DeviceMesh,
            Please file an issue if you need to redistribute DTensor to different DeviceMesh.

        """
        # NOTE: This redistribute API currently only supports out
        # of place redistribution, i.e. it always create a new
        # DTensor object and leave the original one unchanged.

        # if device_mesh is not specified, use the current device_mesh
        device_mesh = device_mesh or self.device_mesh

        # handle the special case where `Partial` is allowed if we are redistributing to
        # the same type of `Partial`
        if placements is not None:
            for i, placement in enumerate(placements):
                if placement.is_partial() and self.placements[i] != placement:
                    raise RuntimeError(
                        f"Can not redistribute from {self.placements[i]} to {placement}, "
                        "redistributing to Partial is for internal use only!"
                    )

        placements_tuple, shard_order_tuple = _prepare_placements_and_shard_order(
            device_mesh, self.ndim, placements, shard_order
        )
        # pyre-fixme[16]: `Redistribute` has no attribute `apply`.
        return Redistribute.apply(
            self,
            device_mesh,
            placements_tuple,
            shard_order_tuple,
            async_op,
            forward_dtype,
            backward_dtype,
        )

    def full_tensor(
        self, *, grad_placements: Optional[Sequence[Placement]] = None
    ) -> torch.Tensor:
        """
        Return the full tensor of this DTensor. It will perform necessary collectives
        to gather the local tensors from other ranks in its DeviceMesh and concatenate
        them together. It's a syntactic sugar of the following code:

        ``dtensor.redistribute(placements=[Replicate()] * mesh.ndim).to_local()``

        Keyword args:
            grad_placements (List[:class:`Placement`], optional): the placements describes
                the future layout of any gradient layout of the full Tensor returned from this
                function.
                `full_tensor` converts DTensor to a full torch.Tensor and the returned torch.tensor
                might not be used as the original replicated DTensor layout later in the code. This
                argument is the hint that user can give to autograd in case the gradient
                layout of the returned tensor does not match the original replicated DTensor layout.
                If not specified, we will assume the gradient layout of the full tensor be replicated.

        Returns:
            A :class:`torch.Tensor` object that represents the full tensor of this DTensor.

        .. note:: ``full_tensor`` is differentiable.
        """

        redist_res = self.redistribute(
            placements=[Replicate()] * self.device_mesh.ndim, async_op=False
        )
        return _ToTorchTensor.apply(redist_res, grad_placements)  # type: ignore[return-value]

    @property
    def device_mesh(self) -> DeviceMesh:
        """
        The :class:`DeviceMesh` attribute that associates with this DTensor object.

        .. note:: ``device_mesh`` is a read-only property, it can not be set.
        """
        return self._spec.mesh

    @property
    def placements(self) -> tuple[Placement, ...]:
        """
        The placements attribute of this DTensor that describes the layout of this
        DTensor on the its DeviceMesh.

        .. note:: ``placements`` is a read-only property, it can not be set.
        """
        return self._spec.placements

    def _raise_if_contains_partial_placements(self) -> None:
        """
        Raise an error if the DTensor contains partial placements.
        """
        for placement in self._spec.placements:
            if not isinstance(placement, Partial):
                continue

            raise ValueError(
                "Any checkpointing related operations are not supported for "
                "DTensor with partial placements!"
            )

    @property
    def shard_order(self) -> ShardOrder:
        """
        When a tensor dimension is sharded across multiple mesh axes,
        ``shard_order`` specifies the sequence in which these shardings are
        applied. This order determines how tensor shards are mapped and
        distributed across devices.

        ``shard_order`` is a :class:`ShardOrder` (tuple of :class:`ShardOrderEntry` objects),
        where each ``ShardOrderEntry`` is a named tuple containing:

        * ``tensor_dim`` (int): The tensor dimension being sharded
        * ``mesh_dims`` (tuple[int, ...]): The device mesh dimensions across which
          this tensor dimension is sharded, in execution order (first element is
          applied first, second element second, etc.)

        Example:
            For a tensor of shape [8, 16] and a 3D device mesh, if dim 0 is
            sharded over mesh dim 1, and dim 1 is sharded over mesh dim 0 and
            then mesh dim 2, the shard_order would be::

                shard_order = (
                    ShardOrderEntry(tensor_dim=0, mesh_dims=(1,)),
                    ShardOrderEntry(tensor_dim=1, mesh_dims=(0, 2)),
                )

        .. note:: ``shard_order`` is a read-only property, it can not be set.
        """
        return self._spec.shard_order

    def __create_write_items__(self, fqn: str, object: Any):
        self._raise_if_contains_partial_placements()
        from torch.distributed.checkpoint.planner_helpers import (
            _create_write_items_for_dtensor,
        )

        if hasattr(self._local_tensor, "__create_write_items__"):
            return self._local_tensor.__create_write_items__(fqn, object)  # type: ignore[attr-defined]
        elif isinstance(self._local_tensor, torch.Tensor):
            return [_create_write_items_for_dtensor(fqn, object)]
        else:
            raise RuntimeError("Unsupported tensor type!")

    def __create_chunk_list__(self):
        """
        Return a list of ChunkStorageMetadata, which is a dataclass that describes the size/offset of the local shard/replica
        on current rank. For DTensor, each rank will have a single local shard/replica, so the returned list usually only
        has one element.

        This dunder method is primariy used for distributed checkpoint purpose.

        Returns:
            A List[:class:`ChunkStorageMetadata`] object that represents the shard size/offset on the current rank.
        """
        self._raise_if_contains_partial_placements()
        from torch.distributed.checkpoint.planner_helpers import (
            _create_chunk_from_dtensor,
        )

        if hasattr(self._local_tensor, "__create_chunk_list__"):
            return self._local_tensor.__create_chunk_list__()  # type: ignore[attr-defined]
        elif isinstance(self._local_tensor, torch.Tensor):
            return [_create_chunk_from_dtensor(self)]
        else:
            raise RuntimeError("Unsupported tensor type!")

    def __get_tensor_shard__(self, index):
        self._raise_if_contains_partial_placements()
        if hasattr(self._local_tensor, "__get_tensor_shard__"):
            return self._local_tensor.__get_tensor_shard__(index)  # type: ignore[attr-defined]
        elif isinstance(self._local_tensor, torch.Tensor):
            return self.to_local()
        else:
            raise RuntimeError("Unsupported tensor type!")

    @classmethod
    def __metadata_guard__(
        cls, orig: tuple[DTensorSpec, bool], other: tuple[DTensorSpec, bool]
    ) -> bool:
        orig_spec, orig_requires_grad = orig
        other_spec, other_requires_grad = other
        return (
            orig_spec._check_equals(other_spec, skip_shapes=True)
            and orig_requires_grad == other_requires_grad
        )


def distribute_tensor(
    tensor: torch.Tensor,
    device_mesh: Optional[DeviceMesh] = None,
    placements: Optional[Sequence[Placement]] = None,
    shard_order: Optional[TensorShardingDict] = None,
    *,
    src_data_rank: Optional[int] = 0,
) -> DTensor:
    """
    Distribute a leaf ``torch.Tensor`` (i.e. nn.Parameter/buffers) to the ``device_mesh`` according
    to the ``placements`` specified. The rank of ``device_mesh`` and ``placements`` must be the
    same. The ``tensor`` to distribute is the logical or "global" tensor, and the API would use
    the ``tensor`` from first rank of the DeviceMesh dimension as the source of truth to preserve
    the single-device semantic. If you want to construct a DTensor in the middle of the Autograd
    computation, please use :meth:`DTensor.from_local` instead.

    Args:
        tensor (torch.Tensor): torch.Tensor to be distributed. Note that if you
            want to shard a tensor on a dimension that is not evenly divisible by
            the number of devices in that mesh dimension, we use ``torch.chunk``
            semantic to shard the tensor and scatter the shards. The uneven sharding
            behavior is experimental and subject to change.
        device_mesh (:class:`DeviceMesh`, optional): DeviceMesh to distribute the
            tensor, if not specified, must be called under a DeviceMesh context
            manager, default: None
        placements (Sequence[:class:`Placement`], optional): the placements that
            describes how to place the tensor on DeviceMesh, must have the same
            number of elements as ``device_mesh.ndim``. If not specified, we will
            by default replicate the tensor across the ``device_mesh`` from the
            first rank of each dimension of the ``device_mesh``.
        shard_order (dict[int, Sequence[int | str]], optional):
            Specifies the mapping of tensor dimensions to the order of device mesh
            dimensions they are sharded over. It is a dictionary where keys are tensor
            dimensions and values are sequences of mesh dimensions (or mesh dimension names)
            that the tensor dimension is sharded across, in execution order.

            If not specified, a default sharding order is used.

            Example: For a 3D tensor and a 4D device mesh, to shard tensor dimension
            0 over mesh dim 1, and tensor dimension 1 over mesh dim 2 and then mesh dim 0,
            use ``shard_order = {0: [1], 1: [2, 0]}``. In this case, you can also specify the
            ``placements`` as [Shard(1), Shard(0), Shard(1), Replicate()] along with ``shard_order``.

            Note: As long as there are no Partial placements (though this should
            not happen as of today), you may specify only shard_order (and not
            placements if device order is not the default left-to-right), and the
            correct placements will be inferred.

    Keyword args:
        src_data_rank (int, optional): the rank of the source data for the logical/global tensor, it is
            used by :meth:`distribute_tensor` to scatter/broadcast the shards/replicas to other ranks.
            By default, we use ``group_rank=0`` on each DeviceMesh dimension as the source data to preserve
            the single-device semantic. If passing ``None`` explicitly, :meth:`distribute_tensor` simply uses
            its local data instead of trying to preserve the single-device semantic via scatter/broadcast.
            Default: 0

    Returns:
        A :class:`DTensor` or ``XLAShardedTensor`` object.

    .. note::
        When initialize the DeviceMesh with the ``xla`` device_type, ``distribute_tensor``
        return `XLAShardedTensor` instead. see `this issue <https://github.com/pytorch/pytorch/issues/92909>`__
        for more details. The XLA integration is experimental and subject to change.
    """

    torch._C._log_api_usage_once("torch.dtensor.distribute_tensor")

    # get default device mesh if there's nothing specified
    device_mesh = device_mesh or _mesh_resources.get_current_mesh()
    device_type = device_mesh.device_type

    if placements is not None:
        for placement in placements:
            if placement.is_partial():
                raise RuntimeError(
                    f"Can not distribute to {placements}, "
                    "redistributing to Partial is for internal use only!"
                )

    placements_tuple, shard_order_tuple = _prepare_placements_and_shard_order(
        device_mesh, tensor.ndim, placements, shard_order
    )

    if device_type == "xla":
        try:
            # call PyTorch/XLA SPMD for `xla` backend type device mesh.
            # This returns XLAShardedTensor
            from torch_xla.distributed.spmd import (  # type:ignore[import]
                xla_distribute_tensor,
            )

            # TODO: use `shard_order_tuple` instead of `placements_tuple` for XLA
            return xla_distribute_tensor(tensor, device_mesh, placements_tuple)  # type:ignore[return-value]
        except ImportError as e:
            msg = "To use DTensor API with xla, you must install the torch_xla package!"
            raise ImportError(msg) from e

    if not tensor.is_leaf:
        raise RuntimeError(
            "`distribute_tensor` should be used to distribute leaf tensors! but found non-leaf tensor!"
        )

    # convert tensor to the corresponding device type if it's not in that device type
    if device_type != tensor.device.type and not tensor.is_meta:
        tensor = tensor.to(device_type)

    if isinstance(tensor, DTensor):
        # if the tensor is already a DTensor, we need to check:
        # 1. if the we can further shard this DTensor if the two device mesh belong to
        #   the same parenet mesh and further sharding is possible.
        # 2. check if device mesh and placements are the same
        if tensor.device_mesh != device_mesh:
            raise ValueError(
                f"Cannot distribute a DTensor with device mesh {tensor.device_mesh} "
                f"to a different device mesh {device_mesh}."
            )
        # TODO(zpcore): make sure the shard_order also matches.
        if tensor.placements != placements_tuple:
            raise ValueError(
                f"Cannot distribute a DTensor with placements {tensor.placements} "
                f"to a different placements {placements_tuple}. do you want to call "
                f"`redistribute` instead?"
            )
        return tensor

    local_tensor = tensor.detach()

    use_strided_shard = placements is not None and any(
        isinstance(p, _StridedShard) for p in placements
    )

    if use_strided_shard:
        # keep original code for backward compatibility considering
        # _StridedShard case
        assert shard_order is None, "shard_order conflicts with _StridedShard"
        for mesh_dim, placement in enumerate(placements_tuple):
            if placement.is_shard():
                assert placement.dim >= 0  # type: ignore[attr-defined]
                placement = cast(Shard, placement)
                local_tensor = placement._shard_tensor(
                    local_tensor, device_mesh, mesh_dim, src_data_rank
                )
            elif placement.is_replicate():
                local_tensor = Replicate._make_replicate_tensor(
                    local_tensor, device_mesh, mesh_dim, src_data_rank
                )
            else:
                raise RuntimeError(
                    f"Trying to distribute tensor with unsupported placements {placement} on device mesh dimension {mesh_dim}!"
                )
    else:
        replicate_on_mesh_dims = set(range(device_mesh.ndim))
        for entry in shard_order_tuple:
            tensor_dim = entry.tensor_dim
            mesh_dims = entry.mesh_dims
            for mesh_dim in mesh_dims:
                assert isinstance(mesh_dim, int)
                replicate_on_mesh_dims.remove(mesh_dim)
                local_tensor = Shard._make_shard_tensor(
                    tensor_dim, local_tensor, device_mesh, mesh_dim, src_data_rank
                )
        for mesh_dim in replicate_on_mesh_dims:
            local_tensor = Replicate._make_replicate_tensor(
                local_tensor, device_mesh, mesh_dim, src_data_rank
            )

    assert local_tensor is not None, "distributing a tensor should not be None"
    # detach the local tensor passed to DTensor since after the construction
    # of DTensor, autograd would work on top of DTensor instead of local tensor
    spec = DTensorSpec(
        mesh=device_mesh,
        placements=placements_tuple,
        shard_order=shard_order_tuple,
        tensor_meta=TensorMeta(
            shape=tensor.size(),
            stride=tensor.stride(),
            dtype=tensor.dtype,
        ),
    )

    return DTensor(
        local_tensor.requires_grad_(tensor.requires_grad),
        spec,
        requires_grad=tensor.requires_grad,
    )


@deprecated("Please use `distribute_tensor` with `src_data_rank=None` instead.")
def _shard_tensor(
    full_tensor: torch.Tensor,
    placements: Sequence[Shard],
    device_mesh: Optional[DeviceMesh] = None,
) -> "DTensor":
    """
    Locally shards a full tensor based on indicated sharding arrangement, and
    returns a DTensor containing the local shard.

    .. warning:: This is a private API that is subject to change. It skips the
        communication otherwise required by `distribute_tensor`. It is only
        applicable to cases where all ranks have the same `full_tensor`. For
        example, in distributed inference all ranks load from the same
        checkpoint. This API will not check for data equality between ranks, it
        is thus user's responsibility to ensure the `full_tensor` is the same
        across ranks.

    Args:
        full_tensor (torch.Tensor): the full tensor to be sharded.
        placements (Sequence[:class:`Shard`]): the placements that
            describes how to place the local tensor on DeviceMesh.
        device_mesh (:class:`DeviceMesh`, optional): DeviceMesh to place the
            DTensor.  Must have same dimension as the number of placements.
            If not specified, would be retrieve from current context.

    Returns:
        A :class:`DTensor` object with the shard as its local tensor.

    Examples:
        >>> # xdoctest: +SKIP("need world_size and rank")
        >>> device_mesh = torch.distributed.init_device_mesh("cuda", (world_size,))
        >>> full_tensor = torch.arange(world_size, device=f"cuda:{rank}")
        >>> dtensor = _shard_tensor(full_tensor, [Shard(1)], device_mesh)
    """
    return distribute_tensor(full_tensor, device_mesh, placements, src_data_rank=None)


def distribute_module(
    module: nn.Module,
    device_mesh: Optional[DeviceMesh] = None,
    partition_fn: Optional[Callable[[str, nn.Module, DeviceMesh], None]] = None,
    input_fn: Optional[Callable[[nn.Module, Any, DeviceMesh], None]] = None,
    output_fn: Optional[Callable[[nn.Module, Any, DeviceMesh], None]] = None,
) -> nn.Module:
    """
    This function expose three functions to control the parameters/inputs/outputs of the module:

    1. To perform sharding on the module before runtime execution by specifying the
    ``partition_fn`` (i.e. allow user to convert Module parameters to :class:`DTensor`
    parameters according to the `partition_fn` specified).
    2. To control the inputs or outputs of the module during runtime execution by
    specifying the ``input_fn`` and ``output_fn``. (i.e. convert the input to
    :class:`DTensor`, convert the output back to ``torch.Tensor``)

    Args:
        module (:class:`nn.Module`): user module to be partitioned.
        device_mesh (:class:`DeviceMesh`): the device mesh to place the module.
        partition_fn (Callable): the function to partition parameters (i.e. shard certain
            parameters across the ``device_mesh``). If ``partition_fn`` is not specified,
            by default we replicate all module parameters of ``module`` across the mesh.
        input_fn (Callable): specify the input distribution, i.e. could control how the
            input of the module is sharded. ``input_fn`` will be installed as a module
            ``forward_pre_hook`` (pre forward hook).
        output_fn (Callable): specify the output distribution, i.e. could control how the
            output is sharded, or convert it back to torch.Tensor. ``output_fn`` will be
            installed as a module ``forward_hook`` (post forward hook).

    Returns:
        A module that contains parameters/buffers that are all ``DTensor`` s.

    .. note::
        When initialize the DeviceMesh with the ``xla`` device_type, ``distribute_module``
        return nn.Module with PyTorch/XLA SPMD annotated parameters. See
        `this issue <https://github.com/pytorch/pytorch/issues/92909>`__
        for more details. The XLA integration is experimental and subject to change.

    """

    torch._C._log_api_usage_once("torch.dtensor.distribute_module")

    already_distributed = getattr(module, "_distribute_module_applied", False)
    if already_distributed:
        raise RuntimeError(
            "distribute_module should only be called once on a module, "
            "but it has already been called on this module!"
        )

    device_mesh = device_mesh or _mesh_resources.get_current_mesh()
    device_type = device_mesh.device_type
    if device_type == "xla":
        try:
            # This function annotates all module parameters for auto-partitioning with
            # PyTorch/XLA SPMD or explicitly partition to :class:`XLAShardedTensor` parameters
            # according to the `partition_fn` specified.
            from torch_xla.distributed.spmd import (  # type:ignore[import]
                xla_distribute_module,
            )

            return xla_distribute_module(
                module, device_mesh, partition_fn, input_fn, output_fn
            )  # type:ignore[return-value]
        except ImportError as e:
            msg = "To use DTensor API with xla, you must install the torch_xla package!"
            raise ImportError(msg) from e

    def replicate_module_params_buffers(m: nn.Module, mesh: DeviceMesh) -> None:
        # This function loop over the immediate module parameters and
        # buffers, replicate all non DTensor params/buffers to DTensor
        # parameters/buffers, if they have not been partitioned in the
        # partition_fn, we can't easily use `module._apply` here
        # because we don't know what happened inside partition_fn as
        # user could do anything, i.e. install hooks, and we want to
        # preserve those.
        full_replicate = [Replicate()] * mesh.ndim
        for key, param in m._parameters.items():
            if param is not None and not isinstance(param, DTensor):
                m.register_parameter(
                    key,
                    nn.Parameter(distribute_tensor(param.data, mesh, full_replicate)),
                )
        for key, buffer in m._buffers.items():
            if buffer is not None and not isinstance(buffer, DTensor):
                m._buffers[key] = distribute_tensor(buffer, mesh, full_replicate)

    if partition_fn is None:
        # if partition_fn not specified, we by default replicate
        # all module params/buffers
        for name, submod in module.named_modules():
            replicate_module_params_buffers(submod, device_mesh)
    else:
        # apply partition_fun to submodules
        for name, submod in module.named_modules():
            partition_fn(name, submod, device_mesh)
            replicate_module_params_buffers(submod, device_mesh)

    # register input_fn as module forward pre hook
    if input_fn is not None:
        # check the input_fn signature
        num_args = len(inspect.signature(input_fn).parameters)
        if num_args == 2:
            # input_fn only takes in inputs and device mesh
            warnings.warn(
                "Deprecating input_fn that takes two arguments (inputs, device_mesh), "
                "please use input_fn that takes in (module, inputs, device_mesh) instead!",
                FutureWarning,
                stacklevel=2,
            )
            module.register_forward_pre_hook(
                lambda _, inputs: input_fn(inputs, device_mesh)  # type: ignore[call-arg]
            )
        elif num_args == 3:
            # input_fn takes in module, inputs, device mesh
            module.register_forward_pre_hook(
                lambda mod, inputs: input_fn(mod, inputs, device_mesh)
            )
        else:
            raise ValueError(
                f"input_fn should take in 3 arguments, but got {num_args} arguments!"
            )
    # register output_fn as module forward hook
    if output_fn is not None:
        num_args = len(inspect.signature(output_fn).parameters)
        if num_args == 2:
            # output_fn only takes in outputs and device mesh
            warnings.warn(
                "Deprecating output_fn that takes two arguments (inputs, device_mesh), "
                "please use output_fn that takes in (module, inputs, device_mesh) instead!",
                FutureWarning,
                stacklevel=2,
            )
            module.register_forward_hook(
                lambda mod, inputs, outputs: output_fn(outputs, device_mesh)  # type: ignore[call-arg]
            )
        elif num_args == 3:
            module.register_forward_hook(
                lambda mod, inputs, outputs: output_fn(mod, outputs, device_mesh)
            )
        else:
            raise ValueError(
                f"output_fn should take in 3 arguments, but got {num_args} arguments!"
            )

    module._distribute_module_applied = True  # type: ignore[assignment]
    return module


# Below are tensor factory function APIs, which are used to create a DTensor directly. We need
# to make separate factory function APIs because tensor subclass could not override the tensor
# factory methods, and we need user to call the factory functions with user intended device_mesh
# and placements to create a proper DTensor.


def _dtensor_init_helper(  # type: ignore[no-untyped-def]
    init_op,
    size: torch.Size,
    device_mesh: Optional[DeviceMesh] = None,
    placements: Optional[Sequence[Placement]] = None,
    **kwargs,
) -> DTensor:
    # if device_mesh is None, use the one from mesh resources
    device_mesh = device_mesh or _mesh_resources.get_current_mesh()
    kwargs["device"] = device_mesh.device_type

    # set default placements to replicated if not specified
    placements = placements or tuple(Replicate() for _ in range(device_mesh.ndim))

    # check device_mesh against placements
    assert device_mesh.ndim == len(placements), (
        "mesh dimension does not match the length of placements"
    )

    assert kwargs["layout"] == torch.strided, "layout value not supported!"
    torch_stride = torch._prims_common.make_contiguous_strides_for(size)

    # get local tensor shape
    local_shape, _ = compute_local_shape_and_global_offset(
        size, device_mesh, placements
    )

    # initialize the local tensor
    if init_op == torch.full:
        fill_value = kwargs.pop("fill_value", 0)
        local_tensor = init_op(local_shape, fill_value, **kwargs)
    elif init_op == torch.rand or init_op == torch.randn:
        # this tensor meta is not used except `shape`
        dtype = kwargs.get("dtype", torch.get_default_dtype())

        tensor_meta = TensorMeta(size, (0,), dtype)
        spec = DTensorSpec(device_mesh, tuple(placements), tensor_meta=tensor_meta)

        if random.is_rng_supported_mesh(device_mesh) and not random._rng_tracker:
            random._rng_tracker = random.OffsetBasedRNGTracker(device_mesh)

        assert random._rng_tracker is not None
        with random._rng_tracker._distribute_region(spec):
            local_tensor = init_op(local_shape, **kwargs)
    else:
        local_tensor = init_op(local_shape, **kwargs)

    spec = DTensorSpec(
        device_mesh,
        tuple(placements),
        tensor_meta=TensorMeta(
            size,
            torch_stride,
            local_tensor.dtype,
        ),
    )

    return DTensor(
        local_tensor,
        spec,
        requires_grad=kwargs["requires_grad"],
    )


def ones(  # type: ignore[no-untyped-def]
    *size,
    dtype: Optional[torch.dtype] = None,
    layout: torch.layout = torch.strided,
    requires_grad: bool = False,
    device_mesh: Optional[DeviceMesh] = None,
    placements: Optional[Sequence[Placement]] = None,
) -> DTensor:
    """
    Returns a :class:`DTensor` filled with the scalar value 1, with the shape defined
    by the variable argument ``size``.

    Args:
        size (int...): a sequence of integers defining the shape of the output :class:`DTensor`.
            Can be a variable number of arguments or a collection like a list or tuple.
            E.g.: ones(1,2,3..) or ones([1,2,3..]) or ones((1,2,3..))

    Keyword args:
        dtype (:class:`torch.dtype`, optional): the desired data type of returned :class:`DTensor`.
            Default: if ``None``, uses a global default (see :func:`torch.set_default_dtype`).
        layout (:class:`torch.layout`, optional): the desired layout of returned DTensor.
            Default: ``torch.strided``.
        requires_grad (bool, optional): If autograd should record operations on the
            returned :class:`DTensor`. Default: ``False``.
        device_mesh: :class:`DeviceMesh` type, contains the mesh info of ranks
        placements: a sequence of :class:`Placement` type: ``Shard``, ``Replicate``

    Returns:
        A :class:`DTensor` object on each rank
    """
    torch_size = normalize_to_torch_size(size)

    return _dtensor_init_helper(
        torch.ones,
        torch_size,
        dtype=dtype,
        layout=layout,
        requires_grad=requires_grad,
        device_mesh=device_mesh,
        placements=placements,
    )


def empty(  # type: ignore[no-untyped-def]
    *size,
    dtype: Optional[torch.dtype] = None,
    layout: torch.layout = torch.strided,
    requires_grad: bool = False,
    device_mesh: Optional[DeviceMesh] = None,
    placements: Optional[Sequence[Placement]] = None,
) -> DTensor:
    """
    Returns a :class:`DTensor` filled with uninitialized data. The shape of the :class:`DTensor`
    is defined by the variable argument ``size``.

    Args:
        size (int...): a sequence of integers defining the shape of the output :class:`DTensor`.
            Can be a variable number of arguments or a collection like a list or tuple.
            E.g.: empty(1,2,3..) or empty([1,2,3..]) or empty((1,2,3..))

    Keyword args:
        dtype (:class:`torch.dtype`, optional): the desired data type of returned :class:`DTensor`.
            Default: if ``None``, uses a global default (see :func:`torch.set_default_dtype`).\
        layout (:class:`torch.layout`, optional): the desired layout of returned :class:`DTensor`.
            Default: ``torch.strided``.
        requires_grad (bool, optional): If autograd should record operations on the
            returned :class:`DTensor`. Default: ``False``.
        device_mesh: :class:`DeviceMesh` type, contains the mesh info of ranks
        placements: a sequence of :class:`Placement` type: ``Shard``, ``Replicate``

    Returns:
        A :class:`DTensor` object on each rank
    """
    torch_size = normalize_to_torch_size(size)

    return _dtensor_init_helper(
        torch.empty,
        torch_size,
        dtype=dtype,
        layout=layout,
        requires_grad=requires_grad,
        device_mesh=device_mesh,
        placements=placements,
    )


def full(  # type: ignore[no-untyped-def]
    size,
    fill_value,
    *,
    dtype: Optional[torch.dtype] = None,
    layout: torch.layout = torch.strided,
    requires_grad: bool = False,
    device_mesh: Optional[DeviceMesh] = None,
    placements: Optional[Sequence[Placement]] = None,
) -> DTensor:
    """
    Returns a :class:`DTensor` filled with ``fill_value`` according to ``device_mesh`` and
    ``placements``, with the shape defined by the argument ``size``.

    Args:
        size (int...): a sequence of integers defining the shape of the output :class:`DTensor`.
            Can be a variable number of arguments or a collection like a list or tuple.
            E.g.: ones(1,2,3..) or ones([1,2,3..]) or ones((1,2,3..))
        fill_value(Scalar): the value to fill the output tensor with.

    Keyword args:
        dtype (:class:`torch.dtype`, optional): the desired data type of returned :class:`DTensor`.
            Default: if ``None``, uses a global default (see :func:`torch.set_default_dtype`).
        layout (:class:`torch.layout`, optional): the desired layout of returned DTensor.
            Default: ``torch.strided``.
        requires_grad (bool, optional): If autograd should record operations on the
            returned :class:`DTensor`. Default: ``False``.
        device_mesh: :class:`DeviceMesh` type, contains the mesh info of ranks.
        placements: a sequence of :class:`Placement` type: ``Shard``, ``Replicate``

    Returns:
        A :class:`DTensor` object on each rank
    """
    torch_size = normalize_to_torch_size(size)

    return _dtensor_init_helper(
        torch.full,
        torch_size,
        fill_value=fill_value,
        dtype=dtype,
        layout=layout,
        requires_grad=requires_grad,
        device_mesh=device_mesh,
        placements=placements,
    )


def rand(  # type: ignore[no-untyped-def]
    *size,
    requires_grad: bool = False,
    dtype: Optional[torch.dtype] = None,
    layout: torch.layout = torch.strided,
    device_mesh: Optional[DeviceMesh] = None,
    placements: Optional[Sequence[Placement]] = None,
) -> DTensor:
    """
    Returns a :class:`DTensor` filled with random numbers from a uniform distribution
    on the interval ``[0, 1)``. The shape of the tensor is defined by the variable
    argument ``size``.

    Args:
        size (int...): a sequence of integers defining the shape of the output :class:`DTensor`.
            Can be a variable number of arguments or a collection like a list or tuple.
            E.g.: ones(1,2,3..) or ones([1,2,3..]) or ones((1,2,3..))

    Keyword args:
        dtype (:class:`torch.dtype`, optional): the desired data type of returned :class:`DTensor`.
            Default: if ``None``, uses a global default (see :func:`torch.set_default_dtype`).
        layout (:class:`torch.layout`, optional): the desired layout of returned DTensor.
            Default: ``torch.strided``.
        requires_grad (bool, optional): If autograd should record operations on the
            returned :class:`DTensor`. Default: ``False``.
        device_mesh: :class:`DeviceMesh` type, contains the mesh info of ranks.
        placements: a sequence of :class:`Placement` type: ``Shard``, ``Replicate``

    Returns:
        A :class:`DTensor` object on each rank
    """
    torch_size = normalize_to_torch_size(size)

    return _dtensor_init_helper(
        torch.rand,
        torch_size,
        dtype=dtype,
        layout=layout,
        requires_grad=requires_grad,
        device_mesh=device_mesh,
        placements=placements,
    )


def randn(  # type: ignore[no-untyped-def]
    *size,
    requires_grad: bool = False,
    dtype: Optional[torch.dtype] = None,
    layout: torch.layout = torch.strided,
    device_mesh: Optional[DeviceMesh] = None,
    placements: Optional[Sequence[Placement]] = None,
) -> DTensor:
    """
    Returns a :class:`DTensor` filled with random numbers from a normal distribution
    with mean 0 and variance 1. The shape of the tensor is defined by the variable
    argument ``size``.

    Args:
        size (int...): a sequence of integers defining the shape of the output :class:`DTensor`.
            Can be a variable number of arguments or a collection like a list or tuple.
            E.g.: ones(1,2,3..) or ones([1,2,3..]) or ones((1,2,3..))

    Keyword args:
        dtype (:class:`torch.dtype`, optional): the desired data type of returned :class:`DTensor`.
            Default: if ``None``, uses a global default (see :func:`torch.set_default_dtype`).
        layout (:class:`torch.layout`, optional): the desired layout of returned DTensor.
            Default: ``torch.strided``.
        requires_grad (bool, optional): If autograd should record operations on the
            returned :class:`DTensor`. Default: ``False``.
        device_mesh: :class:`DeviceMesh` type, contains the mesh info of ranks.
        placements: a sequence of :class:`Placement` type: ``Shard``, ``Replicate``

    Returns:
        A :class:`DTensor` object on each rank
    """
    torch_size = normalize_to_torch_size(size)

    return _dtensor_init_helper(
        torch.randn,
        torch_size,
        dtype=dtype,
        layout=layout,
        requires_grad=requires_grad,
        device_mesh=device_mesh,
        placements=placements,
    )


def zeros(  # type: ignore[no-untyped-def]
    *size,
    requires_grad: bool = False,
    dtype: Optional[torch.dtype] = None,
    layout: torch.layout = torch.strided,
    device_mesh: Optional[DeviceMesh] = None,
    placements: Optional[Sequence[Placement]] = None,
) -> DTensor:
    """
    Returns a :class:`DTensor` filled with the scalar value 0.

    Args:
        size (int...): a sequence of integers defining the shape of the output :class:`DTensor`.
            Can be a variable number of arguments or a collection like a list or tuple.
            E.g.: zeros(1,2,3..) or zeros([1,2,3..]) or zeros((1,2,3..))
    Keyword args:
        requires_grad (bool, optional): If autograd should record operations on the
            returned :class:`DTensor`. Default: ``False``.
        dtype (:class:`torch.dtype`, optional): the desired data type of returned :class:`DTensor`.
            Default: if ``None``, uses a global default (see :func:`torch.set_default_dtype`).
        layout (:class:`torch.layout`, optional): the desired layout of returned :class:`DTensor`.
            Default: ``torch.strided``.
        device_mesh: :class:`DeviceMesh` type, contains the mesh info of ranks
        placements: a sequence of :class:`Placement` type: ``Shard``, ``Replicate``

    Returns:
        A :class:`DTensor` object on each rank
    """
    torch_size = normalize_to_torch_size(size)

    return _dtensor_init_helper(
        torch.zeros,
        torch_size,
        dtype=dtype,
        layout=layout,
        requires_grad=requires_grad,
        device_mesh=device_mesh,
        placements=placements,
    )
