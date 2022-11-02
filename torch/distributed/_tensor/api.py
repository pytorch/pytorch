# Copyright (c) Meta Platforms, Inc. and affiliates
import copy
import warnings
import torch
from torch.utils._pytree import tree_flatten
from typing import Dict, Callable, Optional, Sequence, cast
from torch.distributed._tensor.device_mesh import get_global_device_mesh, DeviceMesh
from torch.distributed._tensor.placement_types import (
    Placement,
    Shard,
    Replicate,
    _Partial,
    DTensorSpec,
)
from torch.distributed._tensor.redistribute import Redistribute

from torch.distributed._tensor.dispatch import operator_dispatch, OpSchema, OutputSharding

# NOTE [Autograd interaction between torch.Tensor]
#
# The autograd functions defined below are being used by the public
# facing APIs (i.e. from_local, to_local) to ensure our DTensor
# works together with torch.Tensor within autograd engine. This
# allows DistributedTensor to exist on part of the module hierarchy
# and still able to calculate gradients across the torch.Tensor and
# DistributedTensor boundary.
# As an example, we have the a module that consists of submodules
# A, B, and C, the execution flow would be like:
#  input(torch.Tensor) -> Module A -> Module B -> Module C -> output (torch.Tensor)
#
# Suppose I only want to make Module B be a sharded module with
# DistributedTensor params, we would need to make the folloing
# flow to work:
#
#  input(torch.Tensor) -> Module A
#       -> DTensor input -> Sharded Module B -> DTensor output
#           -> output (torch.Tensor) -> Module C -> output (torch.Tensor)
#
# We need the conversion from Module A to DTensor input, which is
# `from_local`, and conversion from DTensor output to output, which
# is `to_local`, thus these two functions must be Autograd functions.
#
class ToTorchTensor(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: "DTensor"):  # type: ignore[override]
        ctx.dtensor_device_mesh = input.device_mesh
        ctx.dtensor_placements = input.placements
        ctx.dtensor_shape = input.shape
        ctx.dtensor_requires_grad = input.requires_grad
        return input._local_tensor.detach()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        device_mesh = ctx.dtensor_device_mesh
        placements = ctx.dtensor_placements
        return DTensor(
            grad_output,
            device_mesh,
            placements,
            size=ctx.dtensor_shape,
            requires_grad=grad_output.requires_grad,
        )


class FromTorchTensor(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,  # pyre-ignore[2]: Parameter must be annotated.
        input: torch.Tensor,
        device_mesh: DeviceMesh,
        placements: Sequence[Placement],
        run_check: bool,
    ) -> "DTensor":
        ctx.previous_placement = placements
        ctx.previous_device_mesh = device_mesh

        if run_check:
            # TODO: by default check tensor metas across rank
            # TODO: See if we need to make this run_check logic
            # have a corresponding backward.
            for idx, placement in enumerate(placements):
                if placement.is_replicate():
                    # broadcast rank 0 tensor to all ranks
                    # only broadcast if run_check is True
                    input = input.contiguous()
                    device_mesh.broadcast(input, mesh_dim=idx)

        # if it's not by default run_check, we assume user is certain that each
        # rank has the same tensor shape, and we just use that to calculate the
        # global shape
        tensor_shape = list(input.size())
        for idx, placement in enumerate(placements):
            if placement.is_shard():
                shard_dim = cast(Shard, placement).dim
                local_dim_size = tensor_shape[shard_dim]
                tensor_shape[shard_dim] = local_dim_size * device_mesh.size(idx)

        dist_tensor = DTensor(
            input,
            device_mesh,
            placements,
            size=torch.Size(tensor_shape),
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
            # pyre-fixme[16]: `Redistribute` has no attribute `apply`.
            grad_output = Redistribute.apply(
                grad_output, previous_device_mesh, previous_placement
            )

        # TODO: backward is also differentiable now, add a test
        # to test higher level gradients.
        return grad_output.to_local(), None, None, None


class DTensor(torch.Tensor):  # pyre-ignore[13]: pyre is bad at __new__
    _local_tensor: torch.Tensor
    _spec: DTensorSpec
    __slots__ = ["_local_tensor", "_spec"]

    # class attribute that handles operator placements propagation
    # rules, keyed by aten op name, value is propagation func
    _op_to_rules: Dict[str, Callable[[OpSchema], OutputSharding]] = {}

    # class attribute that handles custom registered ops, all handled
    # custom ops should appear in this table, and overriding the default
    # operators that's been covered by _op_to_rules or fallbacks.
    # (custom operator is the highest priority when dispatching).
    # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
    _custom_dispatch_ops: Dict[str, Callable] = {}

    @staticmethod
    def __new__(
        cls,
        local_tensor: torch.Tensor,
        device_mesh: DeviceMesh,
        placements: Sequence[Placement],
        *,
        size: torch.Size,
        requires_grad: bool = False,
    ) -> "DTensor":
        """
        Construct a DTensor from a local tensor, device mesh, and placement and
        other tensor properties (i.e. shape, requires_grad, strides, etc).
        Note: This is not a public API and it's only supposed to be used by the
            operator implementations and internals. If you want to construct a
            DTensor from a local tensor, consider using `DTensor.from_local`, if
            you want to construct a DTensor from a "global" tensor (where you
            already have tensor initialized and want to shard this tensor),
            consider using `distribute_tensor`.
        """
        # recover tensor strides from local tensor strides and global size info
        # in the case of sharding
        # TODO: we should try to use meta tensor for shape and stride calculation
        tensor_stride = list(local_tensor.stride())
        local_size = list(local_tensor.size())
        for placement in placements:
            if isinstance(placement, Shard):
                shard_dim = placement.dim
                # recover tensor stride by modifying the stride that larger than
                # the current stride on the shard_dim
                for i in range(len(tensor_stride)):
                    if (
                        i != shard_dim
                        and tensor_stride[i] >= tensor_stride[shard_dim]
                    ):
                        # rescale the stride by the shard size
                        tensor_stride[i] = (
                            tensor_stride[i] // local_size[shard_dim]
                        ) * size[shard_dim]
            elif not isinstance(placement, (Replicate, _Partial)):
                raise RuntimeError(
                    f"placement type {type(placement)} not supported!"
                )

        if requires_grad != local_tensor.requires_grad:
            warnings.warn(
                "To construct DTensor from torch.Tensor, it's recommended to "
                "use local_tensor.detach() and make requires_grad consistent."
            )

        # new method instruct wrapper tensor from local_tensor and add
        # placement spec, it does not do actual distribution
        r = torch.Tensor._make_wrapper_subclass(  # type: ignore[attr-defined]
            cls,
            size,
            strides=tensor_stride,
            dtype=local_tensor.dtype,
            device=local_tensor.device,
            layout=local_tensor.layout,
            requires_grad=requires_grad,
        )
        # deepcopy and set spec
        r._spec = DTensorSpec(
            device_mesh, copy.deepcopy(placements), shape=r.size()
        )
        # detach local tensor from autograd graph as we initialize the
        # distributed tensor and autograd will be working on top of
        # the wrapper tensor directly instead of local torch.Tensor
        r._local_tensor = local_tensor.detach()
        return r

    # pyre-fixme[14]: `__repr__` overrides method defined in `DTensor` inconsistently.
    # pyre-fixme[3]: Return type must be annotated.
    def __repr__(self):
        # TODO: consider all_gather the local tensors for better debugging
        return f"DTensor(local_tensor={self._local_tensor}, device_mesh={self._spec.mesh}, placements={self._spec.placements})"

    @classmethod
    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def __torch_function__(cls, func, types, args=(), kwargs={}):
        # if we find nn.functional name in dispatch op, dispatch to it instead,
        # this allow us to override some python level behaviors that wouldn't be
        # possible in __torch_dispatch__ level.
        if func.__name__ in DTensor._custom_dispatch_ops:
            # dispatch to the same table as the name should be different between
            # torch_function and torch_dispatch
            return DTensor._custom_dispatch_ops[func.__name__](*args, **kwargs)
        else:
            # if not, just do nothing here
            return super().__torch_function__(func, types, args, kwargs)

    @classmethod
    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        # check that we are not getting mixed vanilla and Distributed tensors
        arg_list, _ = tree_flatten(args)
        for arg in arg_list:
            if isinstance(arg, torch.Tensor) and not isinstance(arg, DTensor):
                raise RuntimeError(
                    f"{func}: got mixed distributed and non-distributed tensors."
                )

        if kwargs is None:
            kwargs = {}

        return operator_dispatch(
            func,
            args,
            kwargs,
            DTensor._op_to_rules,
            DTensor._custom_dispatch_ops,
        )

    @classmethod
    def from_local(
        cls,
        local_tensor: torch.Tensor,
        device_mesh: Optional[DeviceMesh] = None,
        placements: Optional[Sequence[Placement]] = None,
        run_check: bool = True,
    ) -> "DTensor":
        """
        Create a :class:`DTensor` from a local torch.Tensor on each rank
        according to the `device_mesh` and `placements` specified.

        Args:
            local_tensor (torch.Tensor): local torch.Tensor on each rank.
            device_mesh (:class:`DeviceMesh`, optional): DeviceMesh to place the
                tensor, if not specified, must be called under a DeviceMesh
                context manager, default: None
            placements (List[:class:`Placement`], optional): the placements that
                describes how to place the local torch.Tensor on DeviceMesh, must
                have the same number of elements as `device_mesh.ndim`. If not
                specified, we will by default replicate the tensor across the
                `device_mesh` from the first rank of each dimension of the `device_mesh`.
            run_check (bool, optional): indicate whether to run check across ranks
                to check meta information and data. if have :class:`Replicate` in
                `placements`, the data on first rank of the device mesh dimension
                will be broadcasted to other ranks.

        Returns:
            A :class:`DTensor` object

        .. note:: `from_local` is differentiable, the `requires_grad` of the created
            `DTensor` object will depend on if `local_tensor` requires_grad or not.
        """
        # if same shape/dtype, no need to run_check, if not, must allgather
        # the metadatas to check the size/dtype across ranks
        # There should be no data communication unless there's replication
        # strategy, where we broadcast the replication from the first rank
        # in the mesh dimension
        device_mesh = (
            get_global_device_mesh() if device_mesh is None else device_mesh
        )
        # convert the local tensor to desired device base on device mesh's device_type
        local_tensor = local_tensor.to(device_mesh.device_type)

        # set default placements to replicated if not specified
        if placements is None:
            placements = [Replicate() for _ in range(device_mesh.ndim)]

        # `from_local` is differentiable, and the gradient of the dist tensor this function
        # created should flow back the gradients to the local_tensor, so we call an autograd
        # function to construct the dist tensor instead.
        return FromTorchTensor.apply(  # pyre-ignore[16]: autograd func
            local_tensor, device_mesh, placements, run_check
        )

    def to_local(self) -> torch.Tensor:
        """
        Get the local tensor of this DTensor on its current rank. For sharding it returns
        a local shard of the logical tensor view, for replication it returns the replica on
        its current rank.

        Returns:
            A :class:`torch.Tensor` object that represents the local tensor of its current rank.

        .. note:: `to_local` is differentiable, the `requires_grad` of the local tensor returned
            will depend on if the `DTensor` requires_grad or not.
        """
        return ToTorchTensor.apply(self)  # pyre-ignore[16]: autograd func

    def redistribute(
        self,
        device_mesh: Optional[DeviceMesh] = None,
        placements: Optional[Sequence[Placement]] = None,
    ) -> "DTensor":
        """
        `redistribute` performs necessary collective operations that redistribute the current
        DTensor from its current placements to a new placements, or from is current DeviceMesh
        to a new DeviceMesh. i.e. we can turn a Sharded DTensor to a Replicated DTensor by
        specifying a Replicate placement for each dimension of the DeviceMesh.

        Args:
            device_mesh (:class:`DeviceMesh`, optional): DeviceMesh to place the
                DTensor, if not specified, must be called under a DeviceMesh
                context manager, default: None
            placements (List[:class:`Placement`], optional): the new placements that
                describes how to place the DTensor into the DeviceMesh, must
                have the same number of elements as `device_mesh.ndim`.

        Returns:
            A :class:`DTensor` object

        .. note:: `redistribute` is differentiable.
        """
        # This API perform necessary transformations and get
        # a new DTensor with the new spec. i.e. for
        # sharding it's a reshard behavior.
        # Note that redistribute currently only supports out
        # of place redistribution, i.e. it always create a new
        # DTensor object and leave the original one unchanged.
        device_mesh = (
            get_global_device_mesh() if device_mesh is None else device_mesh
        )
        # raise error if new placements not specified
        if placements is None:
            raise RuntimeError("placements is needed for redistribute!")

        for placement in placements:
            if placement.is_partial():
                raise RuntimeError(
                    "Can not redistribute to _Partial, _Partial is for internal use only!"
                )

        # pyre-fixme[16]: `Redistribute` has no attribute `apply`.
        return Redistribute.apply(self, device_mesh, placements)

    @property
    def device_mesh(self) -> DeviceMesh:
        """
        The :class:`DeviceMesh` attribute that associates with this DTensor object.

        .. note:: device_mesh is a read-only property, it can not be set.
        """
        return self._spec.mesh

    @property
    def placements(self) -> Sequence[Placement]:
        """
        The placements attribute of this DTensor that describes the layout of this
        DTensor on the its DeviceMesh.

        .. note:: placements is a read-only property, it can not be set.
        """
        return self._spec.placements
