# Copyright (c) Meta Platforms, Inc. and affiliates
import warnings
from typing import Callable, cast, Optional, Sequence, Tuple

import torch

import torch.distributed._tensor.dispatch as op_dispatch
import torch.distributed._tensor.random as random
import torch.nn as nn
from torch.distributed._tensor._collective_utils import mesh_broadcast
from torch.distributed._tensor._utils import compute_global_tensor_info
from torch.distributed._tensor.device_mesh import DeviceMesh, mesh_resources
from torch.distributed._tensor.placement_types import (
    DTensorSpec,
    Placement,
    Replicate,
    Shard,
)
from torch.distributed._tensor.random import (
    is_rng_supported_mesh,
    OffsetBasedRNGTracker,
)
from torch.distributed._tensor.redistribute import Redistribute
from torch.distributed._tensor.sharding_prop import ShardingPropagator
from torch.fx.passes.shape_prop import TensorMetadata
from torch.utils._pytree import tree_flatten


__all__ = ["DTensor", "distribute_tensor", "distribute_module"]


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
# DistributedTensor params, we would need to make the following
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
class _ToTorchTensor(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: "DTensor"):  # type: ignore[override]
        ctx.dtensor_spec = input._spec
        # We need to return a fresh Tensor object there as autograd metadata
        # will be inplaced into it. So we don't want to polute the Tensor
        # object stored in _local_tensor.
        return input._local_tensor.view_as(input._local_tensor)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        dtensor_spec = ctx.dtensor_spec
        dtensor_meta = dtensor_spec.tensor_meta
        _, tensor_stride = compute_global_tensor_info(
            grad_output, dtensor_spec.mesh, dtensor_spec.placements
        )
        return DTensor(
            grad_output,
            dtensor_spec.mesh,
            dtensor_spec.placements,
            shape=dtensor_meta.shape,
            dtype=dtensor_meta.dtype,
            requires_grad=grad_output.requires_grad,
            stride=tuple(tensor_stride),
        )


class _FromTorchTensor(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,  # pyre-ignore[2]: Parameter must be annotated.
        input: torch.Tensor,
        device_mesh: DeviceMesh,
        placements: Tuple[Placement, ...],
        run_check: bool,
    ) -> "DTensor":
        ctx.previous_placement = placements
        ctx.previous_device_mesh = device_mesh

        # if it's not by default run_check, we assume user is certain that each
        # rank has the same tensor shape, and we just use that to calculate the
        # global shape
        tensor_shape, tensor_stride = compute_global_tensor_info(
            input, device_mesh, placements
        )

        if device_mesh.get_coordinate() is None:
            # if the global rank is not participating in the device mesh, we
            # simply set the local tensor to an empty tensor
            input = input.new_empty(0, requires_grad=input.requires_grad)
        elif run_check:
            # TODO: by default check tensor metas across rank
            # TODO: See if we need to make this run_check logic
            # have a corresponding backward.
            for idx, placement in enumerate(placements):
                if placement.is_replicate():
                    # broadcast rank 0 tensor to all ranks
                    # only broadcast if run_check is True
                    input = input.contiguous()
                    mesh_broadcast(input, device_mesh, mesh_dim=idx)

        # We want a fresh Tensor object that shares memory with the input tensor
        dist_tensor = DTensor(
            input.view_as(input),
            device_mesh,
            placements,
            shape=torch.Size(tensor_shape),
            dtype=input.dtype,
            # requires_grad of the dist tensor depends on if input
            # requires_grad or not
            requires_grad=input.requires_grad,
            stride=tuple(tensor_stride),
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
    _propagator: ShardingPropagator = ShardingPropagator()

    @staticmethod
    def __new__(
        cls,
        local_tensor: torch.Tensor,
        device_mesh: DeviceMesh,
        placements: Tuple[Placement, ...],
        *,
        shape: torch.Size,
        dtype: torch.dtype,
        requires_grad: bool,
        stride: Tuple[int, ...],
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
        if requires_grad != local_tensor.requires_grad:
            warnings.warn(
                "To construct DTensor from torch.Tensor, it's recommended to "
                "use local_tensor.detach() and make requires_grad consistent."
            )

        # new method instruct wrapper tensor from local_tensor and add
        # placement spec, it does not do actual distribution
        r = torch.Tensor._make_wrapper_subclass(  # type: ignore[attr-defined]
            cls,
            shape,
            strides=stride,
            dtype=dtype,
            device=local_tensor.device,
            layout=local_tensor.layout,
            requires_grad=requires_grad,
        )

        # TODO: populate all tensor meta fields properly
        # NOTE: memory_format is non-pickable so we intentionally skip it
        tensor_meta = TensorMetadata(
            shape, dtype, requires_grad, stride, None, False, {}
        )
        # deepcopy and set spec
        r._spec = DTensorSpec(device_mesh, placements, tensor_meta=tensor_meta)
        r._local_tensor = local_tensor
        return r

    # pyre-fixme[14]: `__repr__` overrides method defined in `DTensor` inconsistently.
    # pyre-fixme[3]: Return type must be annotated.
    def __repr__(self):
        # TODO: consider all_gather the local tensors for better debugging
        return f"DTensor(local_tensor={self._local_tensor}, device_mesh={self._spec.mesh}, placements={self._spec.placements})"

    def __tensor_flatten__(self):
        """
        protocol to inform how to flatten a DTensor to local tensor
        for PT2 tracing
        """
        return [self._local_tensor], self._spec

    @staticmethod
    def __tensor_unflatten__(inner_tensors, spec):
        assert (
            spec is not None
        ), "Expecting spec to be not None from `__tensor_flatten__` return value!"
        assert isinstance(inner_tensors, (list, tuple)) and len(inner_tensors) == 1
        local_tensor = inner_tensors[0]
        return DTensor(
            local_tensor,
            spec.mesh,
            spec.placements,
            shape=spec.tensor_meta.shape,
            dtype=spec.tensor_meta.dtype,
            requires_grad=spec.tensor_meta.requires_grad,
            stride=spec.tensor_meta.stride,
        )

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

        return op_dispatch.operator_dispatch(
            func,
            args,
            kwargs,
            DTensor._propagator,
        )

    @staticmethod
    def from_local(
        local_tensor: torch.Tensor,
        device_mesh: Optional[DeviceMesh] = None,
        placements: Optional[Sequence[Placement]] = None,
        *,
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

        Keyword args:
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
        device_mesh = device_mesh or mesh_resources.get_current_mesh()

        # convert the local tensor to desired device base on device mesh's device_type
        if not local_tensor.is_meta:
            local_tensor = local_tensor.to(device_mesh.device_type)

        # set default placements to replicated if not specified
        if placements is None:
            placements = [Replicate() for _ in range(device_mesh.ndim)]

        # `from_local` is differentiable, and the gradient of the dist tensor this function
        # created should flow back the gradients to the local_tensor, so we call an autograd
        # function to construct the dist tensor instead.
        return _FromTorchTensor.apply(  # pyre-ignore[16]: autograd func
            local_tensor, device_mesh, tuple(placements), run_check
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
        return _ToTorchTensor.apply(self)  # pyre-ignore[16]: autograd func

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
        # NOTE: This redistribute API currently only supports out
        # of place redistribution, i.e. it always create a new
        # DTensor object and leave the original one unchanged.

        # if device_mesh is not specified, use the current device_mesh
        device_mesh = device_mesh or self.device_mesh
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


def distribute_tensor(
    tensor: torch.Tensor,
    device_mesh: Optional[DeviceMesh] = None,
    placements: Optional[Sequence[Placement]] = None,
) -> DTensor:
    """
    Distribute a torch.Tensor to the `device_mesh` according to the `placements`
    specified. The rank of `device_mesh` and `placements` must be the same.

    Args:
        tensor (torch.Tensor): torch.Tensor to be distributed. Note that if you
            want to shard a tensor on a dimension that is not evenly divisible by
            the number of devices in that mesh dimension, we use `torch.chunk`
            semantic to shard the tensor and scatter the shards.
        device_mesh (:class:`DeviceMesh`, optional): DeviceMesh to distribute the
            tensor, if not specified, must be called under a DeviceMesh context
            manager, default: None
        placements (List[:class:`Placement`], optional): the placements that
            describes how to place the tensor on DeviceMesh, must have the same
            number of elements as `device_mesh.ndim`. If not specified, we will
            by default replicate the tensor across the `device_mesh` from the
            first rank of each dimension of the `device_mesh`.

    Returns:
        A :class:`DTensor` object
    """

    torch._C._log_api_usage_once("torch.dtensor.distribute_tensor")

    # get default device mesh if there's nothing specified
    device_mesh = device_mesh or mesh_resources.get_current_mesh()

    # instantiate a RNG tracker if haven't. By default DTensor uses an
    # OffsetBasedRNGTracker to perform random operators.
    # TODO: the value assignment to global variable is not the ideal solution
    # we can replace it in future.
    if is_rng_supported_mesh(device_mesh) and not random._rng_tracker:
        random._rng_tracker = OffsetBasedRNGTracker(device_mesh.device_type)

    if not tensor.is_leaf:
        raise RuntimeError(
            "`distribute_tensor` should be used to distribute leaf tensors! but found non-leaf tensor!"
        )

    # convert tensor to the corresponding device type if it's not in that device type
    if not tensor.is_meta:
        tensor = tensor.to(device_mesh.device_type)
    # set default placements to replicated if not specified
    if placements is None:
        placements = [Replicate() for _ in range(device_mesh.ndim)]

    if len(placements) != device_mesh.ndim:
        raise ValueError(
            f"`placements` must have the same length as `device_mesh.ndim`! "
            f"Found placements length: {len(placements)}, and device_mesh.ndim: {device_mesh.ndim}."
        )

    if isinstance(tensor, DTensor):
        # if the tensor is already a DTensor, we just need to check if the
        # device mesh and placements are the same
        if tensor.device_mesh != device_mesh:
            raise ValueError(
                f"Cannot distribute a DTensor with device mesh {tensor.device_mesh} "
                f"to a different device mesh {device_mesh}."
            )
        if tensor.placements != tuple(placements):
            raise ValueError(
                f"Cannot distribute a DTensor with placements {tensor.placements} "
                f"to a different placements {placements}. do you want to call "
                f"`redistribute` instead?"
            )
        return tensor

    local_tensor = tensor

    # distribute the tensor according to the placements.
    for idx, placement in enumerate(placements):
        if placement.is_shard():
            placement = cast(Shard, placement)
            output = placement._shard_tensor(local_tensor, device_mesh, idx)
            # scatter call could not return a tensor with correct requires_grad
            # field, as ProcessGroupNCCL refuse to take a tensor with requires_grad
            # to do inplace update! So we manually set it here
            output.requires_grad_(tensor.requires_grad)
            local_tensor = output
        elif placement.is_replicate():
            placement = cast(Replicate, placement)
            local_tensor = placement._replicate_tensor(local_tensor, device_mesh, idx)
        else:
            raise RuntimeError(
                f"Trying to distribute tensor with unsupported placements {placement} on device mesh dimension {idx}!"
            )

    assert local_tensor is not None, "distributing a tensor should not be None"
    # detach the local tensor passed to DTensor since after the construction
    # of DTensor, autograd would work on top of DTensor instead of local tensor
    return DTensor(
        local_tensor.detach(),
        device_mesh,
        tuple(placements),
        shape=tensor.size(),
        dtype=tensor.dtype,
        requires_grad=tensor.requires_grad,
        stride=tensor.stride(),
    )


def distribute_module(
    module: nn.Module,
    device_mesh: Optional[DeviceMesh] = None,
    partition_fn: Optional[Callable[[str, nn.Module, DeviceMesh], None]] = None,
    input_fn: Optional[Callable[..., None]] = None,
    output_fn: Optional[Callable[..., None]] = None,
) -> nn.Module:
    """
    This function converts all module parameters to :class:`DTensor` parameters
    according to the `partition_fn` specified. It could also control the input or
    output of the module by specifying the `input_fn` and `output_fn`. (i.e. convert
    the input to :class:`DTensor`, convert the output back to torch.Tensor)
    Args:
        module (:class:`nn.Module`): user module to be partitioned.
        device_mesh (:class:`DeviceMesh`): the device mesh to place the module.
        partition_fn (Callable): the function to partition parameters (i.e. shard certain
            parameters across the `device_mesh`). If `partition_fn` is not specified,
            by default we replicate all module parameters of `module` across the mesh.
        input_fn (Callable): specify the input distribution, i.e. could control how the
            input of the module is sharded. `input_fn` will be installed as a module
            `forward_pre_hook` (pre forward hook).
        output_fn (Callable): specify the output distribution, i.e. could control how the
            output is sharded, or convert it back to torch.Tensor. output_fn will be
            installed as a module `forward_hook` (post forward hook).

    Returns:
        A module that contains parameters/buffers that are all `DTensor`s.
    """

    torch._C._log_api_usage_once("torch.dtensor.distribute_module")

    device_mesh = device_mesh or mesh_resources.get_current_mesh()

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
        module.register_forward_pre_hook(lambda _, inputs: input_fn(inputs, device_mesh))  # type: ignore[misc]
    # register input_fn as module forward hook
    if output_fn is not None:
        module.register_forward_hook(
            lambda mod, inputs, outputs: output_fn(outputs, device_mesh)  # type: ignore[misc]
        )

    return module
