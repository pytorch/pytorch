import torch
from torch.distributed.device_mesh import (
    _register_device_mesh_as_opaque_type,
    DeviceMesh,
)
from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta
from torch.distributed.tensor._redistribute import redistribute_local_tensor
from torch.distributed.tensor._utils import compute_global_tensor_info
from torch.distributed.tensor.placement_types import Placement, Replicate


_register_device_mesh_as_opaque_type()


@torch.library.custom_op(
    "_dtensor::_to_local_tensor",
    mutates_args=(),
    schema="(Tensor(a) input, torch.distributed.tensor.placement_types.Placement[]? grad_placements) -> Tensor(a)",
)
def _to_local_tensor(
    input: torch.Tensor,
    grad_placements: tuple[Placement, ...] | None = None,
) -> torch.Tensor:
    raise RuntimeError(
        "This function should not be directly called. "
        "Instead _to_local_tensor_handler should be called."
    )


def _to_local_tensor_backward(ctx, grad_output):
    from torch.distributed.tensor._api import DTensor

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
        # pyrefly: ignore [bad-argument-type]
        DTensor(
            # pyrefly: ignore [bad-argument-count]
            grad_output,
            grad_spec,
            # pyrefly: ignore [unexpected-keyword]
            requires_grad=grad_output.requires_grad,
        ),
        None,
    )


def _to_local_tensor_setup_context(ctx, inputs, output):
    ctx._is_pure_view = True
    input, grad_placements = inputs
    ctx.dtensor_spec = input._spec
    ctx.grad_placements = grad_placements


_to_local_tensor.register_autograd(
    _to_local_tensor_backward,
    setup_context=_to_local_tensor_setup_context,
)


def _to_local_tensor_handler(
    op_call: torch._ops.OpOverload,
    args: tuple[object, ...],
    kwargs: dict[str, object],
):
    local_tensor = args[0]._local_tensor  # pyrefly: ignore[missing-attribute]
    result = local_tensor.view_as(local_tensor)
    return result


@torch.library.custom_op(
    "_dtensor::_redistribute",
    mutates_args=(),
    schema="(Tensor input, torch.distributed.device_mesh.DeviceMesh mesh, torch.distributed.tensor.placement_types.Placement[] placements, bool async_op=False, ScalarType? forward_dtype=None, ScalarType? backward_dtype=None) -> Tensor",  # noqa: B950
)
def _redistribute(
    input,
    device_mesh: DeviceMesh,
    placements: tuple[Placement, ...],
    async_op: bool = False,
    forward_dtype: torch.dtype | None = None,
    backward_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    raise RuntimeError(
        "This function should not be directly called. "
        "Instead _redistribute_handler should be called."
    )


def _redistribute_handler(
    op_call: torch._ops.OpOverload,
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> torch.Tensor:
    input, device_mesh, placements = args[:3]
    async_op = args[3] if len(args) > 3 else False
    forward_dtype = args[4] if len(args) > 4 else None

    from torch.distributed.tensor._api import DTensor

    input = input.to(dtype=forward_dtype)  # pyrefly: ignore [missing-attribute]
    local_tensor = input._local_tensor
    current_spec = input._spec

    if current_spec.placements != placements:
        target_spec = DTensorSpec(
            device_mesh,  # pyrefly: ignore [bad-argument-type]
            placements,  # pyrefly: ignore [bad-argument-type]
            tensor_meta=current_spec.tensor_meta,
        )

        output = redistribute_local_tensor(
            local_tensor,
            current_spec,
            target_spec,
            async_op=async_op,  # pyrefly: ignore [bad-argument-type]
        )
    else:
        # use the same local tensor if placements are the same.
        output = local_tensor
        target_spec = current_spec

    # pyrefly: ignore [bad-argument-type]
    return DTensor(
        # pyrefly: ignore [bad-argument-count]
        output,
        target_spec,
        # pyrefly: ignore [unexpected-keyword]
        requires_grad=input.requires_grad,
    )


def _redistribute_backward(ctx, grad_output):
    previous_spec = ctx.current_spec

    backward_dtype = ctx.backward_dtype or ctx.original_dtype

    grad_output = grad_output.to(dtype=backward_dtype)

    current_spec = grad_output._spec
    previous_spec = DTensorSpec(
        mesh=previous_spec.device_mesh,
        placements=previous_spec.placements,
        tensor_meta=current_spec.tensor_meta,
    )

    # skip the replicate to partial transformation when we are in backward pass
    # In this case we keep the grad as replicate, this is because we don't
    # want to convert the replicated gradients back to partial, although
    # that's logically conform with the same layout, converting the gradients
    # back to partial is actually useless as you would have to do reduce later
    # which would be more expensive than keeping it replicate!

    # for backward shard -> partial, we just do shard -> replicate
    # for backward replicate -> partial, we skip the transformation
    normalized_placements: list[Placement] = []
    for current, target in zip(current_spec.placements, previous_spec.placements):
        if (current.is_shard() or current.is_replicate()) and target.is_partial():
            normalized_placements.append(Replicate())
        else:
            normalized_placements.append(target)

    if tuple(normalized_placements) != current_spec.placements:
        grad_output.redistribute(placements=tuple(normalized_placements))

    grad_output = grad_output.to(ctx.original_dtype)

    return (
        grad_output,
        None,
        None,
        None,
        None,
        None,
    )


def _redistribute_setup_context(ctx, inputs, output):
    """Setup context for backward pass."""
    input, device_mesh, placements, async_op, forward_dtype, backward_dtype = inputs
    ctx.async_op = async_op
    ctx.backward_dtype = backward_dtype
    ctx.original_dtype = input._local_tensor.dtype

    input = input.to(dtype=forward_dtype)
    current_spec = input._spec

    ctx.current_spec = current_spec


_redistribute.register_autograd(
    _redistribute_backward,
    setup_context=_redistribute_setup_context,
)
