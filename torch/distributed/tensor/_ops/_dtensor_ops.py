import torch
from torch.distributed.device_mesh import _register_device_mesh_as_opaque_type
from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta
from torch.distributed.tensor._utils import compute_global_tensor_info
from torch.distributed.tensor.placement_types import Placement


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
