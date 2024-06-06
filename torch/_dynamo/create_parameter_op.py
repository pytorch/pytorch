import torch

import logging

torch_log = logging.getLogger("torch")

doc = """
This is used when dynamo traces torch.nn.Parameter, which normally would not trace properly
with AOTAutograd.  We instead create a placeholder torch.nn.Parameter before the graph, which
becomes a graph arg and has no storage backing it.  At the point in the graph where the parameter
actually should be created we mutate this sacrificial placeholder into it.  This allows gradients
to flow into the parameter as if it were an input to the graph (which is the only thing we are
allowed to compute gradients on).
""".strip()


lib = torch.library.Library("create_parameter_op", "FRAGMENT")

lib.define("set_(Tensor(a!) tensor, Tensor data) -> ()")

@torch.library.impl(lib, "set_", "Meta")
def set_(tensor, data):
    tensor.set_(data)

@torch.library.impl(lib, "set_", "CUDA")
def set_(tensor, data):
    tensor.set_(data)


class TracableCreateParameter(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, placeholder):
        assert not tensor.requires_grad
        # torch_log.warning(f"before: placeholder: {placeholder}")
        # torch_log.warning(f"before: tensor: {tensor}")
        if isinstance(tensor, torch.distributed._tensor.api.DTensor):
            with torch.no_grad():
                placeholder.copy_(tensor)
            # torch_log.warning(f"before: placeholder._local_tensor: {placeholder._local_tensor}")
            # torch_log.warning(f"before: tensor._local_tensor: {tensor._local_tensor}")
            # placeholder._local_tensor.set_(tensor._local_tensor)
            # placeholder._spec = tensor._spec
        else:
            # TODO(yf225): we should use `torch.ops.create_parameter_op.set_` here,
            # but somehow Dynamo/AOTAutograd will turn that into a `copy_`
            # which causes segfault because the sacrificial placeholder is size-0.
            # We need to just keep using `set_` instead of `copy_` in the graph.
            placeholder.set_(tensor)
            # torch.ops.create_parameter_op.set_(placeholder, tensor)
        # torch_log.warning(f"after: placeholder: {placeholder}")
        # torch_log.warning(f"after: tensor: {tensor}")
        return placeholder

    @staticmethod
    def backward(ctx, grad):
        # torch_log.warning(f"grad: {grad}")
        return None, grad  # grad flows to placeholder


def tracable_create_parameter(tensor, placeholder):
    with torch.set_grad_enabled(placeholder.requires_grad):
        out = TracableCreateParameter.apply(tensor, placeholder)
        # out = out.clone()
    return out


def new_parameter_placeholder(size, dtype, device, requires_grad):
    """Create a placeholder to be passed to the above functions"""
    result = torch.nn.Parameter(
        torch.empty(size, dtype=dtype, device=device), requires_grad=requires_grad
    )
    # TODO(jansel): alloc followed by free is inefficient, need a way to allocate an unbacked tensor.
    # Allocating a zero tensor would causes assert failures in autograd.
    result.untyped_storage().resize_(0)
    return result


def new_parameter_placeholder_dtensor(local_tensor_size, local_tensor_dtype, local_tensor_device, requires_grad, device_mesh, placements):
    """Create a placeholder to be passed to the above functions"""
    data_tensor = torch.empty(local_tensor_size, dtype=local_tensor_dtype, device=local_tensor_device)
    # data_tensor.untyped_storage().resize_(0)  # this causes segfault, need to figure out why
    # NOTE(yf225): allocate a placeholder nn.Parameter(DTensor), whose content will get swapped out in TracableCreateParameter.forward
    data_tensor = torch.distributed._tensor.api.DTensor.from_local(
        data_tensor,
        device_mesh=device_mesh,
        placements=placements,
    )
    result = torch.nn.Parameter(
        data_tensor, requires_grad=requires_grad
    )
    # torch_log.warning(f"new_parameter_placeholder_dtensor: result: {result}")
    return result
