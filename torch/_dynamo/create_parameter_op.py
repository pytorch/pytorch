# mypy: allow-untyped-defs
import torch

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
        # TODO(yf225): we should use `torch.ops.create_parameter_op.set_` here,
        # but somehow Dynamo/AOTAutograd will turn that into a `copy_`
        # which causes segfault because the sacrificial placeholder is size-0.
        # We need to just keep using `set_` instead of `copy_` in the graph.
        placeholder.set_(tensor)
        # torch.ops.create_parameter_op.set_(placeholder, tensor)
        return placeholder

    @staticmethod
    def backward(ctx, grad):
        return None, grad  # grad flows to placeholder


def tracable_create_parameter(tensor, placeholder):
    with torch.set_grad_enabled(placeholder.requires_grad):
        out = TracableCreateParameter.apply(tensor, placeholder)
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
