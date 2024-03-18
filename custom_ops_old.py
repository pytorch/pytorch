import torch
import numpy as np
# aka torch.library
from library import OpDef, traceable, device_types

# =====================================================================
# This was the initial design. It has been superceded with custom_ops.py.
# Leaving it here for comparison purposes.
# =====================================================================

# User provides their custom op schema and implementations
class MySin(OpDef):
    schema = "(Tensor x) -> Tensor"

    # the black-box cpu kernel
    @staticmethod
    def impl_cpu(x):
        return torch.from_numpy(np.sin(x.detach().cpu().numpy()))

    # the black-box cuda kernel
    @staticmethod
    def impl_cuda(x):
        return torch.from_numpy(np.sin(x.detach().cpu().numpy())).to(x.device)

    # the abstract impl. Must be "traceable". User must use opcheck to test.
    @staticmethod
    def abstract(x):
        return torch.empty_like(x)

    # autograd: provide us setup_backward() and backward() methods.
    # these must be "traceable". User must use opcheck to test.
    @staticmethod
    def setup_backward(ctx, inputs, output):
        x, = inputs
        ctx.save_for_backward(x)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors 
        return grad_output * x.cos()


# Builds the provided implementations into a "custom op" that behaves like a function.
# We don't do something like MySin.apply (i.e. the autograd.Function way) because
# dynamic changes to methods on the MySin object will not be included in the op.
my_sin_op = MySin.build()


# The user should provide a function that calls the custom op so they can add a docstring.
def my_sin(x):
    """my_sin(x: Tensor) -> Tensor

    Returns the sin of x.
    """
    return my_sin_op(x)


# Example of an operator that is implemented with pytorch operations
# We automatically generate an abstract impl for it.
class MySinCos(OpDef):
    schema = "(Tensor x) -> Tensor"

    @staticmethod
    @traceable  # specifies that we can auto-generate an abstract impl for this op
    @device_types("CPU", "CUDA")  # specifies which devices this is valid for
    def impl(x):
        return x.sin().cos()

my_sin_cos_op = MySinCos.build()


def my_sin_cos(x):
    """my_sin_cos(x: Tensor) -> Tensor

    Returns x.sin().cos()
    """
    return my_sin_cos_op(x)
