import torch
from torch._ops import PyOperator
from torch._C._functorch import TransformType
from torch._functorch.utils import enable_autograd_function
import torch.utils._pytree as pytree
from torch._C._functorch import (
    _wrap_for_grad,
    _unwrap_for_grad,
    _unwrap_batched,
)
from torch._functorch.vmap import (
    _broadcast_to_and_flatten,
    _create_batched_inputs,
)
from torch.autograd.forward_ad import _set_fwd_grad_enabled
from typing import NamedTuple

# autograd.Function technically runs before the regular PyTorch dispatcher.
# This is how features like autocast and torch_dispatch (e.g. PythonTLSSnapshot)
# work with it. One day we might decide to change this, but until then,
# we need to give the illusion that autograd.Function runs before those things.
#
# We do this by using creating a custom PyOperator that only functorch
# dispatches specially.
class CustomFunctionPyOperator(PyOperator):
    def __init__(self):
        super().__init__('custom_function_call')

    def __call__(self, *args, **kwargs):
        # When custom_function_call is done dispatching through functorch,
        # it should just invoke the autograd.Function. This is consistent
        # with the autograd.Function behavior of being invoked before the
        # PyTorch dispatcher.
        #
        # This will lead us into trouble later down the line, but this is
        # pre-existing. There is an invariant that a function traced by
        # make_fx should have the same behavior when provided the same
        # Tensor. However, make_fx sees autograd.Function as a composite
        # (because autograd.Function happens before the Python dispatch key)
        # and only traces the forward pass.
        if torch._C._are_functorch_transforms_active():
            return super().__call__(*args, **kwargs)
        autograd_function = args[0]
        return autograd_function.apply(*args[1:], **kwargs)


# "custom_function_call"
# This is the mechanism for an autograd.Function that works with functorch transforms.
# It wraps an autograd.Function; interactions with functorch transforms are defined
# via PyDispatcher and PyOperator rather than through the traditional PyTorch
# dispatcher.
custom_function_call = CustomFunctionPyOperator()


# The grad rule for custom_function_call is to construct a new _SingleLevelFunction
# (autograd.Function that only works with a single layer (level) of functorch) that:
# - unwraps the inputs
# - redispatches to custom_function_call
# - wraps the outputs
# and whose backward pass calls the original autograd.Function's backward.
#
# Why do we need to redispatch to custom_function_call?
# -----------------------------------------------------
# This is consistent with how ATen operators work with functorch's grad transform:
# they always redispatch to the original operator.
# Consider torch.sin, and let's say we do grad0(grad1(torch.sin))(x)
#
# grad1 will:
# - set up the autograd graph
# - unwrap the inputs
# - redispatch to at::sin (*)
# - rewrap the outputs on the return
#
# On the redispatch in (*), grad0 will:
# - set up the autograd graph
# - unwrap the inputs
# - redispatch to at::sin
# - rewrap the outputs on the return
#
# To "set up the autograd graph", we generate a _SingleLevelFunction
# and apply it.
@custom_function_call.py_impl(TransformType.Grad)
@custom_function_call.py_impl(TransformType.Jvp)
def custom_function_call_grad(interpreter, autograd_function, *operands):
    Generated = generate_single_level_function(interpreter, autograd_function)
    with enable_autograd_function():
        flat_out = Generated.apply(*operands)
    return flat_out


def generate_single_level_function(interpreter, autograd_function):
    level = interpreter.level()

    def forward(*operands):
        unwrapped_operands = pytree.tree_map_only(
            torch.Tensor,
            lambda x: _unwrap_for_grad(x, level),
            operands)
        # Both enable_grad() and _set_fwd_grad_enabled() are necessary no matter
        # the transform. _SingleLevelFunction will turn off both fwd and bwd
        # gradient computation and we need to turn it back on here.
        with torch.enable_grad(), _set_fwd_grad_enabled(True), interpreter.lower():
            output = custom_function_call(autograd_function, *unwrapped_operands)

        return pytree.tree_map_only(
            torch.Tensor,
            lambda x: _wrap_for_grad(x, level),
            output)

    def setup_context(ctx, outputs, *operands):
        ctx.mark_dirty = mark_dirty_error
        return autograd_function.setup_context(ctx, outputs, *operands)

    # backward is only used if the transform is TransformType.Grad
    def backward(ctx, *grads):
        result = autograd_function.backward(ctx, *grads)
        return result

    # jvp is only used if the transform is TransformType.Jvp
    def jvp(ctx, *tangents):
        result = autograd_function.jvp(ctx, *tangents)
        return result

    # This is the sequence of magic words to dynamically generate a Subclass with
    # a given name. A Tensor's .grad_fn field has a class name that is the original
    # autograd.Function's name + Backward, so we do this to generate some
    # meaningful name.
    name = f'{autograd_function.__name__}Generated'
    Generated = type(
        name,
        (torch.autograd.function._SingleLevelFunction,),
        {
            'forward': staticmethod(forward),
            'backward': staticmethod(backward),
            'jvp': staticmethod(jvp),
            'setup_context': staticmethod(setup_context),
        },
    )
    return Generated


# https://github.com/pytorch/pytorch/issues/90225
# If an input was marked as dirty, and the autograd.Function returns the input
# from the forward, then the grad rule for custom_function_call must also
# return the corresponding input from the forward() of the Generated autograd.Function
#
# We haven't figured out how to do this yet. One possibility is to rely
# on if the return from the redispatched custom_function_call in Generated.forward
# has the same object id as one of the inputs,
# but https://github.com/pytorch/pytorch/issues/90209 means we cannot rely on
# that property.
def mark_dirty_error(*args, **kwargs):
    raise RuntimeError(
        'NYI: we do not yet support ctx.mark_dirty with functorch transforms. '
        'Please try to avoid modifying inputs to the autograd.Function in-place '
        'by using out-of-place operations or by cloning the inputs. '
        'Please see https://github.com/pytorch/pytorch/issues/90209 for more details'
    )


# NOTE: [functorch vjp and autograd interaction]
# There's an edge case with the functorch vjp and autograd interaction
# that will eventually be fixed by mode-only functorch.
# The TL;DR is that there's no way to unwrap a dead GradTensorWrapper,
# so we (the framework) need to do it manually. Regular PyTorch operators
# automatically do so this is consisent.
#
# class MyExp(torch.autograd.Function):
#     @staticmethod
#     def forward(x):
#         return x.exp()
#
#     @staticmethod
#     def setup_context(ctx, outputs, x):
#         y = outputs
#         ctx.save_for_backward(y)
#
#     @staticmethod
#     def backward(gy):
#         y, = ctx.saved_tensors()
#         return MyMul.apply(gy, y)
#
# x = torch.randn([], requires_grad=True)
# gy = torch.randn([], requires_grad=True)
# _, vjp_fn = vjp(MySin.apply, x)
# result = vjp_fn(gy)
#
# MyMul is an autograd.Function that is not shown here.
# It saves a `y` for backward (since gy requires grad).
#
# in vjp_fn(gy), we get:
# > MyMul.apply(gy, GradTensorWrapper(y, level=dead))
# Because the y that is saved for backward by MyExp is a GradTensorWrapper
# but is now dead since we are outside the vjp context.
#
# PyTorch dispatcher operations, upon seeing a dead GradTensorWrapper,
# will automatically unwrap the GradTensorWrapper when applied.
# But since autograd.Function technically sits above the regular PyTorch
# dispatcher, it doesn't get this treatment. So we manually do
# the unwrapping to be consistent with regular PyTorch dispatcher operations.


class VmapInfo(NamedTuple):
    batch_size: int
    randomness: str


@custom_function_call.py_impl(TransformType.Vmap)
def custom_function_call_vmap(interpreter, autograd_function, *operands):
    if not hasattr(autograd_function, "vmap"):
        # TODO: link docs when they're ready.
        # https://github.com/pytorch/pytorch/issues/90224
        raise RuntimeError(
            f"You tried to vmap over {autograd_function.__name__}, but "
            f"it does not have a vmap rule defined. Please add a vmap "
            f"staticmethod to it.")

    current_level = interpreter.level()
    info = VmapInfo(
        batch_size=interpreter.batch_size(),
        randomness=interpreter.randomness(),
    )
    unwrapped_operands, in_dims = unwrap_batched(operands, current_level)

    # If none of the tensors are batched at the current level, then we skip the
    # current level. This saves the user from needing to handle this case in
    # their vmap staticmethod (and is consistent with our C++ batching rule API)
    if pytree.tree_all(lambda dim: dim is None, in_dims):
        with interpreter.lower():
            return custom_function_call(autograd_function, *operands)

    with interpreter.lower():
        unwrapped_output, out_dims = autograd_function.vmap(info, in_dims, *unwrapped_operands)

    output = wrap_batched(unwrapped_output, out_dims, current_level)
    return output


def unwrap_batched(args, level):
    flat_args, spec = pytree.tree_flatten(args)
    if len(flat_args) == 0:
        return args, ()
    result = [_unwrap_batched(arg, level) if isinstance(arg, torch.Tensor)
              else (arg, None) for arg in flat_args]
    output, bdims = zip(*result)
    return pytree.tree_unflatten(output, spec), pytree.tree_unflatten(bdims, spec)


def wrap_batched(args, bdims, level):
    # TODO: raise better error message to the user when they don't follow the API.
    # Should probably mimic the logic of _process_batched_inputs,
    # but that one is hyperspecialized on error messages.
    # https://github.com/pytorch/pytorch/issues/90224
    flat_args, spec = pytree.tree_flatten(args)
    flat_bdims = _broadcast_to_and_flatten(bdims, spec)
    assert flat_bdims is not None
    result = _create_batched_inputs(flat_bdims, flat_args, level, spec)
    return result


@custom_function_call.py_impl(TransformType.Functionalize)
def custom_function_call_functionalize(interpreter, autograd_function, *operands):
    raise RuntimeError("NYI: Functionalize rule for custom_function_call")
