# mypy: allow-untyped-defs
from typing import Any, NamedTuple, Tuple

import torch
import torch.utils._pytree as pytree
from torch._C._functorch import (
    _unwrap_for_grad,
    _wrap_for_grad,
    current_level,
    TransformType,
)
from torch._functorch.apis import vmap
from torch._functorch.utils import enable_single_level_autograd_function
from torch._functorch.vmap import (
    _add_batch_dim,
    _broadcast_to_and_flatten,
    restore_vmap,
    unwrap_batched,
    wrap_batched,
)
from torch._ops import HigherOrderOperator
from torch.autograd.forward_ad import _set_fwd_grad_enabled


# autograd.Function technically runs before the regular PyTorch dispatcher.
# This is how features like autocast and torch_dispatch (e.g. PythonTLSSnapshot)
# work with it. One day we might decide to change this, but until then,
# we need to give the illusion that autograd.Function runs before those things.
#
# We do this by using creating a custom HigherOrderOperator that only functorch
# dispatches specially.
class CustomFunctionHigherOrderOperator(HigherOrderOperator):
    def __init__(self):
        super().__init__("custom_function_call")

    def __call__(self, autograd_function, *args, **kwargs):
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
            return super().__call__(autograd_function, *args, **kwargs)
        return autograd_function.apply(*args, **kwargs)


# "custom_function_call"
# This is the mechanism for an autograd.Function that works with functorch transforms.
# It wraps an autograd.Function; interactions with functorch transforms are defined
# via PyDispatcher and HigherOrderOperator rather than through the traditional PyTorch
# dispatcher.
custom_function_call = CustomFunctionHigherOrderOperator()


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
    with enable_single_level_autograd_function():
        flat_out = Generated.apply(*operands)
    return flat_out


def generate_single_level_function(interpreter, autograd_function):
    level = interpreter.level()

    def forward(*operands):
        unwrapped_operands = pytree.tree_map_only(
            torch.Tensor, lambda x: _unwrap_for_grad(x, level), operands
        )
        # Both enable_grad() and _set_fwd_grad_enabled() are necessary no matter
        # the transform. _SingleLevelFunction will turn off both fwd and bwd
        # gradient computation and we need to turn it back on here.
        with torch.enable_grad(), _set_fwd_grad_enabled(True), interpreter.lower():
            unwrapped_output = custom_function_call(
                autograd_function, *unwrapped_operands
            )

        # See NOTE [mark_dirty object identity check]
        def wrap_fn(output):
            return _wrap_for_grad(output, level)

        return wrap_outputs_maintaining_identity(
            unwrapped_output, unwrapped_operands, operands, wrap_fn
        )

    def setup_context(ctx, inputs, output):
        return autograd_function.setup_context(ctx, inputs, output)

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
    name = f"{autograd_function.__name__}Generated"
    Generated = type(
        name,
        (torch.autograd.function._SingleLevelFunction,),
        {
            "forward": staticmethod(forward),
            "backward": staticmethod(backward),
            "jvp": staticmethod(jvp),
            "setup_context": staticmethod(setup_context),
        },
    )
    return Generated


# wrap_outputs_maintaining_identity handles outputs from the vmap,
# backward (vjp), and jvp staticmethod. The way it distinguishes
# between the vmap case and the {backward, jvp} case is if the out_dims
# are specified or not.
#
# NB: we cannot use out_dims=None as the deciding factor. This because
# out_dims=None can still happen in the vmap staticmethod! What the
# user is saying in that case is that their output does not have a
# dimension that is being vmapped over, which is valid.
NO_OUT_DIMS = "not specified"


# NOTE [mark_dirty object identity check]
# autograd.Function's ctx.mark_dirty expect a returned input
# to have the same object identity as the input.
# Mode-only functorch will greatly simplify this logic.
def wrap_outputs_maintaining_identity(
    outputs, unwrapped_inputs, orig_inputs, wrap_fn, out_dims=NO_OUT_DIMS
):
    flat_unwrapped_inputs = pytree.arg_tree_leaves(*unwrapped_inputs)
    flat_orig_inputs = pytree.arg_tree_leaves(*orig_inputs)

    unwrapped_input_to_orig_input = {
        id(unwrapped): orig
        for unwrapped, orig in zip(flat_unwrapped_inputs, flat_orig_inputs)
    }

    flat_outputs, spec = pytree.tree_flatten(outputs)
    result = []

    out_dims_specified = out_dims != NO_OUT_DIMS

    if out_dims_specified:
        flat_out_dims = _broadcast_to_and_flatten(out_dims, spec)
        # _broadcast_to_and_flatten returns None if it is unable to broadcast.
        # TODO: update following link from master to stable once that's out
        if flat_out_dims is None:
            raise RuntimeError(
                f"The autograd.Function's vmap staticmethod returned an "
                f"incompatible (output, out_dims) tuple. "
                f"Expected out_dims={out_dims} "
                f"to be compatible with the structure of `output`. "
                f"out_dims has structure {pytree.tree_flatten(out_dims)[1]} "
                f"but output has structure {spec}. "
                f"For more details, please see "
                f"https://pytorch.org/docs/main/notes/extending.func.html"
            )

    for i, output in enumerate(flat_outputs):
        if not isinstance(output, torch.Tensor):
            result.append(output)
            continue
        if id(output) in unwrapped_input_to_orig_input:
            result.append(unwrapped_input_to_orig_input[id(output)])
            continue
        if out_dims_specified:
            result.append(wrap_fn(output, flat_out_dims[i]))  # type: ignore[possibly-undefined, index]
        else:
            result.append(wrap_fn(output))

    return pytree.tree_unflatten(result, spec)


# NOTE: [functorch vjp and autograd interaction]
# There's an edge case with the functorch vjp and autograd interaction
# that will eventually be fixed by mode-only functorch.
# The TL;DR is that there's no way to unwrap a dead GradTensorWrapper,
# so we (the framework) need to do it manually. Regular PyTorch operators
# automatically do so this is consistent.
#
# class MyExp(torch.autograd.Function):
#     @staticmethod
#     def forward(x):
#         return x.exp()
#
#     @staticmethod
#     def setup_context(ctx, inputs, output):
#         y = output
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


def has_overriden_vmap_rule(autograd_function):
    return autograd_function.vmap is not torch.autograd.Function.vmap


def validate_vmap_returns_tuple_of_two_elements(result):
    base_error_msg = (
        "Expected the vmap staticmethod to have two returns, an output "
        "and out_dims with pytree structure compatible with the output. "
    )
    if not isinstance(result, tuple):
        raise RuntimeError(base_error_msg + f"Got a {type(result)} instead")
    if not len(result) == 2:
        raise RuntimeError(base_error_msg + f"Got {len(result)} returns instead")


@custom_function_call.py_impl(TransformType.Vmap)
def custom_function_call_vmap(interpreter, autograd_function, *operands):
    if autograd_function.generate_vmap_rule:
        if has_overriden_vmap_rule(autograd_function):
            # TODO: Update link to stable once that's out
            # https://github.com/pytorch/pytorch/issues/92029
            raise RuntimeError(
                f"You tried to vmap over {autograd_function.__name__}, but "
                f"it has both generate_vmap_rule=True and an overriden vmap "
                f"staticmethod. Please set generate_vmap_rule=False or delete "
                f"the overriden vmap staticmethod to avoid ambiguity. "
                f"For more details, please see "
                f"https://pytorch.org/docs/main/notes/extending.func.html"
            )
        return custom_function_call_vmap_generate_rule(
            interpreter, autograd_function, *operands
        )

    if not has_overriden_vmap_rule(autograd_function):
        # TODO: Update link to stable once that's out
        # https://github.com/pytorch/pytorch/issues/92029
        raise RuntimeError(
            f"You tried to vmap over {autograd_function.__name__}, but "
            f"it does not have vmap support. Please override and implement the "
            f"vmap staticmethod or set generate_vmap_rule=True. "
            f"For more details, please see "
            f"https://pytorch.org/docs/main/notes/extending.func.html"
        )

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
        result = autograd_function.vmap(info, in_dims, *unwrapped_operands)
    validate_vmap_returns_tuple_of_two_elements(result)
    unwrapped_output, out_dims = result

    # See NOTE [mark_dirty object identity check]
    def wrap_fn(output, out_dim):
        return (
            output
            if out_dim is None
            else _add_batch_dim(output, out_dim, current_level)
        )

    return wrap_outputs_maintaining_identity(
        unwrapped_output, unwrapped_operands, operands, wrap_fn, out_dims=out_dims
    )


def custom_function_call_vmap_generate_rule(interpreter, autograd_function, *operands):
    unwrapped_operands, in_dims = unwrap_batched(operands, interpreter.level())
    vmapped_function, get_out_dims = vmapify_autograd_function(
        autograd_function, in_dims, interpreter.batch_size(), interpreter.randomness()
    )

    with interpreter.lower():
        output = custom_function_call(vmapped_function, *unwrapped_operands)

    out_dims = get_out_dims()
    return wrap_batched(output, out_dims, interpreter.level())


@custom_function_call.py_impl(TransformType.Functionalize)
def custom_function_call_functionalize(
    interpreter, autograd_function, generate_vmap_rule, *operands
):
    raise RuntimeError("NYI: Functionalize rule for custom_function_call")


def vmapify_autograd_function(autograd_function, in_dims, batch_size, randomness):
    # The following values are saved from the forward() and setup_context()
    # and used in backward().
    # Why do we save the values out here instead of on the ctx object?
    # - out_dims: There's no way to retrieve this from forward()
    # - input_shapes, saved_tensors_bdims: I'm a bit scared of nesting
    #   vmap(vmap( but not completely sure if it is a problem. If we
    #   assigned those fields to the ctx object, the worry is that they
    #   get overwritten.
    init_val = "not populated"
    out_dims = init_val
    input_shapes: Any = init_val
    saved_tensors_bdims: Any = init_val

    def forward(*operands):
        nonlocal out_dims
        outputs, out_dims = restore_vmap(
            autograd_function.forward, in_dims, batch_size, randomness
        )(*operands)
        return outputs

    def setup_context(ctx, inputs, outputs):
        input_shapes_ = None
        saved_tensors_bdims_ = None

        def inner(inputs, outputs):
            # wrapped_ctx.save_for_backward will:
            # - unwrap batchedtensors into (tensor, bdim)
            # - save_for_backward(*unwrapped_tensors)
            # - assign the bdims to wrapped_ctx._pt_saved_tensors_bdims
            wrapped_ctx = CtxCustomSave(ctx, current_level())
            autograd_function.setup_context(wrapped_ctx, inputs, outputs)

            # input_shapes are used for reductify later to reduce expanded gradients
            # to the correct shape.
            # See NOTE: [Why can't we rely on autograd to reduce expanded gradients?]
            # for more details
            nonlocal input_shapes_
            input_shapes_ = tuple(
                inp.shape if isinstance(inp, torch.Tensor) else None for inp in inputs
            )
            nonlocal saved_tensors_bdims_
            saved_tensors_bdims_ = wrapped_ctx._pt_saved_tensors_bdims

        # See NOTE: [Why do we need to run setup_context under a vmap?]
        restore_vmap(
            inner,
            (in_dims, out_dims),
            batch_size,
            randomness,
        )(inputs, outputs)

        nonlocal input_shapes
        input_shapes = input_shapes_
        nonlocal saved_tensors_bdims
        saved_tensors_bdims = saved_tensors_bdims_

    def jvp(ctx, *tangents):
        assert out_dims != init_val
        assert saved_tensors_bdims != init_val

        def jvp_no_context(saved_tensors, tangents):
            wrapped_ctx = CtxWithSavedTensors(ctx, saved_tensors)
            return autograd_function.jvp(wrapped_ctx, *tangents)

        tangent_in_dims = get_tangents_in_dims(in_dims, tangents)
        out_tangents, out_tangents_dims = restore_vmap(
            jvp_no_context,
            (saved_tensors_bdims, tangent_in_dims),
            batch_size,
            randomness,
        )(ctx.saved_tensors, tangents)

        result = reductify(out_tangents, out_tangents_dims, out_dims, batch_size)
        return result

    def backward(ctx, *grad_outputs):
        assert out_dims != init_val
        assert input_shapes != init_val
        assert saved_tensors_bdims != init_val

        def backward_no_context(inputs):
            saved_tensors, grad_outputs = inputs
            wrapped_ctx = CtxWithSavedTensors(ctx, saved_tensors)
            return autograd_function.backward(wrapped_ctx, *grad_outputs)

        grad_ins, grad_ins_dims = restore_vmap(
            backward_no_context,
            ((saved_tensors_bdims, out_dims),),
            batch_size,
            randomness,
        )((ctx.saved_tensors, grad_outputs))
        result = reductify(grad_ins, grad_ins_dims, in_dims, batch_size, input_shapes)
        return result

    name = f"Vmapped{autograd_function.__name__}"
    Generated = type(
        name,
        (torch.autograd.Function,),
        {
            "forward": staticmethod(forward),
            "backward": staticmethod(backward),
            "jvp": staticmethod(jvp),
            "setup_context": staticmethod(setup_context),
            "generate_vmap_rule": True,
        },
    )

    def get_out_dims():
        assert out_dims != init_val
        return out_dims

    return Generated, get_out_dims


# tangents might be None, so we need to replace
# the corresponding in_dims with None.
def get_tangents_in_dims(input_dims, tangents):
    flat_in_dims, spec = pytree.tree_flatten(input_dims)
    flat_tangents = pytree.arg_tree_leaves(*tangents)
    result = [
        None if tangent is None else in_dim
        for in_dim, tangent in zip(flat_in_dims, flat_tangents)
    ]
    return pytree.tree_unflatten(result, spec)


# NOTE: [Why do we need to run setup_context under a vmap?]
# Consider the following autograd.Function
#
# class Sum(torch.autograd.Function):
#    @staticmethod
#    def forward(x):
#        return x.sum()
#    @staticmethod
#    def setup_context(ctx, inputs, outputs):
#        ctx.x_shape = inputs[0]
#    @staticmethod
#    def backward(ctx, gy):
#        return gy.expand(ctx.x_shape)
#
# x = torch.randn(B, 4)
# in_dims = 0
# vmap(Sum.apply, in_dims)(x)
#
# Let's assume for a moment that we didn't vmap setup_context in VmappedSum:
#
# class VmappedSum(torch.autograd.Function):
#    @staticmethod
#    def forward(x):
#        return vmap(Sum.forward, in_dims)(x)
#
#    @staticmethod
#    def setup_context(ctx, inputs, outputs):
#        Sum.setup_context(ctx, inputs, outputs)
#
#    @staticmethod
#    def backward(ctx, gy):
#        def backward_no_context(gy):
#            return gy.expand(ctx.x_shape)
#
#        dims = (0,)
#        gx = vmap(backward_no_context, dims)(gy)
#        return gx
#
# We end up saving [B, 4] as x_shape. In the backward, gy has shape [B],
# and we're doing:
#
# def backward_no_context(gy):
#     return gy.expand([B, 4])
#
# gx = vmap(backward_no_context, dims)(gy: "Tensor[B]")
#
# This gives us the wrong result (gx has shape [B, B, 4], but it should
# have shape [4]). Performing vmap over setup_context means the shape
# saved has shape [4] and leads to a correct result shape for gx.


# Wraps a ctx object. Forwards all attr accesses to the underlying object
# except for the attrs in _pt_attrs
class WrappedCtx:
    _pt_reserved_attrs: Tuple[str, ...] = ("_pt_reserved_attrs", "_pt_inner_ctx")

    def __init__(self, ctx):
        if not isinstance(ctx, WrappedCtx):
            reserved_attrs = type(self)._pt_reserved_attrs
            for name in reserved_attrs:
                if not hasattr(ctx, name):
                    continue
                raise RuntimeError(
                    f"PyTorch reserves the {reserved_attrs} field on ctx. "
                    "Please name your fields on ctx something else to avoid name "
                    "collision."
                )
        self._pt_inner_ctx = ctx

    def __getattr__(self, name):
        return getattr(self._pt_inner_ctx, name)

    def __setattr__(self, name, value):
        if name in type(self)._pt_reserved_attrs:
            self.__dict__[name] = value
            return
        return setattr(self._pt_inner_ctx, name, value)


# Wraps ctx to create a new ctx object that overrides saved_tensors.
class CtxWithSavedTensors(WrappedCtx):
    _pt_reserved_attrs = ("_pt_new_saved_tensors", *WrappedCtx._pt_reserved_attrs)

    def __init__(self, ctx, new_saved_tensors):
        super().__init__(ctx)
        self._pt_new_saved_tensors = new_saved_tensors

    @property
    def saved_tensors(self):
        return self._pt_new_saved_tensors


class CtxCustomSave(WrappedCtx):
    _pt_reserved_attrs = (
        "_pt_saved_tensors_bdims",
        "_pt_current_level",
        *WrappedCtx._pt_reserved_attrs,
    )

    def __init__(self, ctx, current_level):
        super().__init__(ctx)
        self._pt_saved_tensors_bdims = ()
        self._pt_current_level = current_level

    def save_for_backward(self, *tensors):
        unwrapped_tensors, bdims = unwrap_batched(tensors, self._pt_current_level)
        self._pt_inner_ctx.save_for_backward(*unwrapped_tensors)
        self._pt_saved_tensors_bdims = bdims

    def save_for_forward(self, *tensors):
        unwrapped_tensors, bdims = unwrap_batched(tensors, self._pt_current_level)
        self._pt_inner_ctx.save_for_forward(*unwrapped_tensors)
        self._pt_saved_tensors_bdims = bdims


def reductify(
    grad_input,
    grad_input_bdim,
    input_bdim,
    batch_size,
    target_shape_without_bdim_to_reduce_to=None,
):
    if not isinstance(grad_input, tuple):
        grad_input = (grad_input,)
    if not isinstance(grad_input_bdim, tuple):
        grad_input_bdim = (grad_input_bdim,)
    if not isinstance(input_bdim, tuple):
        input_bdim = (input_bdim,)

    if target_shape_without_bdim_to_reduce_to is None:
        target_shape_without_bdim_to_reduce_to = len(grad_input) * (None,)
    result = tuple(
        reductify_leaf(gi, gi_bdim, i_bdim, batch_size, maybe_ishape)
        for gi, gi_bdim, i_bdim, maybe_ishape in zip(
            grad_input,
            grad_input_bdim,
            input_bdim,
            target_shape_without_bdim_to_reduce_to,
        )
    )
    return result


def reductify_leaf(
    grad_input,
    grad_input_bdim,
    input_bdim,
    batch_size,
    target_shape_without_bdim_to_reduce_to=None,
):
    if grad_input is None:
        return None

    if grad_input_bdim is None and input_bdim is None:
        return grad_input

    if grad_input_bdim is not None and input_bdim is None:
        return grad_input.sum(grad_input_bdim)

    # NOTE: [Why can't we rely on autograd to reduce expanded gradients?]
    # For reverse-mode AD,
    # given a grad_input and input, it is valid for the user to return a
    # grad_input that has a broadcasted shape when compared to the input.
    # In this situation, autograd automatically reduces the grad_input to
    # the shape of the input.
    #
    # However, when input_bdim is not None, we have problems.
    #
    # [example 1]
    # grad_input: Tensor[3, 4], input: Tensor[B, 4]
    # We can expand grad_input to Tensor[B, 3, 4], but that isn't broadcastable
    # from [B, 4].
    #
    # [example 2]
    # grad_input: Tensor[3, B, 4], input: Tensor[B, 4]
    # We can swizzle grad_input to Tensor[B, 3, 4], but that isn't broadcastable
    # from [B, 4].
    #
    # This means that we need to also reduce the grad_input to the shape of the
    # input. This behavior is controlled by the `target_shape_without_bdim_to_reduce_to` flag;
    # if not-None then we do the reducing manually, otherwise, we do not do a reduction.
    assert input_bdim is not None

    if grad_input_bdim is None:
        grad_input = grad_input.unsqueeze(input_bdim)
        new_shape = list(grad_input.shape)
        new_shape[input_bdim] = batch_size
        grad_input = grad_input.expand(new_shape)
        grad_input_bdim = input_bdim

    if target_shape_without_bdim_to_reduce_to is not None:
        return vmap(
            torch.Tensor.sum_to_size,
            in_dims=(grad_input_bdim, None),
            out_dims=input_bdim,
        )(grad_input, target_shape_without_bdim_to_reduce_to)

    if input_bdim != grad_input_bdim:
        grad_input = grad_input.movedim(grad_input_bdim, input_bdim)
    return grad_input


def autograd_function_forward_rewritten(original_forward, original_setup_context):
    def new_forward(ctx, *args, **kwargs):
        output = original_forward(*args, **kwargs)
        original_setup_context(ctx, args, output)
        return output

    return new_forward


class AutogradFunctionApply(HigherOrderOperator):
    def __init__(self):
        super().__init__("autograd_function_apply")

    def __call__(self, fwd, bwd, *fwd_args, **fwd_kwargs):
        saved_values = None
        args_tensor_mask = fwd_kwargs["args_tensor_mask"]
        length_of_tensor_args = sum(args_tensor_mask)
        # Filter out the original tensor args from fwd_args,
        # lifted freevars should not be args of ApplyTemplate.apply
        # since we don't need to calculate the gradients of them.
        new_fwd_args = fwd_args[:length_of_tensor_args]

        class ApplyTemplate(torch.autograd.Function):
            @staticmethod
            def forward(ctx, *args):
                nonlocal saved_values
                output, saved_values = fwd(None, *fwd_args)
                return output

            @staticmethod
            def backward(ctx, *grad):
                return bwd(None, *grad, *saved_values)

        return ApplyTemplate.apply(*new_fwd_args)


autograd_function_apply = AutogradFunctionApply()
