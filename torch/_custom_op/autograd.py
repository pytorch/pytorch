# mypy: allow-untyped-defs
import functools
from collections import namedtuple

import torch
import torch.utils._pytree as pytree


# NOTE [CustomOp autograd kernel indirection]
# We register `inner` as the autograd kernel for this custom_op.
# `inner` either calls the autograd formula registered by the user,
# or goes into an `autograd_not_implemented` kernel.
#
# The reason why this indirection exists is
# so that we can swap out the autograd kernel (the PyTorch dispatcher
# doesn't actually allow us to do this). By default, we want
# the `autograd_not_implemented` behavior, but then the user may come
# and register something that is actually a backward formula
def autograd_kernel_indirection(custom_op):
    autograd_fallback = autograd_not_implemented(custom_op)

    def inner(*args, **kwargs):
        if custom_op._has_impl("autograd"):
            kernel = custom_op._get_impl("autograd").func
            return kernel(*args, **kwargs)
        # As explained in NOTE ["backward", "save_for_backward", and "autograd"],
        # after the user gives us "backward" and "save_for_backward", we generate
        # the "autograd" impl. If the user only provided one, then we tell
        # the user they've done something wrong.
        if custom_op._has_impl("save_for_backward") or custom_op._has_impl("backward"):
            missing = (
                "save_for_backward" if custom_op._has_impl("backward") else "backward"
            )
            found = "save_for_backward" if missing == "backward" else "backward"
            loc = custom_op._get_impl(found).location
            raise RuntimeError(
                f"We found a '{found}' registration for {custom_op} at "
                f"{loc} but were unable to find a '{missing}' registration. "
                f"To use the CustomOp API to register a backward formula, "
                f"please provide us both a backward function and a "
                f"'save for backward' function via `impl_backward` and "
                f"`impl_save_for_backward` respectively."
            )
        return autograd_fallback(*args, **kwargs)

    return inner


# TODO(#101191): Use the actual C++ autograd not implemented fallback,
# or change the default autograd fallback to the autograd not implemented fallback.
def autograd_not_implemented(custom_op):
    def kernel(*args, **kwargs):
        if torch.is_grad_enabled() and pytree.tree_any(
            lambda x: isinstance(x, torch.Tensor) and x.requires_grad, (args, kwargs)
        ):
            raise RuntimeError("Autograd has not been implemented for operator")
        with torch._C._AutoDispatchBelowAutograd():
            return custom_op(*args, **kwargs)

    return kernel


def mark_non_differentiable(ctx, output, output_differentiability):
    # Output types are restricted to be:
    # - Tensor
    # - Tensor[]
    # - int, bool, Scalar, float
    # See _check_can_register_backward
    if output_differentiability is not None:
        if not isinstance(output, tuple):
            tuple_output = (output,)
        else:
            tuple_output = output  # type: ignore[assignment]
        assert len(output_differentiability) == len(tuple_output)
        non_differentiable_tensors = []
        for idx, (differentiable, out) in enumerate(
            zip(output_differentiability, tuple_output)
        ):
            if isinstance(out, torch.Tensor):
                if not differentiable:
                    non_differentiable_tensors.append(out)
                continue
            if isinstance(out, list):
                if not differentiable:
                    non_differentiable_tensors.extend(out)
                continue
            if differentiable:
                raise RuntimeError(
                    f"With output_differentiability={output_differentiability}. "
                    f"At idx {idx}, we received an object of type {type(out)} that "
                    f"is not a Tensor, so it cannot have be marked as differentiable in "
                    f"output_differentiability."
                )
        if non_differentiable_tensors:
            ctx.mark_non_differentiable(*non_differentiable_tensors)


def construct_autograd_kernel(
    schema,
    output_differentiability,
    custom_op,
    op_overload,
    save_for_backward_fn,
    backward_fn,
):
    def apply(*args):
        flat_args, spec = pytree.tree_flatten(args)
        out_spec = None

        def forward(ctx, *flat_args):
            ctx.set_materialize_grads(True)
            args = pytree.tree_unflatten(list(flat_args), spec)
            with torch._C._AutoDispatchBelowAutograd():
                output = op_overload(*args)

            # We use the info about args to give better error messages in backward
            args_info = namedtuple_args(schema, pytree.tree_map(type, args))

            save_for_backward_fn_inputs = namedtuple_args(schema, args)
            to_save = save_for_backward_fn(save_for_backward_fn_inputs, output)

            save_pytree_for_backward(ctx, (to_save, args_info))
            mark_non_differentiable(ctx, output, output_differentiability)

            nonlocal out_spec
            flat_output, out_spec = pytree.tree_flatten(output)
            return tuple(flat_output)

        def backward(ctx, *flat_grad_output):
            assert out_spec is not None
            grads = pytree.tree_unflatten(list(flat_grad_output), out_spec)
            saved, args_info = unpack_saved(ctx)
            # There is nothing on the ctx object for now, it is just there so
            # that we can add additional things in the future.
            inner_ctx = object()
            if not isinstance(grads, tuple):
                grads = (grads,)
            grad_inputs_dict = backward_fn(inner_ctx, saved, *grads)

            # Massage the grad_inputs_dict to a form acceptable by
            # autograd.Function.
            validate_grad_inputs_dict(grad_inputs_dict, custom_op, args_info)
            return grad_inputs_dict_to_flat_tuple(grad_inputs_dict, args_info)

        generated_cls = gen_autograd_function(
            custom_op._opname + "_customop", forward, backward
        )

        flat_output = generated_cls.apply(*flat_args)
        assert out_spec is not None
        return pytree.tree_unflatten(list(flat_output), out_spec)

    return apply


def gen_autograd_function(name, forward, backward):
    generated_cls = type(
        name,
        (torch.autograd.Function,),
        {
            "forward": staticmethod(forward),
            "backward": staticmethod(backward),
        },
    )
    return generated_cls


@functools.lru_cache
def namedtuple_args_cls(schema):
    attribs = [arg.name for arg in schema.arguments.flat_all]
    name = str(schema.name) + "_args"
    # mypy doesn't support dynamic namedtuple name
    tuple_cls = namedtuple(name, attribs)  # type: ignore[misc]
    return tuple_cls


def namedtuple_args(schema, args):
    assert isinstance(args, tuple)
    tuple_cls = namedtuple_args_cls(schema)
    return tuple_cls(*args)


def validate_grad_inputs_dict(grad_inputs_dict, forward_op, args_info):
    def error(what):
        backward = forward_op._get_impl("backward")
        raise RuntimeError(
            f"In the backward function defined for {forward_op} at "
            f"{backward.location} using the CustomOp API, {what}"
        )

    if not isinstance(grad_inputs_dict, dict):
        error(
            f"expected the output of the backward function to be a dict but "
            f"got {type(grad_inputs_dict)}"
        )

    expected_keys = {
        arg.name
        for arg in forward_op._schema.arguments.flat_all
        if arg.type.is_tensor_like()
    }
    actual_keys = grad_inputs_dict.keys()
    if expected_keys != actual_keys:
        error(
            f"expected the returned grad_input dict to have keys "
            f"{expected_keys} but got {actual_keys}. The backward "
            f"function must return a gradient (can be None) for each arg "
            f"to the CustomOp that may be a Tensor or Sequence[Tensor]. "
            f"Args declared to be non-Tensor-like types should not appear "
            f"in the grad_input dict"
        )

    for name, grad in grad_inputs_dict.items():
        arg_info = getattr(args_info, name)

        if isinstance(arg_info, list):
            if not isinstance(grad, (tuple, list)):
                error(
                    f"for input '{name}' expected the grad_input dict to "
                    f"hold a list of gradients but got object of type "
                    f"{type(grad)}."
                )
            if not len(grad) == len(arg_info):
                error(
                    f"for input '{name}' expected the grad_input dict to "
                    f"hold a list of {len(arg_info)} gradients but got "
                    f"{len(grad)}"
                )
            for idx, (g, info) in enumerate(zip(grad, arg_info)):
                if g is None:
                    continue
                if not isinstance(g, torch.Tensor):
                    error(
                        f"for input '{name}' expected the grad_input dict to "
                        f"hold a list of None or Tensor gradients but got "
                        f"object of {type(g)} at index {idx}"
                    )
                if not issubclass(info, torch.Tensor):
                    error(
                        f"for input '{name}', got a Tensor as the gradient "
                        f"for the {idx}-th value but expected None because "
                        f"the {idx}-th value was not a Tensor (it was "
                        f"type {arg_info}"
                    )
            continue

        if grad is None:
            continue
        if not isinstance(grad, torch.Tensor):
            error(
                f"got object of type {type(grad)} as the gradient for input "
                f"'{name}', "
                f"but expected the gradient to be either None or a Tensor"
            )
        if not issubclass(arg_info, torch.Tensor):
            error(
                f"got a Tensor as the gradient for input '{name}' but "
                f"expected None as the gradient because input '{name}' "
                f"was not a Tensor (it was type {arg_info})."
            )


def grad_inputs_dict_to_flat_tuple(grad_inputs_dict, args_info):
    result = []
    for name, arg_info in args_info._asdict().items():
        if name not in grad_inputs_dict:
            result.append(pytree.tree_map(lambda x: None, arg_info))
            continue
        result.append(grad_inputs_dict[name])
    return tuple(pytree.tree_leaves(result))


# Saves "stuff" (a pytree) onto the ctx object. Use unpack_saved to unpack it.
# autograd.Function prefers that users use ctx.save_for_backward to
# save Tensors (to avoid reference cycles) and for non-Tensors to go onto the
# ctx object.
def save_pytree_for_backward(ctx, stuff):
    flat_stuff, spec = pytree.tree_flatten(stuff)
    num_elts = len(flat_stuff)
    tensor_idxs = [
        idx for idx, thing in enumerate(flat_stuff) if isinstance(thing, torch.Tensor)
    ]
    non_tensor_idxs = [
        idx
        for idx, thing in enumerate(flat_stuff)
        if not isinstance(thing, torch.Tensor)
    ]
    tensors = [thing for thing in flat_stuff if isinstance(thing, torch.Tensor)]
    non_tensors = [thing for thing in flat_stuff if not isinstance(thing, torch.Tensor)]

    ctx.spec = spec
    ctx.num_elts = num_elts
    ctx.save_for_backward(*tensors)
    ctx.tensor_idxs = tensor_idxs
    ctx.saved_non_tensors = non_tensors
    ctx.non_tensor_idxs = non_tensor_idxs


# Inverse operation to save_pytree_for_backward
def unpack_saved(ctx):
    flat_stuff = [None] * ctx.num_elts
    for tensor, idx in zip(ctx.saved_tensors, ctx.tensor_idxs):
        flat_stuff[idx] = tensor
    for non_tensor, idx in zip(ctx.saved_non_tensors, ctx.non_tensor_idxs):
        flat_stuff[idx] = non_tensor
    stuff = pytree.tree_unflatten(flat_stuff, ctx.spec)
    return stuff
