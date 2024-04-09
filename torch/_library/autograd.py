from typing import Any, Callable, Optional

from .. import _C, autograd, Tensor
from . import utils


def make_autograd_impl(opdef: Any) -> Callable:
    name: str = f"GeneratedBackwardFor_{opdef._namespace}_{opdef._name}"

    def forward(ctx, *args):
        with _C._AutoDispatchBelowAutograd():
            result = opdef._opoverload(*args)
            if opdef._setup_context_fn:
                opdef._setup_context_fn(ctx, args, result)
            return result

    def backward(ctx, *grads):
        if opdef._backward_fn:
            return opdef._backward_fn(ctx, *grads)
        raise RuntimeError(
            f"Trying to backward through {opdef} but no autograd "
            f"formula was registered. "
            f"Please use register_autograd to add one."
        )

    Generated = type(
        name,
        (autograd.Function,),
        {
            "forward": staticmethod(forward),
            "backward": staticmethod(backward),
        },
    )

    schema = opdef._opoverload._schema
    if any(
        utils.is_tensorlist_like_type(a.type)
        for a in (*schema.arguments, *schema.returns)
    ):
        Generated = supports_tensorlist(Generated)

    def autograd_impl(*args):
        result = Generated.apply(*args)  # type: ignore[attr-defined]
        return result

    return autograd_impl


def supports_tensorlist(cls: Any) -> Any:
    """Allows a given autograd.Function class to support List[Tensor] inputs/outputs.

    Regular autograd.Function has a constraint that it only directly supports autograd for
    Tensors. Applying @supports_tensorlist enables an autograd.Function to support
    autograd for List[Tensor] inputs and outputs.
    """
    input_spec: Optional[spec_t] = None
    output_spec: Optional[spec_t] = None
    result_is_tuple = None

    orig_forward = cls.forward
    orig_backward = cls.backward
    orig_apply = cls.apply

    def new_forward(ctx, *args):
        assert input_spec is not None
        args = unflatten(list(args), input_spec)
        result = orig_forward(ctx, *args)
        nonlocal output_spec
        nonlocal result_is_tuple
        result_is_tuple = isinstance(result, tuple)
        if not result_is_tuple:
            result = (result,)
        flat_result, output_spec = flatten(result)
        return tuple(flat_result)

    def new_backward(ctx, *grads):
        assert output_spec is not None
        grads = unflatten(list(grads), output_spec)
        grad_inputs = orig_backward(ctx, *grads)
        if not isinstance(grad_inputs, tuple):
            grad_inputs = (grad_inputs,)
        flat_grad_inputs, grad_inputs_spec = flatten(grad_inputs)
        if grad_inputs_spec != input_spec:
            raise RuntimeError(
                "Expected the return from backward to be of the same structure "
                "as the inputs. Got: {grad_inputs_spec} (return from backward), "
                "{input_spec} (inputs)"
            )
        return tuple(flat_grad_inputs)

    def new_apply(*args):
        nonlocal input_spec
        flat_args, input_spec = flatten(args)
        result = orig_apply(*flat_args)  # type: ignore[misc]
        assert output_spec is not None
        result = unflatten(list(result), output_spec)
        if not result_is_tuple:
            assert isinstance(result, tuple)
            assert len(result) == 1
            return result[0]
        return result

    cls.forward = new_forward
    cls.backward = new_backward
    cls.apply = new_apply
    return cls


import functools

from ..utils import _pytree


def is_leaf(tree):
    if isinstance(tree, tuple):
        return False
    if isinstance(tree, list):
        return any(not isinstance(l, Tensor) for l in tree)
    return True


# Inputs/outputs to operators are at most TensorList, so we use a custom is_leaf
# to say that everything that is not a TensorList is a leaf.
flatten = functools.partial(_pytree.tree_flatten, is_leaf=is_leaf)
unflatten = _pytree.tree_unflatten
spec_t = _pytree.TreeSpec
