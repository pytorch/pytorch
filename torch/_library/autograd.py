# mypy: allow-untyped-defs
import dataclasses
from dataclasses import dataclass
from typing import Any, Callable, Optional, Protocol

from torch import _C, _ops, autograd, Tensor
from torch.utils import _pytree

from . import utils


class InfoProtocol(Protocol):
    _backward_fn: Optional[Callable]
    _setup_context_fn: Optional[Callable]


@dataclasses.dataclass
class Info:
    _backward_fn: Optional[Callable]
    _setup_context_fn: Optional[Callable]


def make_autograd_impl(op: _ops.OpOverload, info: InfoProtocol) -> Callable:
    name: str = f"GeneratedBackwardFor_{op._namespace}_{op._opname}_{op._overloadname}"

    has_kwarg_only_args = utils.has_kwarg_only_args(op._schema)

    @dataclass
    class Metadata:
        keyset: _C.DispatchKeySet
        keyword_only_args: dict[str, Any]

    def forward_no_grad(*args):
        metadata = args[-1]
        args = args[:-1]

        with _C._AutoDispatchBelowAutograd():
            keyset = metadata.keyset
            kwargs = metadata.keyword_only_args
            result = op.redispatch(keyset & _C._after_autograd_keyset, *args, **kwargs)
            return result

    def forward(ctx, *args):
        metadata = args[-1]
        args = args[:-1]

        with _C._AutoDispatchBelowAutograd():
            keyset = metadata.keyset
            kwargs = metadata.keyword_only_args
            result = op.redispatch(keyset & _C._after_autograd_keyset, *args, **kwargs)
            if info._setup_context_fn:
                # The Dispatcher will remove args that are equal to their default
                # values from (args, kwargs). We're going to add it back so that
                # the user can access them.
                #
                # This is OK to do: The Dispatcher removed the args for serialization
                # FC/BC reasons (that is, a graph will not store args that are equal
                # to their default values), but that doesn't matter here. If the user
                # adds a new default arg, then they must update
                # their setup_context (along with the rest of their operator
                # registrations)
                args, kwargs = utils.fill_defaults(op._schema, args, kwargs)

                if has_kwarg_only_args:
                    info._setup_context_fn(
                        ctx=ctx, inputs=args, keyword_only_inputs=kwargs, output=result
                    )
                else:
                    info._setup_context_fn(ctx=ctx, inputs=args, output=result)
            return result

    def backward(ctx, *grads):
        if info._backward_fn:
            try:
                prev_needs_input_grad = ctx.needs_input_grad
                ctx.needs_input_grad = ctx.needs_input_grad[:-1]
                result = info._backward_fn(ctx, *grads)
            finally:
                ctx.needs_input_grad = prev_needs_input_grad
            if isinstance(result, tuple):
                return (*result, None)
            return result, None
        raise RuntimeError(
            f"Trying to backward through {op} but no autograd "
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

    schema = op._schema
    if any(
        utils.is_tensorlist_like_type(a.type)
        for a in (*schema.arguments, *schema.returns)
    ):
        Generated = supports_tensorlist(Generated)

    # The dispatcher passes any keyword-only-args as kwargs and the
    # rest of the args (even if specified as kwargs) as args.
    def autograd_impl(keyset, *args, **keyword_only_args):
        if _C.is_grad_enabled() and _pytree.tree_any_only(
            Tensor, lambda x: x.requires_grad, args, not_list_of_tensor
        ):
            result = Generated.apply(*args, Metadata(keyset, keyword_only_args))  # type: ignore[attr-defined]
        else:
            result = forward_no_grad(*args, Metadata(keyset, keyword_only_args))
        return result

    return autograd_impl


def supports_tensorlist(cls: Any) -> Any:
    """Allows a given autograd.Function class to support List[Tensor] inputs/outputs.

    Regular autograd.Function has a constraint that it only directly supports autograd for
    Tensors. Applying @supports_tensorlist enables an autograd.Function to support
    autograd for List[Tensor] inputs and outputs.
    """
    orig_forward = cls.forward
    orig_backward = cls.backward
    orig_apply = cls.apply

    @dataclass
    class Metadata:
        input_spec: spec_t
        output_spec: Optional[spec_t] = None
        result_is_tuple: Optional[bool] = None

    def new_forward(ctx, *args):
        metadata = args[-1]
        args = args[:-1]
        if not isinstance(metadata, Metadata):
            raise NotImplementedError(
                "NYI: calling supports_tensorlist autograd.Function.forward directly. "
                "You should probably be calling .apply instead. "
                "Please file an issue if not."
            )
        args = unflatten(list(args), metadata.input_spec)
        result = orig_forward(ctx, *args)
        metadata.result_is_tuple = isinstance(result, tuple)
        if not metadata.result_is_tuple:
            result = (result,)
        flat_result, output_spec = flatten(result, not_list_of_tensor)
        metadata.output_spec = output_spec

        if hasattr(ctx, "_pt_metadata"):
            raise RuntimeError(
                "Please don't set ctx._pt_metadata; PyTorch uses it to store info"
            )
        ctx._pt_metadata = metadata

        return tuple(flat_result)

    def new_backward(ctx, *grads):
        if not hasattr(ctx, "_pt_metadata"):
            raise NotImplementedError(
                "NYI: calling supports_tensorlist autograd.Function.backward directly. "
                "This will automatically get called by PyTorch autograd. "
                "Please file an issue if you need this."
            )

        metadata = ctx._pt_metadata
        grads = unflatten(list(grads), metadata.output_spec)

        # If the user's input is ([x, y, z], w),
        # then needs_input_grad is (bool, bool, bool, bool, bool).
        # We need to
        # 1. get rid of the additional bool (which comes from the extra
        # `metadata input`)
        # 2. unflatten to get the right structure.
        prev_needs_input_grad = ctx.needs_input_grad
        try:
            ctx.needs_input_grad = unflatten(
                list(ctx.needs_input_grad[:-1]), metadata.input_spec
            )
            grad_inputs = orig_backward(ctx, *grads)
        finally:
            ctx.needs_input_grad = prev_needs_input_grad

        if not isinstance(grad_inputs, tuple):
            grad_inputs = (grad_inputs,)
        # Assume that any Nones in the backward are Tensors.
        # If the forward has an arg that is [1, 2, 3], the backward should
        # return None as the grad.
        # If the forward has an arg that is [tensor, tensor], the backward
        # may return [None, None], [grad, None], [None, grad], or [grad, grad].
        flat_grad_inputs, grad_inputs_spec = flatten(
            grad_inputs, not_list_of_optional_tensor
        )
        if grad_inputs_spec != metadata.input_spec:
            raise RuntimeError(
                f"Expected the return from backward to be of the same structure "
                f"as the inputs. Got: {grad_inputs_spec} (return from backward), "
                f"{metadata.input_spec} (inputs)"
            )
        return tuple(flat_grad_inputs + [None])

    def new_apply(*args):
        flat_args, input_spec = flatten(args, is_leaf=not_list_of_tensor)
        metadata = Metadata(input_spec)
        result = orig_apply(*flat_args, metadata)  # type: ignore[misc]
        assert metadata.output_spec is not None
        result = unflatten(list(result), metadata.output_spec)
        if not metadata.result_is_tuple:
            assert isinstance(result, tuple)
            assert len(result) == 1
            return result[0]
        return result

    cls.forward = new_forward
    cls.backward = new_backward
    cls.apply = new_apply
    return cls


def not_list_of_tensor(tree):
    if isinstance(tree, tuple):
        return False
    if isinstance(tree, list):
        return any(not isinstance(l, Tensor) for l in tree)
    return True


def not_list_of_optional_tensor(tree):
    if isinstance(tree, tuple):
        return False
    if isinstance(tree, list):
        return any(l is not None and not isinstance(l, Tensor) for l in tree)
    return True


flatten = _pytree.tree_flatten
unflatten = _pytree.tree_unflatten
spec_t = _pytree.TreeSpec
