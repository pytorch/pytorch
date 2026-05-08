# mypy: allow-untyped-defs
import dataclasses
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from torch import _C, _ops, autograd, Tensor
from torch.utils import _pytree

from . import utils


class InfoProtocol(Protocol):
    _backward_fn: Callable | None
    _setup_context_fn: Callable | None


@dataclasses.dataclass
class Info:
    _backward_fn: Callable | None
    _setup_context_fn: Callable | None


def _codegen_autograd_forward(
    op: _ops.OpOverload,
    info: InfoProtocol,
    has_kwarg_only_args: bool,
) -> tuple[Callable, Callable]:
    code_globals: dict[str, object] = {
        "_AutoDispatchBelowAutograd_": _C._AutoDispatchBelowAutograd,
        "_after_autograd_keyset_": _C._after_autograd_keyset,
        "_op_": op,
    }

    lines: list[str] = []

    lines.append("def forward_no_grad(*args):")
    lines.append("    metadata = args[-1]")
    lines.append("    args = args[:-1]")
    lines.append("    with _AutoDispatchBelowAutograd_():")
    lines.append("        keyset = metadata.keyset")
    lines.append("        kwargs = metadata.keyword_only_args")
    lines.append(
        "        return _op_.redispatch(keyset & _after_autograd_keyset_, *args, **kwargs)"
    )

    lines.append("")
    lines.append("def forward(ctx, *args):")
    lines.append("    metadata = args[-1]")
    lines.append("    args = args[:-1]")
    lines.append("    with _AutoDispatchBelowAutograd_():")
    lines.append("        keyset = metadata.keyset")
    lines.append("        kwargs = metadata.keyword_only_args")

    if info._setup_context_fn:
        code_globals["_fill_defaults_"] = utils.fill_defaults
        code_globals["_schema_"] = op._schema
        code_globals["_setup_context_fn_"] = info._setup_context_fn
        lines.append(
            "        result = _op_.redispatch(keyset & _after_autograd_keyset_, *args, **kwargs)"
        )
        # The dispatcher strips args equal to their default values for
        # serialization forward/backward compatibility. We fill them back
        # so setup_context sees all declared args — adding a new default
        # arg requires the user to update setup_context anyway.
        lines.append("        args, kwargs = _fill_defaults_(_schema_, args, kwargs)")
        if has_kwarg_only_args:
            lines.append(
                "        _setup_context_fn_(ctx=ctx, inputs=args, keyword_only_inputs=kwargs, output=result)"
            )
        else:
            lines.append(
                "        _setup_context_fn_(ctx=ctx, inputs=args, output=result)"
            )
        lines.append("        return result")
    else:
        lines.append(
            "        return _op_.redispatch(keyset & _after_autograd_keyset_, *args, **kwargs)"
        )

    source = "\n".join(lines)
    code = compile(source, f"<custom_op_autograd_{op._namespace}_{op._opname}>", "exec")
    local_dict: dict[str, object] = {}
    exec(code, code_globals, local_dict)

    return local_dict["forward_no_grad"], local_dict["forward"]  # type: ignore[return-value]


def make_autograd_impl(op: _ops.OpOverload, info: InfoProtocol) -> Callable:
    name: str = f"GeneratedBackwardFor_{op._namespace}_{op._opname}_{op._overloadname}"

    has_kwarg_only_args = utils.has_kwarg_only_args(op._schema)

    @dataclass
    class Metadata:
        keyset: _C.DispatchKeySet
        keyword_only_args: dict[str, Any]

    forward_no_grad, forward = _codegen_autograd_forward(op, info, has_kwarg_only_args)

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
        if _C.is_grad_enabled() and _C._any_requires_grad(*args):
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
    class TensorListMetadata:
        input_spec: _pytree.TreeSpec
        output_spec: _pytree.TreeSpec | None = None
        result_is_tuple: bool | None = None

    def new_forward(ctx, *args):
        metadata = args[-1]
        args = args[:-1]
        if not isinstance(metadata, TensorListMetadata):
            raise NotImplementedError(
                "NYI: calling supports_tensorlist autograd.Function.forward directly. "
                "You should probably be calling .apply instead. "
                "Please file an issue if not."
            )
        args = _pytree.tree_unflatten(list(args), metadata.input_spec)
        result = orig_forward(ctx, *args)
        metadata.result_is_tuple = isinstance(result, tuple)
        if not metadata.result_is_tuple:
            result = (result,)
        flat_result, output_spec = _pytree.tree_flatten(result, not_list_of_tensor)
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
        grads = _pytree.tree_unflatten(list(grads), metadata.output_spec)

        # If the user's input is ([x, y, z], w),
        # then needs_input_grad is (bool, bool, bool, bool, bool).
        # We need to
        # 1. get rid of the additional bool (which comes from the extra
        # `metadata input`)
        # 2. _pytree.tree_unflatten to get the right structure.
        prev_needs_input_grad = ctx.needs_input_grad
        try:
            ctx.needs_input_grad = _pytree.tree_unflatten(
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
        flat_grad_inputs, grad_inputs_spec = _pytree.tree_flatten(
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
        flat_args, input_spec = _pytree.tree_flatten(args, is_leaf=not_list_of_tensor)
        metadata = TensorListMetadata(input_spec)
        result = orig_apply(*flat_args, metadata)  # type: ignore[misc]
        if metadata.output_spec is None:
            raise AssertionError("metadata.output_spec must not be None")
        result = _pytree.tree_unflatten(list(result), metadata.output_spec)
        if not metadata.result_is_tuple:
            if not isinstance(result, tuple):
                raise AssertionError(f"result must be tuple, got {type(result)}")
            if len(result) != 1:
                raise AssertionError(
                    f"result tuple must have length 1, got {len(result)}"
                )
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
