import contextlib
from collections.abc import Callable, Sequence
from typing import Any

import torch
import torch.utils._pytree as pytree
from torch._C import DispatchKey, DispatchKeySet
from torch._higher_order_ops.utils import register_fake
from torch._ops import HigherOrderOperator
from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode, track_tensor_tree
from torch.nn.utils.stateless import _reparametrize_module

from .flat_apply import func_to_graphable


leaf_function_module_inputs: dict[str, torch.nn.Module] = {}


def invoke_with_flattened_inputs(fn, input_spec, *flat_args):
    with reconstruct_original_args(input_spec, flat_args) as (args, kwargs):
        return fn(*args, **kwargs)


def autograd_grad_with_mixed_inputs(
    outputs: Sequence[Any],
    inputs: Sequence[Any],
    grad_outputs: Sequence[Any] | None = None,
    retain_graph: bool | None = None,
    create_graph: bool = False,
    allow_unused: bool = False,
) -> tuple[torch.Tensor | None, ...]:
    """
    A helper wrapper for torch.autograd.grad that handles mixed inputs.

    This wrapper handles the case where:
    - outputs can be a mix of tensors (with or without grad_fn) and non-tensors
    - inputs can be a mix of tensors (with or without requires_grad) and non-tensors

    For each element in inputs:
    - If it's not a tensor: returns None
    - If it's a tensor without requires_grad: returns None
    - If it's a tensor with requires_grad: returns the gradient from torch.autograd.grad

    Args:
        outputs: Sequence of outputs (tensors and non-tensors)
        inputs: Sequence of inputs (tensors and non-tensors)
        grad_outputs: Gradients w.r.t. each output (matching the structure of outputs)
        retain_graph: Whether to retain the computation graph
        create_graph: Whether to create a graph of the derivative
        allow_unused: Whether to allow unused inputs

    Returns:
        Tuple of gradients, with None for non-tensor or non-requires_grad inputs
    """
    # Step 1: Filter outputs - keep only tensors with grad_fn
    filtered_outputs = []
    filtered_grad_outputs = []

    for i, out in enumerate(outputs):
        if isinstance(out, torch.Tensor) and out.grad_fn is not None:
            filtered_outputs.append(out)
            if grad_outputs is not None:
                filtered_grad_outputs.append(grad_outputs[i])

    # Step 2: Filter inputs - keep only tensors with requires_grad
    # Also track the original indices for reconstruction
    filtered_inputs = []
    input_indices = []  # Maps filtered index -> original index

    for i, inp in enumerate(inputs):
        if isinstance(inp, torch.Tensor) and inp.requires_grad:
            filtered_inputs.append(inp)
            input_indices.append(i)

    # Step 3: Handle edge cases
    if not filtered_outputs or not filtered_inputs:
        return tuple(None for _ in inputs)

    # Step 4: Call torch.autograd.grad on filtered inputs/outputs
    grad_outputs_arg = filtered_grad_outputs if grad_outputs is not None else None

    grads = torch.autograd.grad(
        outputs=filtered_outputs,
        inputs=filtered_inputs,
        grad_outputs=grad_outputs_arg,
        retain_graph=retain_graph,
        create_graph=create_graph,
        allow_unused=allow_unused,
    )

    # Step 5: Reconstruct the full gradient tuple with Nones at proper positions
    result: list[torch.Tensor | None] = [None] * len(inputs)
    for filtered_idx, original_idx in enumerate(input_indices):
        result[original_idx] = grads[filtered_idx]

    return tuple(result)


def create_fn_with_grad(
    fn: Callable,
    input_spec: pytree.TreeSpec,
    include_key_set: DispatchKeySet,
    exclude_key_set: DispatchKeySet,
    python_dispatcher_active: bool,
) -> tuple[Callable, Callable]:
    import functools

    leaf_fn_fw_inputs = None
    leaf_fn_fw_outputs = None

    @functools.wraps(fn)
    def fw_fn(*args):
        nonlocal leaf_fn_fw_inputs
        leaf_fn_fw_inputs = args

        # Compute the effective include/exclude sets
        # If PythonDispatcher was active in forward but TLS is no longer set up,
        # we need to exclude it to avoid the "PythonDispatcherTLS was not set" error
        effective_include = include_key_set
        effective_exclude = exclude_key_set

        # Check if PythonDispatcher TLS is currently available
        # If it was active before but not now, we need to exclude it
        current_python_dispatcher = torch._C._dispatch_tls_is_dispatch_key_included(
            DispatchKey.PythonDispatcher
        )
        if python_dispatcher_active and not current_python_dispatcher:
            # PythonDispatcher was active in forward but TLS is no longer set up
            # Remove it from include set and add to exclude set
            effective_include = effective_include - DispatchKeySet(
                DispatchKey.PythonDispatcher
            )
            effective_exclude = effective_exclude | DispatchKeySet(
                DispatchKey.PythonDispatcher
            )

        with torch._C._ForceDispatchKeyGuard(effective_include, effective_exclude):
            with torch.enable_grad():
                nonlocal leaf_fn_fw_outputs
                leaf_fn_fw_outputs = invoke_with_flattened_inputs(fn, input_spec, *args)
        return leaf_fn_fw_outputs

    def bw_fn(*grads):
        if leaf_fn_fw_inputs is None or leaf_fn_fw_outputs is None:
            raise RuntimeError(
                "For invoke_leaf_funcion, backward fn expects fw outputs/inputs to be set in forward."
            )
        return autograd_grad_with_mixed_inputs(
            outputs=leaf_fn_fw_outputs,
            inputs=leaf_fn_fw_inputs,
            grad_outputs=grads,
            allow_unused=True,
        )

    return fw_fn, bw_fn


def unwrap_fn_spec(fn_spec: pytree.TreeSpec) -> Callable:
    return pytree.tree_unflatten((), fn_spec)


@contextlib.contextmanager
def reconstruct_original_args(input_spec, flat_args):
    reparametrize_contexts = []
    try:
        if input_spec is None:
            yield flat_args, {}

        else:
            args, kwargs = pytree.tree_unflatten(flat_args, input_spec)

            def _retrive_orig_module(source: str):
                # We need a way to pass around the locals and globals from user code
                mod = leaf_function_module_inputs[source]
                assert isinstance(mod, torch.nn.Module), (
                    f"Expecting the object reference by source {source} to nn.Module."
                )
                return mod

            new_args = []
            for arg in args:
                if (
                    isinstance(arg, dict)
                    and "source_name" in arg
                    and "named_parameters" in arg
                    and "named_buffers" in arg
                ):
                    orig_module = _retrive_orig_module(arg["source_name"])
                    named_parameters = arg["named_parameters"]
                    named_buffers = arg["named_buffers"]
                    # Enter _reparametrize_module context
                    ctx = _reparametrize_module(
                        orig_module, {**named_parameters, **named_buffers}
                    )
                    ctx.__enter__()
                    reparametrize_contexts.append(ctx)
                    new_args.append(orig_module)
                else:
                    new_args.append(arg)
            yield new_args, kwargs

    finally:
        # Exit all _reparametrize_module contexts in reverse order
        for ctx in reversed(reparametrize_contexts):
            ctx.__exit__(None, None, None)


class InvokeLeafFunction(HigherOrderOperator):
    def __init__(self):
        super().__init__("invoke_leaf_function")

    def __call__(self, real_fn_spec, fake_fn_spec, input_spec, *flat_args):
        return super().__call__(real_fn_spec, fake_fn_spec, input_spec, *flat_args)  # type: ignore[attr-defined]


invoke_leaf_function = InvokeLeafFunction()


class InvokeLeafFunctionAutogradOp(torch.autograd.Function):
    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(ctx, real_fn_spec, fake_fn_spec, input_spec, *flat_args):
        real_fn = unwrap_fn_spec(real_fn_spec)
        fake_fn = unwrap_fn_spec(fake_fn_spec)

        include_key_set = torch._C._dispatch_tls_local_include_set()
        exclude_key_set = torch._C._dispatch_tls_local_exclude_set()
        python_dispatcher_active = include_key_set.has(DispatchKey.PythonDispatcher)

        real_fn_with_grad, bw_real_fn = create_fn_with_grad(
            real_fn,
            input_spec,
            include_key_set,
            exclude_key_set,
            python_dispatcher_active,
        )
        fake_fn_with_grad, bw_fake_fn = create_fn_with_grad(
            fake_fn,
            input_spec,
            include_key_set,
            exclude_key_set,
            python_dispatcher_active,
        )

        _, new_real_fn_spec = func_to_graphable(real_fn_with_grad)
        _, new_fake_fn_spec = func_to_graphable(fake_fn_with_grad)
        new_input_spec = None

        with torch._C._AutoDispatchBelowAutograd():
            fw_outputs = invoke_leaf_function(
                new_real_fn_spec, new_fake_fn_spec, new_input_spec, *flat_args
            )

        ctx.fw_inputs = (new_real_fn_spec, new_fake_fn_spec, input_spec, *flat_args)
        ctx.bw_real_fn = bw_real_fn
        ctx.bw_fake_fn = bw_fake_fn

        return fw_outputs

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(ctx, *grads):
        _, bw_real_fn_spec = func_to_graphable(ctx.bw_real_fn)
        _, bw_fake_fn_spec = func_to_graphable(ctx.bw_fake_fn)
        new_input_spec = None
        fw_grads = invoke_leaf_function(
            bw_real_fn_spec, bw_fake_fn_spec, new_input_spec, *grads
        )
        return None, None, None, *fw_grads


@invoke_leaf_function.py_autograd_impl
def invoke_leaf_function_autograd(real_fn_spec, fake_fn_spec, input_spec, *flat_args):
    return InvokeLeafFunctionAutogradOp.apply(
        real_fn_spec, fake_fn_spec, input_spec, *flat_args
    )


# TODO: allow user annotated mutation and aliasing info
@invoke_leaf_function.py_functionalize_impl
def invoke_leaf_function_functionalization(ctx, *all_args):
    unwrapped_args = ctx.unwrap_tensors(all_args)
    with ctx.redispatch_to_next():
        return ctx.wrap_tensors(invoke_leaf_function(*unwrapped_args))


@invoke_leaf_function.py_impl(ProxyTorchDispatchMode)
def invoke_leaf_function_proxy_mode(proxy_mode, *all_args):
    out = invoke_leaf_function(*all_args)

    proxies = pytree.tree_map(proxy_mode.tracer.unwrap_proxy, all_args)
    proxy = proxy_mode.tracer.create_proxy(
        "call_function", invoke_leaf_function, proxies, {}
    )

    return track_tensor_tree(out, proxy, constant=None, tracer=proxy_mode.tracer)


def _maybe_detach_tensor_args(args: tuple[Any, ...]):
    return pytree.tree_map(
        lambda t: t.detach().requires_grad_(t.requires_grad)
        if isinstance(t, torch.Tensor)
        else t,
        args,
    )


@register_fake(invoke_leaf_function)
def invoke_leaf_function_fake(real_fn_spec, fake_fn_spec, input_spec, *flat_args):
    fake_fn = unwrap_fn_spec(fake_fn_spec)
    flat_args = _maybe_detach_tensor_args(flat_args)
    out = invoke_with_flattened_inputs(fake_fn, input_spec, *flat_args)
    return _maybe_detach_tensor_args(out)


@invoke_leaf_function.py_impl(DispatchKey.CompositeExplicitAutograd)
def invoke_leaf_function_dense(real_fn_spec, fake_fn_spec, input_spec, *flat_args):
    real_fn = unwrap_fn_spec(real_fn_spec)
    flat_args = _maybe_detach_tensor_args(flat_args)
    out = invoke_with_flattened_inputs(real_fn, input_spec, *flat_args)
    return _maybe_detach_tensor_args(out)
