import contextlib
import functools
from collections.abc import Callable, Generator, Sequence
from typing import Any, NamedTuple

import torch
import torch.utils._pytree as pytree
from torch._C import DispatchKey, DispatchKeySet
from torch._higher_order_ops.utils import register_fake
from torch._ops import HigherOrderOperator
from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode, track_tensor_tree
from torch.nn.utils.stateless import _reparametrize_module

from .flat_apply import func_to_graphable


class LeafModuleState(NamedTuple):
    """
    A pytree representation of an nn.Module for use in invoke_leaf_function.

    In dynamo, nn.Module arguments to leaf functions are converted to this
    pytree format (index, parameters, buffers). This structure is then
    flattened to produce the actual inputs to invoke_leaf_function. At
    runtime, the original module is reconstructed via pytree unflatten and
    _reparametrize_module.

    Fields:
    - nn_module_index: Index to retrieve the original nn module at runtime.
      See register_user_object() and get_external_object_by_index() in
      graph_bytecode_inputs.py - objects are registered during tracing and
      retrieved by index during graph execution.
    - named_parameters: Named parameters of the module, used to reparametrize
    - named_buffers: Named buffers of the module, used to reparametrize
    """

    nn_module_index: int
    named_parameters: dict[str, torch.nn.Parameter]
    named_buffers: dict[str, torch.Tensor]


def unwrap_fn_spec(fn_spec: pytree.TreeSpec) -> Callable:
    return pytree.tree_unflatten((), fn_spec)


def _retrieve_module_by_index(nn_module_index: int) -> torch.nn.Module:
    from torch._dynamo.graph_bytecode_inputs import get_external_object_by_index

    mod = get_external_object_by_index(nn_module_index)
    if not isinstance(mod, torch.nn.Module):
        raise TypeError(
            f"Expected nn.Module at index {nn_module_index} for leaf function invocation, "
            f"but got {type(mod).__name__}. This may indicate the module index is invalid."
        )
    return mod


def _detach_tensors(tree: Any) -> Any:
    """Detach all tensors in the tree while preserving requires_grad."""
    return pytree.tree_map_only(
        torch.Tensor,
        lambda t: t.detach().requires_grad_(t.requires_grad),
        tree,
    )


@contextlib.contextmanager
def reconstruct_original_args(
    input_spec: pytree.TreeSpec | None, flat_args: tuple[Any, ...]
) -> Generator[tuple[list[Any] | tuple[Any, ...], dict[str, Any]], None, None]:
    """
    Reconstruct original (args, kwargs) from flattened arguments.

    Handles LeafModuleState by retrieving the original module and reparametrizing
    it with the parameters/buffers arguments.
    """
    if input_spec is None:
        yield flat_args, {}
        return

    args, kwargs = pytree.tree_unflatten(flat_args, input_spec)

    with contextlib.ExitStack() as stack:

        def process_module_state(state: LeafModuleState) -> torch.nn.Module:
            orig_module = _retrieve_module_by_index(state.nn_module_index)
            stack.enter_context(
                _reparametrize_module(
                    orig_module,
                    {**state.named_parameters, **state.named_buffers},
                )
            )
            return orig_module

        new_args, new_kwargs = pytree.tree_map_only(
            LeafModuleState,
            process_module_state,
            (args, kwargs),
            is_leaf=lambda x: isinstance(x, LeafModuleState),
        )
        yield new_args, new_kwargs


def autograd_grad_with_mixed_inputs(
    outputs: Sequence[Any],
    inputs: Sequence[Any],
    grad_outputs: Sequence[Any] | None = None,
    retain_graph: bool | None = None,
    create_graph: bool = False,
    allow_unused: bool = False,
) -> tuple[torch.Tensor | None, ...]:
    """
    Wrapper for torch.autograd.grad that handles mixed tensor/non-tensor inputs.

    Unlike torch.autograd.grad, this function accepts:
    - outputs: mix of tensors (with or without grad_fn) and non-tensors
    - inputs: mix of tensors (with or without requires_grad) and non-tensors

    Returns a tuple of gradients with None for non-tensor or non-requires_grad inputs.
    """
    # Filter outputs to only tensors with grad_fn
    filtered_outputs = []
    filtered_grad_outputs = []
    for i, out in enumerate(outputs):
        if isinstance(out, torch.Tensor) and out.grad_fn is not None:
            filtered_outputs.append(out)
            if grad_outputs is not None:
                filtered_grad_outputs.append(grad_outputs[i])

    # Filter inputs to only tensors with requires_grad, tracking original indices
    filtered_inputs = []
    input_indices = []
    for i, inp in enumerate(inputs):
        if isinstance(inp, torch.Tensor) and inp.requires_grad:
            filtered_inputs.append(inp)
            input_indices.append(i)

    # Early return if no valid outputs or inputs
    if not filtered_outputs or not filtered_inputs:
        return tuple(None for _ in inputs)

    # Compute gradients
    grads = torch.autograd.grad(
        outputs=filtered_outputs,
        inputs=filtered_inputs,
        grad_outputs=filtered_grad_outputs if grad_outputs is not None else None,
        retain_graph=retain_graph,
        create_graph=create_graph,
        allow_unused=allow_unused,
    )

    # Reconstruct full gradient tuple with Nones at proper positions
    result: list[torch.Tensor | None] = [None] * len(inputs)
    for filtered_idx, original_idx in enumerate(input_indices):
        result[original_idx] = grads[filtered_idx]

    return tuple(result)


def _make_forward(
    fn: Callable,
    requires_grad_indices: set[int],
    input_spec: pytree.TreeSpec | None,
    include_keys: DispatchKeySet,
    exclude_keys: DispatchKeySet,
) -> tuple[Callable, dict[str, Any]]:
    """
    Create a forward wrapper that captures inputs/outputs for backward.

    Returns (forward_fn, state_dict) where state_dict is shared with backward.
    """
    state: dict[str, Any] = {"inputs": None, "outputs": None}

    @functools.wraps(fn)
    def forward(*args):
        state["inputs"] = tuple(
            arg.requires_grad_(True) if idx in requires_grad_indices else arg
            for idx, arg in enumerate(args)
        )
        # For fake_impl, it can be called at tracing time and runtime (for output validation).
        # So dynamically remove PythonDispatcher if the forward is called at runtime
        effective_keys = include_keys
        if include_keys.has(DispatchKey.PythonDispatcher):
            current_include = torch._C._dispatch_tls_local_include_set()
            if not current_include.has(DispatchKey.PythonDispatcher):
                effective_keys = include_keys.remove(DispatchKey.PythonDispatcher)
        with torch._C._ForceDispatchKeyGuard(effective_keys, exclude_keys):
            with torch.enable_grad():
                with reconstruct_original_args(input_spec, args) as (
                    new_args,
                    new_kwargs,
                ):
                    state["outputs"] = fn(*new_args, **new_kwargs)
        return state["outputs"]

    return forward, state


def make_runtime_wrappers(
    fn: Callable,
    requires_grad_indices: set[int],
    input_spec: pytree.TreeSpec | None,
    include_keys: DispatchKeySet,
    exclude_keys: DispatchKeySet,
) -> tuple[Callable, Callable]:
    """
    Create forward/backward wrappers for runtime execution.
    """
    forward, state = _make_forward(
        fn, requires_grad_indices, input_spec, include_keys, exclude_keys
    )

    def backward(*grads):
        if state["inputs"] is None or state["outputs"] is None:
            raise RuntimeError(
                "invoke_leaf_function backward expects inputs/outputs to be set in forward."
            )
        return autograd_grad_with_mixed_inputs(
            outputs=state["outputs"],
            inputs=state["inputs"],
            grad_outputs=grads,
            allow_unused=True,
        )

    return forward, backward


def make_tracing_wrappers(
    fn: Callable,
    requires_grad_indices: set[int],
    input_spec: pytree.TreeSpec | None,
    include_keys: DispatchKeySet,
    exclude_keys: DispatchKeySet,
) -> tuple[Callable, Callable]:
    """
    Create forward/backward wrappers for tracing.

    Keeps PythonDispatcher in dispatch keys (already active in TLS).
    """
    forward, state = _make_forward(
        fn, requires_grad_indices, input_spec, include_keys, exclude_keys
    )

    def backward(*grads):
        if state["inputs"] is None or state["outputs"] is None:
            raise RuntimeError(
                "invoke_leaf_function backward expects inputs/outputs to be set in forward."
            )
        # Return fake gradients for tracing
        return tuple(
            torch.empty_like(state["inputs"][i]) if i in requires_grad_indices else None
            for i in range(len(state["inputs"]))
        )

    return forward, backward


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

        include_keys = torch._C._dispatch_tls_local_include_set()
        exclude_keys = torch._C._dispatch_tls_local_exclude_set()

        requires_grad_indices = {
            i
            for i, arg in enumerate(flat_args)
            if isinstance(arg, torch.Tensor) and arg.requires_grad
        }

        real_forward, real_backward = make_runtime_wrappers(
            real_fn, requires_grad_indices, input_spec, include_keys, exclude_keys
        )
        fake_forward, fake_backward = make_tracing_wrappers(
            fake_fn, requires_grad_indices, input_spec, include_keys, exclude_keys
        )

        _, new_real_fn_spec = func_to_graphable(real_forward)
        _, new_fake_fn_spec = func_to_graphable(fake_forward)

        with torch._C._AutoDispatchBelowAutograd():
            fw_outputs = invoke_leaf_function(
                new_real_fn_spec, new_fake_fn_spec, None, *flat_args
            )

        ctx.real_backward = real_backward
        ctx.fake_backward = fake_backward

        return fw_outputs

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(ctx, *grads):
        _, real_bw_spec = func_to_graphable(ctx.real_backward)
        _, fake_bw_spec = func_to_graphable(ctx.fake_backward)
        fw_grads = invoke_leaf_function(real_bw_spec, fake_bw_spec, None, *grads)
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


def _validate_outputs_match(
    fake_output: Any,
    real_output: Any,
) -> None:
    """
    Validate that fake_fn and real_fn outputs have matching pytree structure,
    shapes, and dtypes.

    Raises:
        RuntimeError: If outputs don't match with detailed error message.
    """
    # Check pytree structure matches
    fake_flat, fake_spec = pytree.tree_flatten(fake_output)
    real_flat, real_spec = pytree.tree_flatten(real_output)

    if fake_spec != real_spec:
        raise RuntimeError(
            f"Output structure mismatch in @leaf_function decorator.\n"
            f"fake_impl returned structure: {fake_spec}\n"
            f"real_impl returned structure: {real_spec}\n"
            f"The fake_impl must return outputs with the same pytree structure as real_impl."
        )

    if len(fake_flat) != len(real_flat):
        raise RuntimeError(
            f"Output count mismatch in @leaf_function decorator.\n"
            f"fake_impl returned {len(fake_flat)} values\n"
            f"real_impl returned {len(real_flat)} values"
        )

    # Check each tensor's shape and dtype
    for i, (fake_val, real_val) in enumerate(zip(fake_flat, real_flat)):
        fake_is_tensor = isinstance(fake_val, torch.Tensor)
        real_is_tensor = isinstance(real_val, torch.Tensor)

        if fake_is_tensor != real_is_tensor:
            raise RuntimeError(
                f"Output type mismatch at position {i} in @leaf_function decorator.\n"
                f"fake_impl returned: {type(fake_val).__name__}\n"
                f"real_impl returned: {type(real_val).__name__}"
            )

        if fake_is_tensor:
            if fake_val.shape != real_val.shape:
                raise RuntimeError(
                    f"Shape mismatch at output position {i} in @leaf_function decorator.\n"
                    f"fake_impl output shape: {list(fake_val.shape)}\n"
                    f"real_impl output shape: {list(real_val.shape)}\n"
                    f"The fake_impl must produce tensors with the same shapes as real_impl."
                )

            if fake_val.dtype != real_val.dtype:
                raise RuntimeError(
                    f"Dtype mismatch at output position {i} in @leaf_function decorator.\n"
                    f"fake_impl output dtype: {fake_val.dtype}\n"
                    f"real_impl output dtype: {real_val.dtype}\n"
                    f"The fake_impl must produce tensors with the same dtypes as real_impl."
                )


def _invoke_leaf_function_impl(fn_spec, input_spec, *flat_args):
    """
    Shared implementation for fake and dense dispatch.

    Tensors are detached before and after invoking the function to ensure
    the leaf function is treated as an atomic operation from autograd's
    perspective. This prevents internal operations from leaking into the
    autograd graph - gradients flow through InvokeLeafFunctionAutogradOp's
    custom backward instead.
    """
    fn = unwrap_fn_spec(fn_spec)
    flat_args = _detach_tensors(flat_args)
    with reconstruct_original_args(input_spec, flat_args) as (args, kwargs):
        out = fn(*args, **kwargs)
    return _detach_tensors(out)


def _check_no_input_mutation(
    flat_args: tuple[Any, ...],
    version_before: list[int],
) -> None:
    """
    Check that no input tensors were mutated in-place during leaf function execution.

    Raises RuntimeError if any input tensor's version counter changed.
    """
    for i, arg in enumerate(flat_args):
        if isinstance(arg, torch.Tensor):
            if arg._version != version_before[i]:
                raise RuntimeError(
                    f"In-place mutation detected on input tensor at position {i} in "
                    f"@leaf_function. In-place mutations on inputs are not tracked "
                    f"across the leaf function boundary and may cause incorrect results. "
                    f"Consider cloning the input before mutating it."
                )


@contextlib.contextmanager
def _allow_non_fake_inputs() -> Generator[None, None, None]:
    """
    Context manager to temporarily allow non-fake inputs in fake tensor mode.

    This is controlled by torch._dynamo.config.leaf_function_allow_non_fake_inputs.
    When enabled, real tensors in closures are auto-converted to fake tensors.
    When disabled (default), an error is raised if fake_impl uses non-fake tensors.
    """
    from torch._dynamo import config as dynamo_config
    from torch._subclasses.fake_tensor import fake_tensor_tls

    if not dynamo_config.leaf_function_allow_non_fake_inputs:
        yield
        return

    old_override = fake_tensor_tls.allow_non_fake_inputs_override
    try:
        fake_tensor_tls.allow_non_fake_inputs_override = True
        yield
    finally:
        fake_tensor_tls.allow_non_fake_inputs_override = old_override


@register_fake(invoke_leaf_function)
def invoke_leaf_function_fake(real_fn_spec, fake_fn_spec, input_spec, *flat_args):
    with _allow_non_fake_inputs():
        return _invoke_leaf_function_impl(fake_fn_spec, input_spec, *flat_args)


@invoke_leaf_function.py_impl(DispatchKey.CompositeExplicitAutograd)
def invoke_leaf_function_dense(real_fn_spec, fake_fn_spec, input_spec, *flat_args):
    from torch._dynamo import config as dynamo_config

    # Record version counters before calling function to detect input mutations
    version_before = [
        arg._version if isinstance(arg, torch.Tensor) else 0 for arg in flat_args
    ]

    real_output = _invoke_leaf_function_impl(real_fn_spec, input_spec, *flat_args)

    # Check for input mutations
    _check_no_input_mutation(flat_args, version_before)

    # Validate fake_impl outputs match real_impl outputs when enabled
    if dynamo_config.leaf_function_validate_outputs:
        fake_output = _invoke_leaf_function_impl(fake_fn_spec, input_spec, *flat_args)
        _validate_outputs_match(fake_output, real_output)

    return real_output
