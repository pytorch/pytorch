import contextlib
import functools
from collections.abc import Callable, Generator, Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple

import torch
import torch.utils._pytree as pytree
from torch._C import DispatchKey, DispatchKeySet
from torch._higher_order_ops.utils import register_fake
from torch._ops import HigherOrderOperator
from torch.autograd.graph import get_gradient_edge
from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode, track_tensor_tree
from torch.nn.utils.stateless import _reparametrize_module

from .flat_apply import func_to_graphable


# Callback for retrieving nn.Module instances by index for leaf_function.
# This is set by Dynamo's graph_bytecode_inputs module to avoid the HOP
# importing from Dynamo (which would be a layering violation).
_leaf_function_module_retriever: Callable[[int], Any] | None = None


def set_leaf_function_module_retriever(retriever: Callable[[int], Any]) -> None:
    """
    Set the callback for retrieving nn.Module instances by index.

    This is called by torch._dynamo.graph_bytecode_inputs to register
    its get_external_object_by_index function, allowing invoke_leaf_function
    to retrieve nn.Module instances without importing from Dynamo.
    """
    global _leaf_function_module_retriever
    _leaf_function_module_retriever = retriever


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
      Objects are registered by Dynamo during tracing and retrieved via
      the external object retriever callback (set by Dynamo at import time).
    - named_parameters: Named parameters of the module, used to reparametrize
    - named_buffers: Named buffers of the module, used to reparametrize
    """

    nn_module_index: int
    named_parameters: dict[str, torch.nn.Parameter]
    named_buffers: dict[str, torch.Tensor]


@dataclass
class GradientInfo:
    """
    Lightweight gradient metadata for invoke_leaf_function backward.

    Instead of storing full tensors in the forward state (which can be large),
    we store just the information needed for backward:
    - edge: GradientEdge for torch.autograd.grad (points to autograd graph node)
    - size: torch.Size for creating zero gradients when needed
    - dtype: torch.dtype for creating zero gradients
    - device: torch.device for creating zero gradients

    Why we store metadata (size, dtype, device) instead of using weakref:
    - The zeros fallback needs to create a tensor with the same shape/dtype/device
      as the original input. With explicit metadata, we always have this info.
    - Weakref would require a fallback path if the tensor is collected, adding
      complexity. And if we need the fallback anyway, might as well always use it.
    - The metadata overhead (a few bytes for size tuple + dtype + device) is
      negligible compared to actual tensor data we'd otherwise store.
    """

    edge: torch.autograd.graph.GradientEdge
    size: torch.Size
    dtype: torch.dtype
    device: torch.device


def unwrap_fn_spec(fn_spec: pytree.TreeSpec) -> Callable:
    return pytree.tree_unflatten((), fn_spec)


def _retrieve_module_by_index(nn_module_index: int) -> torch.nn.Module:
    if _leaf_function_module_retriever is None:
        raise RuntimeError(
            "Leaf function module retriever not set. This typically means "
            "torch._dynamo.graph_bytecode_inputs was not imported before "
            "invoke_leaf_function was called with nn.Module arguments."
        )

    mod = _leaf_function_module_retriever(nn_module_index)
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


def check_escaped_gradients(
    outputs: Any,
    inputs: Sequence[Any],
    requires_grad_indices: set[int],
) -> None:
    """
    Check if computation graph depends on tensors not passed as explicit inputs.

    When a leaf_function closes over a tensor with requires_grad=True, gradients
    won't flow back to it because backward only computes gradients for explicit
    inputs. This function walks the autograd graph from outputs and raises an
    error if it finds leaf tensors not in our input set.

    Controlled by torch._dynamo.config.leaf_function_check_escaped_gradients.

    Args:
        outputs: Function outputs (tensor or tuple of tensors)
        inputs: Function inputs
        requires_grad_indices: Indices of inputs that require gradients

    Raises:
        RuntimeError: If closure-captured tensors with requires_grad are detected
    """
    # Early exit if no inputs require grad
    if not requires_grad_indices:
        return

    # Lazy import to avoid overhead when check is disabled
    import torch._dynamo.config as config

    if not config.leaf_function_check_escaped_gradients:
        return

    # Collect autograd nodes for tracked inputs - these form the traversal boundary
    input_nodes: set[torch.autograd.graph.Node] = set()
    for i, inp in enumerate(inputs):
        if (
            isinstance(inp, torch.Tensor)
            and i in requires_grad_indices
            and inp.requires_grad
        ):
            edge = get_gradient_edge(inp)
            if edge.node is not None:
                input_nodes.add(edge.node)

    # Get output grad_fns as starting points for graph traversal
    flat_outputs = outputs if isinstance(outputs, tuple) else (outputs,)
    start_nodes: set[torch.autograd.graph.Node] = {
        out.grad_fn
        for out in flat_outputs
        if isinstance(out, torch.Tensor)
        and out.requires_grad
        and out.grad_fn is not None
    }
    if not start_nodes:
        return

    # Walk graph to find leaf nodes not in our input set
    escaped: set[torch.autograd.graph.Node] = set()
    visited: set[torch.autograd.graph.Node] = set()
    stack = list(start_nodes)

    while stack:
        node = stack.pop()
        if node in visited or node in input_nodes:
            continue
        visited.add(node)

        for next_node, _ in node.next_functions:
            if next_node is None or next_node in input_nodes:
                continue
            # Empty next_functions means this is a leaf node (AccumulateGrad)
            if not next_node.next_functions:
                escaped.add(next_node)
            else:
                stack.append(next_node)

    if escaped:
        # Build detailed info about escaped tensors
        tensor_info = []
        for node in escaped:
            if hasattr(node, "variable"):
                t = node.variable
                tensor_info.append(
                    f"  - Tensor(shape={list(t.shape)}, dtype={t.dtype})"
                )
        tensor_details = (
            "\n".join(tensor_info) if tensor_info else "  (tensor details unavailable)"
        )

        raise RuntimeError(
            f"@leaf_function detected {len(escaped)} tensor(s) with requires_grad=True "
            f"that are not passed as explicit inputs:\n{tensor_details}\n"
            f"Gradients will not flow back to closure-captured or global tensors. "
            f"Pass them as explicit arguments to the leaf function."
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


def autograd_grad_with_gradient_info(
    output_infos: Sequence[GradientInfo | None],
    input_infos: Sequence[GradientInfo | None],
    grad_outputs: Sequence[Any] | None = None,
    retain_graph: bool | None = None,
    create_graph: bool = False,
    allow_unused: bool = False,
) -> tuple[torch.Tensor | None, ...]:
    """
    Compute gradients using GradientInfo instead of full tensors.

    This is a lighter-weight alternative that avoids storing full tensors in
    the forward state. Instead, we store GradientInfo containing:
    - GradientEdge for torch.autograd.grad
    - Tensor metadata (size, dtype, device) for creating zeros when needed

    Args:
        output_infos: GradientInfo for each output (None if no grad_fn)
        input_infos: GradientInfo for each input (None if not requires_grad)
        grad_outputs: Gradients w.r.t. outputs

    Returns a tuple of gradients with None for inputs that don't require grad.
    """
    # Filter outputs to only those with valid GradientInfo
    filtered_output_edges = []
    filtered_grad_outputs = []
    for i, info in enumerate(output_infos):
        if info is not None and info.edge.node is not None:
            filtered_output_edges.append(info.edge)
            if grad_outputs is not None:
                filtered_grad_outputs.append(grad_outputs[i])

    # Filter inputs to only those with GradientInfo, tracking original indices
    filtered_input_edges = []
    filtered_input_infos = []
    input_indices = []
    for i, info in enumerate(input_infos):
        if info is not None:
            filtered_input_edges.append(info.edge)
            filtered_input_infos.append(info)
            input_indices.append(i)

    # Early return if no valid outputs or inputs
    if not filtered_output_edges or not filtered_input_edges:
        return tuple(None for _ in input_infos)

    # Compute gradients using GradientEdges
    grads = torch.autograd.grad(
        outputs=filtered_output_edges,
        inputs=filtered_input_edges,
        grad_outputs=filtered_grad_outputs if grad_outputs is not None else None,
        retain_graph=retain_graph,
        create_graph=create_graph,
        allow_unused=allow_unused,
    )

    # Reconstruct full gradient tuple with Nones at proper positions.
    #
    # NB: For unused inputs that require grad, we return zeros instead of None.
    # This is necessary because during AOT tracing, we use fake_impl to determine
    # the backward graph structure. fake_impl may not accurately reflect which
    # inputs are truly used for gradients in real_impl. For example, users often
    # write fake_impl by returning torch.empty_like(input) to get the right shape
    # without actually using the input in a differentiable computation. So we must
    # be permissive and assume all inputs with requires_grad=True could need
    # gradients. When multiple invoke_leaf_function calls share an input (e.g.,
    # a module's forward + hook both receiving the same tensor), the traced
    # backward graph generates explicit add operations to accumulate their
    # gradients. At runtime, if one leaf function doesn't actually use the input,
    # autograd.grad returns None for it, and the traced add(grad1, grad2) fails
    # because one operand is None. Returning zeros ensures the add always works.
    result: list[torch.Tensor | None] = [None] * len(input_infos)
    for filtered_idx, original_idx in enumerate(input_indices):
        grad = grads[filtered_idx]
        if grad is None:
            info = filtered_input_infos[filtered_idx]
            grad = torch.zeros(info.size, dtype=info.dtype, device=info.device)
        result[original_idx] = grad

    return tuple(result)


def _make_forward(
    fn: Callable,
    requires_grad_indices: set[int],
    input_spec: pytree.TreeSpec | None,
    include_keys: DispatchKeySet,
    exclude_keys: DispatchKeySet,
) -> tuple[Callable, dict[str, Any]]:
    """
    Create a forward wrapper that captures gradient info for backward.

    Returns (forward_fn, state_dict) where state_dict is shared with backward.
    Instead of storing full tensors, we store lightweight GradientInfo to
    minimize memory usage.
    """
    state: dict[str, Any] = {"inputs": None, "outputs": None}

    @functools.wraps(fn)
    def forward(*args):
        # NB: We mutate requires_grad in-place here, but this is safe because
        # the args are already detached tensors. In dense/fake dispatch,
        # _detach_tensors() is called before invoking this function, so we're
        # not touching the caller's original tensors' requires_grad state.
        #
        # We need this because we capture which inputs have requires_grad=True
        # at tracing time (in requires_grad_indices). The tricky part is that
        # at runtime with aot_eager, the forward graph runs inside
        # autograd.Function.forward(), which is a no-grad context. This means
        # intermediate tensors can lose their requires_grad status by the time
        # they reach invoke_leaf_function, even though they had it during tracing.
        #
        # So we're essentially restoring the requires_grad state to match what
        # we saw at tracing time. Otherwise the leaf function's internal
        # autograd wouldn't work correctly.
        inputs = tuple(
            arg.requires_grad_(True) if idx in requires_grad_indices else arg
            for idx, arg in enumerate(args)
        )
        # NB: fake_impl can be called in two different contexts:
        # - At tracing time: with fake tensors, where PythonDispatcher is active
        # - At runtime: for output validation (when leaf_function_validate_outputs=True),
        #   where PythonDispatcher is NOT active
        #
        # We capture the dispatch keys when the wrapper is created (at tracing time),
        # but the wrapper may be invoked later at runtime for validation. Since
        # PythonDispatcher should only be active during tracing, we detect runtime
        # by checking if it's currently in the TLS include set—if not, we remove it
        # from the captured keys.
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
                    outputs = fn(*new_args, **new_kwargs)

                # Check for escaped gradients (must be inside enable_grad for get_gradient_edge)
                check_escaped_gradients(outputs, inputs, requires_grad_indices)

                # Capture lightweight gradient info instead of storing full tensors.
                # This significantly reduces memory usage for large tensors.
                state["inputs"] = tuple(
                    GradientInfo(
                        edge=get_gradient_edge(inp),
                        size=inp.size(),
                        dtype=inp.dtype,
                        device=inp.device,
                    )
                    if isinstance(inp, torch.Tensor) and inp.requires_grad
                    else None
                    for inp in inputs
                )

                # Capture output GradientInfo for tensors with grad_fn
                flat_outputs = (
                    outputs if isinstance(outputs, tuple) else (outputs,)
                )
                state["outputs"] = tuple(
                    GradientInfo(
                        edge=get_gradient_edge(out),
                        size=out.size(),
                        dtype=out.dtype,
                        device=out.device,
                    )
                    if isinstance(out, torch.Tensor)
                    and out.requires_grad
                    and out.grad_fn is not None
                    else None
                    for out in flat_outputs
                )

        return outputs

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
        return autograd_grad_with_gradient_info(
            output_infos=state["outputs"],
            input_infos=state["inputs"],
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
    """
    forward, state = _make_forward(
        fn, requires_grad_indices, input_spec, include_keys, exclude_keys
    )

    def backward(*grads):
        if state["inputs"] is None:
            raise RuntimeError(
                "invoke_leaf_function backward expects inputs to be set in forward."
            )
        # Return fake gradients for tracing (shapes match inputs)
        return tuple(
            torch.empty(info.size, dtype=info.dtype, device=info.device)
            if info is not None
            else None
            for info in state["inputs"]
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
                    f"In-place mutation detected on input tensor at position {i} "
                    f"(in the pytree-flattened inputs with nn.Module states expanded) in "
                    f"@leaf_function. In-place mutations on inputs are not tracked "
                    f"across the leaf function boundary and may cause incorrect results. "
                    f"Consider cloning the input before mutating it."
                )


@register_fake(invoke_leaf_function)
def invoke_leaf_function_fake(real_fn_spec, fake_fn_spec, input_spec, *flat_args):
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
