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


_leaf_function_module_retriever: Callable[[int], Any] | None = None


def set_leaf_function_module_retriever(retriever: Callable[[int], Any]) -> None:
    global _leaf_function_module_retriever
    _leaf_function_module_retriever = retriever


class LeafModuleState(NamedTuple):
    """
    In dynamo, nn.Module arguments to leaf functions are converted to this
    pytree format (index, parameters, buffers). This structure is then
    flattened to produce the actual inputs to invoke_leaf_function.

    At runtime, the original module is reconstructed from it.
    """

    nn_module_index: int
    named_parameters: dict[str, torch.nn.Parameter]
    named_buffers: dict[str, torch.Tensor]


@dataclass
class GradientInfo:
    """
    We need the gradient edge to trigger the autograd engine backward.

    We need the tensor metadata (size, stride, dtype, device) to create zeros when
    gradient is None at runtime but required by the backward graph (because the graph
    is traced with fake implementation).
    """

    edge: torch.autograd.graph.GradientEdge
    size: torch.Size
    stride: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device


def unwrap_fn_spec(fn_spec: pytree.TreeSpec) -> Callable:
    return pytree.tree_unflatten((), fn_spec)


def _retrieve_module_by_index(nn_module_index: int) -> torch.nn.Module:
    if _leaf_function_module_retriever is None:
        raise RuntimeError("Leaf function module retriever not set.")

    mod = _leaf_function_module_retriever(nn_module_index)
    if not isinstance(mod, torch.nn.Module):
        raise TypeError(
            f"Expected nn.Module at index {nn_module_index} for leaf function invocation, "
            f"but got {type(mod).__name__}."
        )
    return mod


def check_escaped_gradients(
    outputs: Any,
    inputs: Sequence[Any],
    requires_grad_indices: set[int],
) -> None:
    """
    Check if autograd graph depends on tensors that not passed as explicit inputs.

    Controlled by torch._dynamo.config.leaf_function_check_escaped_gradients.
    """
    if not requires_grad_indices:
        return

    import torch._dynamo.config as config

    if not config.leaf_function_check_escaped_gradients:
        return

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
            if not next_node.next_functions:
                escaped.add(next_node)
            else:
                stack.append(next_node)

    if escaped:
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
def unflatten_args_with_modules(
    flat_args: tuple[Any, ...], input_spec: pytree.TreeSpec
) -> Generator[tuple[list[Any] | tuple[Any, ...], dict[str, Any]], None, None]:
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


def flatten_args_with_modules(
    args_kwargs: tuple[Any, ...],
) -> list[Any]:
    def expand_module(x: Any) -> Any:
        if isinstance(x, torch.nn.Module):
            return LeafModuleState(
                nn_module_index=-1,
                named_parameters=dict(x.named_parameters()),
                named_buffers=dict(x.named_buffers()),
            )
        return x

    expanded = pytree.tree_map(expand_module, args_kwargs)
    return pytree.tree_leaves(expanded)


def autograd_grad_with_gradient_info(
    output_infos: Sequence[GradientInfo | None],
    input_infos: Sequence[GradientInfo | None],
    grad_outputs: Sequence[Any] | None = None,
    retain_graph: bool | None = None,
    create_graph: bool = False,
    allow_unused: bool = False,
) -> tuple[torch.Tensor | None, ...]:
    """
    Compute gradients using GradientInfo, it additionally handles the case
    where input and output infos are None.

    Args:
        output_infos: GradientInfo for each output (None if no grad_fn),
        input_infos: GradientInfo for each input (None if not requires_grad)
        grad_outputs: Gradients w.r.t. outputs

    Returns a tuple of gradients with None for inputs that don't require grad.
    """
    filtered_output_edges = []
    filtered_grad_outputs = []
    for i, info in enumerate(output_infos):
        if info is not None and info.edge.node is not None:
            filtered_output_edges.append(info.edge)
            if grad_outputs is not None:
                filtered_grad_outputs.append(grad_outputs[i])

    filtered_input_edges = []
    filtered_input_infos = []
    input_indices = []
    for i, info in enumerate(input_infos):
        if info is not None:
            filtered_input_edges.append(info.edge)
            filtered_input_infos.append(info)
            input_indices.append(i)

    if not filtered_output_edges or not filtered_input_edges:
        return tuple(None for _ in input_infos)

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
            grad = torch.empty_strided(
                info.size, info.stride, dtype=info.dtype, device=info.device
            ).zero_()
        result[original_idx] = grad

    return tuple(result)


def _make_forward(
    fn: Callable,
    include_keys: DispatchKeySet,
    exclude_keys: DispatchKeySet,
) -> tuple[Callable, dict[str, Any]]:
    state: dict[str, Any] = {"inputs": None, "outputs": None}

    @functools.wraps(fn)
    def forward(*args, **kwargs):
        effective_keys = include_keys
        if include_keys.has(DispatchKey.PythonDispatcher):
            effective_keys = include_keys.remove(DispatchKey.PythonDispatcher)
        with torch._C._ForceDispatchKeyGuard(effective_keys, exclude_keys):
            with torch.enable_grad():
                outputs = fn(*args, **kwargs)

                flat_inputs = flatten_args_with_modules((args, kwargs))
                requires_grad_indices = {
                    i
                    for i, inp in enumerate(flat_inputs)
                    if isinstance(inp, torch.Tensor) and inp.requires_grad
                }
                check_escaped_gradients(outputs, flat_inputs, requires_grad_indices)

                state["inputs"] = tuple(
                    GradientInfo(
                        edge=get_gradient_edge(inp),
                        size=inp.size(),
                        stride=inp.stride(),
                        dtype=inp.dtype,
                        device=inp.device,
                    )
                    if isinstance(inp, torch.Tensor) and inp.requires_grad
                    else None
                    for inp in flat_inputs
                )

                if outputs is None:
                    state["outputs"] = ()
                else:
                    state["outputs"] = tuple(
                        GradientInfo(
                            edge=get_gradient_edge(out),
                            size=out.size(),
                            stride=out.stride(),
                            dtype=out.dtype,
                            device=out.device,
                        )
                        if isinstance(out, torch.Tensor)
                        and out.requires_grad
                        and out.grad_fn is not None
                        else None
                        for out in outputs
                    )

        return pytree.tree_map_only(
            torch.Tensor,
            lambda t: t.detach().requires_grad_(t.requires_grad),
            outputs,
        )

    return forward, state


class InvokeLeafFunction(HigherOrderOperator):
    def __init__(self):
        super().__init__("invoke_leaf_function")

    def __call__(
        self,
        real_fn_spec,
        fake_fn_spec,
        input_spec,
        *flat_args,
        requires_grad_indices=(),
    ):
        """
        real_fn_spec: pytree.TreeSpec for the real function that's wrapped in dynamo
        fake_fn_spec: pytree.TreeSpec for the fake function that's wrapped in dynamo
        input_spec: pytree.TreeSpec for unflattening flat_args back to (args, kwargs)
        requires_grad_indices: tuple of indices for inputs that require grad
        """
        return super().__call__(  # type: ignore[attr-defined]
            real_fn_spec,
            fake_fn_spec,
            input_spec,
            *flat_args,
            requires_grad_indices=requires_grad_indices,
        )

    # pyrefly: ignore [bad-override]
    def gen_schema(self, real_fn_spec, fake_fn_spec, input_spec, *flat_args, requires_grad_indices=()):
        from torch._higher_order_ops.schema import HopSchemaGenerator
        from torch._higher_order_ops.utils import _maybe_fake_prop_ignore_unbacked

        fake_fn = unwrap_fn_spec(fake_fn_spec)
        with unflatten_args_with_modules(flat_args, input_spec) as (args, kwargs):
            fake_outputs = _maybe_fake_prop_ignore_unbacked(
                lambda *a: fake_fn(*a, **kwargs), tuple(args)
            )

        gen = HopSchemaGenerator(self)
        gen.add_arg("real_fn_spec", real_fn_spec)
        gen.add_arg("fake_fn_spec", fake_fn_spec)
        gen.add_arg("input_spec", input_spec)
        for i, arg in enumerate(flat_args):
            gen.add_arg(f"arg{i}", arg)

        if isinstance(fake_outputs, tuple):
            for out in fake_outputs:
                gen.add_output(out)
        else:
            if fake_outputs is not None:
                raise AssertionError(
                    f"Expected fake_outputs to be a tuple or None, got {type(fake_outputs)}"
                )
            gen.add_output(fake_outputs)

        return gen.gen_schema()


invoke_leaf_function = InvokeLeafFunction()


# NOTE: [Autograd support for invoke_leaf_function]
#
# The oveerall idea is that when the real forward executes, we are going to build an autograd graph
# and save it.  When the real backward executes, we are going to invoke the autograd graph.
# We need to build these "real_forward" and "real_backward" functions from the "real_fn".
#
# Inputs:
# real_fn_spec/fake_fn_spec are pytree.TreeSpecs that contain real_fn and fake_fn.
# These functions were created in dynamo by wrapping the user's original leaf function and fake function:
#   - They accept *flat_args (flattened LeafModuleState objects + other args)
#   - They unflatten flat_args and convert LeafModuleState back to nn.Modules
#   - They call the user's original leaf function with the reconstructed args
#   - They return the user function's outputs
#
# We wrap real_fn (via _make_forward) to handle autograd properly:
# 1. Detach inputs and outputs to isolate the leaf function's autograd graph from the
#    outer graph (gradients flow through our backward, not internal ops)
# 2. Real function is invoked at backend dispatch keys (i.e. CompositeExplicitAutograd)
#    where autograd is disabled. We re-enable grad and restore dispatch keys for the autograd engine to work.
# 3. Store GradientInfo (gradient edges + tensor metadata) instead of full tensors for backward,
#    which is sufficient for autograd.grad and avoids keeping tensors alive.
#
# We automatically generate backward functions:
# - real_backward uses the stored input/output GradientInfo and invokes the autograd engine
# - fake_backward creates empty gradients for inputs that requires grad


class InvokeLeafFunctionAutogradOp(torch.autograd.Function):
    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(ctx, real_fn_spec, fake_fn_spec, input_spec, *flat_args):
        real_fn = unwrap_fn_spec(real_fn_spec)

        include_keys = torch._C._dispatch_tls_local_include_set()
        exclude_keys = torch._C._dispatch_tls_local_exclude_set()

        requires_grad_indices = tuple(
            i
            for i, arg in enumerate(flat_args)
            if isinstance(arg, torch.Tensor) and arg.requires_grad
        )

        real_forward, real_state = _make_forward(real_fn, include_keys, exclude_keys)

        def real_backward(*grads):
            if real_state["inputs"] is None or real_state["outputs"] is None:
                raise RuntimeError(
                    "invoke_leaf_function backward expects inputs/outputs to be set in forward."
                )
            return autograd_grad_with_gradient_info(
                output_infos=real_state["outputs"],
                input_infos=real_state["inputs"],
                grad_outputs=grads,
                allow_unused=True,
            )

        input_infos_for_fake = tuple(
            GradientInfo(
                edge=None,  # type: ignore[arg-type]
                size=arg.size(),
                stride=arg.stride(),
                dtype=arg.dtype,
                device=arg.device,
            )
            if isinstance(arg, torch.Tensor) and arg.requires_grad
            else None
            for arg in flat_args
        )

        def fake_backward(*grads):
            return tuple(
                torch.empty_strided(
                    info.size, info.stride, dtype=info.dtype, device=info.device
                )
                if info is not None
                else None
                for info in input_infos_for_fake
            )

        _, new_real_fn_spec = func_to_graphable(real_forward)

        with torch._C._AutoDispatchBelowAutograd():
            fw_outputs = invoke_leaf_function(
                new_real_fn_spec,
                fake_fn_spec,
                input_spec,
                *flat_args,
                requires_grad_indices=requires_grad_indices,
            )

        ctx.real_backward = real_backward
        ctx.fake_backward = fake_backward

        return fw_outputs

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(ctx, *grads):
        _, real_bw_spec = func_to_graphable(ctx.real_backward)
        _, fake_bw_spec = func_to_graphable(ctx.fake_backward)
        _, bw_input_spec = pytree.tree_flatten((grads, {}))
        fw_grads = invoke_leaf_function(
            real_bw_spec, fake_bw_spec, bw_input_spec, *grads
        )
        return None, None, None, *fw_grads


@invoke_leaf_function.py_autograd_impl
def invoke_leaf_function_autograd(
    real_fn_spec, fake_fn_spec, input_spec, *flat_args, requires_grad_indices=()
):
    return InvokeLeafFunctionAutogradOp.apply(
        real_fn_spec, fake_fn_spec, input_spec, *flat_args
    )


# TODO: allow user annotated mutation and aliasing info
@invoke_leaf_function.py_functionalize_impl
def invoke_leaf_function_functionalization(ctx, *all_args, **kwargs):
    from torch._higher_order_ops.effects import handle_effects

    return handle_effects(
        ctx.mode._allow_token_discovery,
        ctx.mode._tokens,
        invoke_leaf_function,
        all_args,
        kwargs,
    )


@invoke_leaf_function.py_impl(ProxyTorchDispatchMode)
def invoke_leaf_function_proxy_mode(proxy_mode, *all_args, **kwargs):
    out = invoke_leaf_function(*all_args, **kwargs)
    proxies = pytree.tree_map(proxy_mode.tracer.unwrap_proxy, all_args)
    proxy = proxy_mode.tracer.create_proxy(
        "call_function", invoke_leaf_function, proxies, kwargs
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


def _check_no_input_mutation(
    flat_args: tuple[Any, ...],
    version_before: list[int],
) -> None:
    for i, arg in enumerate(flat_args):
        if isinstance(arg, torch.Tensor):
            if arg._version != version_before[i]:
                raise RuntimeError(
                    f"In-place mutation detected on input tensor at position {i} "
                    f"(in the pytree-flattened inputs with nn.Module states expanded) in "
                    f"@leaf_function. In-place mutations on inputs are not supported yet."
                    f"Consider cloning the input before mutating it."
                )


@register_fake(invoke_leaf_function)
def invoke_leaf_function_fake(
    real_fn_spec, fake_fn_spec, input_spec, *flat_args, requires_grad_indices=()
):
    fake_fn = unwrap_fn_spec(fake_fn_spec)
    with unflatten_args_with_modules(flat_args, input_spec) as (args, kwargs):
        return fake_fn(*args, **kwargs)


@invoke_leaf_function.py_impl(DispatchKey.CompositeExplicitAutograd)
def invoke_leaf_function_dense(
    real_fn_spec, fake_fn_spec, input_spec, *flat_args, requires_grad_indices=()
):
    from torch._dynamo import config as dynamo_config

    version_before = [
        arg._version if isinstance(arg, torch.Tensor) else 0 for arg in flat_args
    ]

    flat_args = tuple(
        arg.detach() if isinstance(arg, torch.Tensor) else arg for arg in flat_args
    )
    requires_grad_indices_set = set(requires_grad_indices)
    flat_args = tuple(
        arg.requires_grad_(True) if idx in requires_grad_indices_set else arg
        for idx, arg in enumerate(flat_args)
    )

    real_fn = unwrap_fn_spec(real_fn_spec)
    with unflatten_args_with_modules(flat_args, input_spec) as (args, kwargs):
        real_output = real_fn(*args, **kwargs)

        _check_no_input_mutation(flat_args, version_before)

        if dynamo_config.leaf_function_validate_outputs:
            fake_fn = unwrap_fn_spec(fake_fn_spec)
            fake_output = fake_fn(*args, **kwargs)
            _validate_outputs_match(fake_output, real_output)

    return real_output
