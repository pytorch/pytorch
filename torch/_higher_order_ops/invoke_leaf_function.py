import contextlib
import functools
from collections.abc import Callable, Generator, Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple

import torch
import torch.utils._pytree as pytree
from torch._C import DispatchKey, DispatchKeySet
from torch._higher_order_ops.utils import register_fake
from torch._library.opaque_object import OpaqueBase, register_opaque_type
from torch._ops import HigherOrderOperator
from torch.autograd.graph import get_gradient_edge
from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode, track_tensor_tree
from torch.nn.utils.stateless import _reparametrize_module


_leaf_function_module_retriever: Callable[[int], Any] | None = None

# Separate storage for the make_fx / _invoke_leaf_function_python path.
# We can't use Dynamo's register_user_object because it requires a Source
# (for bytecode generation) and adds entries to index_to_bytecode_constructor.
# In the make_fx path we have neither a Source nor bytecode generation, so we
# maintain our own dict keyed by negative indices to avoid collisions with
# Dynamo's non-negative indices.
_makefx_module_storage: dict[int, torch.nn.Module] = {}
_makefx_next_index = 0


def store_makefx_modules(modules: list[torch.nn.Module]) -> tuple[int, ...]:
    """Store modules for the make_fx path and return their assigned indices.

    Uses negative indices to avoid collisions with Dynamo's register_user_object
    which uses non-negative indices.
    """
    global _makefx_next_index
    indices = []
    for mod in modules:
        _makefx_next_index -= 1
        _makefx_module_storage[_makefx_next_index] = mod
        indices.append(_makefx_next_index)
    return tuple(indices)


def reset_makefx_module_storage() -> None:
    global _makefx_next_index
    _makefx_next_index = 0
    _makefx_module_storage.clear()


class _LeafCallable(OpaqueBase):
    def __init__(self, fn: Callable) -> None:
        self._fn = fn

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._fn(*args, **kwargs)


register_opaque_type(_LeafCallable, typ="reference")


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


def convert_modules_to_states(values: Any, module_to_index: dict[int, int]) -> Any:
    """Replace nn.Module instances in a pytree with LeafModuleState objects.

    Args:
        values: A pytree of values that may contain nn.Module instances.
        module_to_index: Mapping from id(module) to its integer index.
    """

    def module_to_state(val: Any) -> Any:
        if isinstance(val, torch.nn.Module):
            return LeafModuleState(
                nn_module_index=module_to_index[id(val)],
                named_parameters=dict(val.named_parameters()),
                named_buffers=dict(val.named_buffers()),
            )
        return val

    return pytree.tree_map(module_to_state, values)


def _resolve_mutated_flat_indices(
    fn: Callable,
    mutates_args: frozenset[str],
    num_flat_args: int,
    input_spec: pytree.TreeSpec,
) -> str:
    """Resolve mutates_args expressions to a comma-separated string of flat-arg indices.

    Each expression in mutates_args (e.g. "x", "model.running_mean") is evaluated
    against sentinel values to determine which flat-arg positions are mutated.

    Example: for ``def fn(x, model)`` where model is an nn.Module with parameters
    ``weight`` and ``bias``, the flat args are ``[x, nn_module_index, weight, bias]``.
    Given ``mutates_args={"model.weight"}``, this assigns sentinels ``[0, 1, 2, 3]``
    to the flat args, evaluates ``model.weight`` to ``2``, and returns ``"2"``.
    """
    import inspect

    class _AttrDict:
        pass

    def _set_nested_attr(obj: _AttrDict, fqn: str, value: Any) -> None:
        parts = fqn.split(".")
        for part in parts[:-1]:
            if not hasattr(obj, part):
                setattr(obj, part, _AttrDict())
            obj = getattr(obj, part)
        setattr(obj, parts[-1], value)

    def _lms_to_attr_dict(val: Any) -> Any:
        if isinstance(val, LeafModuleState):
            target = _AttrDict()
            for fqn, sentinel in val.named_parameters.items():
                _set_nested_attr(target, fqn, sentinel)
            for fqn, sentinel in val.named_buffers.items():
                _set_nested_attr(target, fqn, sentinel)
            return target
        return val

    sig = inspect.signature(fn)
    sentinels = list(range(num_flat_args))
    args_struct, kwargs_struct = pytree.tree_unflatten(sentinels, input_spec)
    args_eval, kwargs_eval = pytree.tree_map(
        _lms_to_attr_dict,
        (args_struct, kwargs_struct),
        is_leaf=lambda x: isinstance(x, LeafModuleState),
    )
    namespace = dict(sig.bind(*args_eval, **kwargs_eval).arguments)

    indices: list[int] = []
    for expr in mutates_args:
        # Empty __builtins__ prevents access to builtins like __import__, open, exec.
        result = eval(expr, {"__builtins__": {}}, namespace)  # noqa: S307
        leaves = pytree.tree_leaves(result)
        for sentinel in leaves:
            if not isinstance(sentinel, int):
                raise ValueError(
                    f"mutates_args expression '{expr}' resolved to a non-leaf value "
                    f"of type {type(sentinel).__name__}. Expressions must resolve to "
                    f"individual tensor positions, e.g. 'model.weight' not 'model'."
                )
            indices.append(sentinel)
    indices.sort()
    return ",".join(str(i) for i in indices)


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


def _retrieve_module_by_index(nn_module_index: int) -> torch.nn.Module:
    # Check make_fx storage first (used by _invoke_leaf_function_python).
    # Fall back to the Dynamo retriever (used by the compiled path).
    if nn_module_index in _makefx_module_storage:
        if nn_module_index >= 0:
            raise RuntimeError(
                f"Expected negative nn_module_index for non-strict trace over leaf_function, but got {nn_module_index}."
            )
        return _makefx_module_storage[nn_module_index]

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


def make_leaf_function_wrappers(
    real_fn: Callable[..., Any],
    fake_fn: Callable[..., Any],
    captured_out_spec: list[pytree.TreeSpec | None],
) -> tuple[Callable[..., tuple[Any, ...]], Callable[..., tuple[Any, ...]]]:
    """Wrap real_fn and fake_fn to flatten outputs and capture the output TreeSpec.

    Both wrappers share the same captured output spec: the first call (typically
    fake_fn during tracing) records it, and subsequent calls verify consistency.
    The caller passes in a single-element list and reads captured_out_spec[0]
    after the wrappers have been called.

    Used by both the Dynamo path (_call_leaf_function in torch.py) and the
    make_fx path (_invoke_leaf_function_python in decorators.py).
    """

    def _wrap(fn: Callable[..., Any]) -> Callable[..., tuple[Any, ...]]:
        if len(captured_out_spec) != 1:
            raise RuntimeError(
                f"captured_out_spec must be a single-element list, got length {len(captured_out_spec)}"
            )

        def wrapper(*args: Any, **kwargs: Any) -> tuple[Any, ...]:
            out = fn(*args, **kwargs)

            flat_out, out_spec = pytree.tree_flatten(out)
            if captured_out_spec[0] is None:
                captured_out_spec[0] = out_spec
            elif captured_out_spec[0] != out_spec:
                raise AssertionError(
                    f"leaf_function output structure mismatch: "
                    f"expected {captured_out_spec[0]}, got {out_spec}. "
                    f"This can happen if the real function and fake function return "
                    f"different pytree structures (e.g., dict vs tuple, different number "
                    f"of elements). Ensure both functions return the same structure."
                )
            return tuple(flat_out)

        return wrapper

    return _wrap(real_fn), _wrap(fake_fn)


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
            effective_keys = effective_keys.remove(DispatchKey.PythonDispatcher)
        if effective_keys.has(DispatchKey.Python):
            effective_keys = effective_keys.remove(DispatchKey.Python)
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
        super().__init__("invoke_leaf_function", supports_training_input_mutation=True)

    def __call__(
        self,
        real_fn_callable,
        fake_fn_callable,
        input_spec,
        mutated_arg_indices,
        *flat_args,
        requires_grad_indices=(),
    ):
        """
        real_fn_callable: _LeafCallable wrapping the real function
        fake_fn_callable: _LeafCallable wrapping the fake function
        input_spec: pytree.TreeSpec for unflattening flat_args back to (args, kwargs)
        mutated_arg_indices: comma-separated string of flat-arg indices that are
            declared as mutated (e.g. "1,2"), or "" for no mutations. Encoded as a
            string so it is a pytree leaf for the HOP schema infrastructure.
        requires_grad_indices: tuple of indices for inputs that require grad
        """
        return super().__call__(  # type: ignore[attr-defined]
            real_fn_callable,
            fake_fn_callable,
            input_spec,
            mutated_arg_indices,
            *flat_args,
            requires_grad_indices=requires_grad_indices,
        )

    # pyrefly: ignore [bad-override]
    def gen_schema(
        self,
        real_fn_callable,
        fake_fn_callable,
        input_spec,
        mutated_arg_indices,
        *flat_args,
        requires_grad_indices=(),
    ):
        from torch._higher_order_ops.schema import HopSchemaGenerator
        from torch._higher_order_ops.utils import _maybe_fake_prop_ignore_unbacked
        from torch.fx.experimental.proxy_tensor import disable_proxy_modes_tracing

        mutated_set = _parse_mutated_arg_indices(mutated_arg_indices)

        with disable_proxy_modes_tracing():
            if mutated_set:
                schema_flat_args = tuple(
                    arg.detach().clone()
                    if isinstance(arg, torch.Tensor) and i in mutated_set
                    else arg
                    for i, arg in enumerate(flat_args)
                )
            else:
                schema_flat_args = flat_args

            def run_fake(*unfunc_flat_args):
                with unflatten_args_with_modules(unfunc_flat_args, input_spec) as (
                    args,
                    kwargs,
                ):
                    return fake_fn_callable(*args, **kwargs)

            fake_outputs = _maybe_fake_prop_ignore_unbacked(run_fake, schema_flat_args)

        gen = HopSchemaGenerator(self)
        gen.add_arg("real_fn_callable", real_fn_callable)
        gen.add_arg("fake_fn_callable", fake_fn_callable)
        gen.add_arg("input_spec", input_spec)
        gen.add_arg("mutated_arg_indices", mutated_arg_indices)
        for i, arg in enumerate(flat_args):
            gen.add_arg(f"arg{i}", arg, is_mutated=i in mutated_set)

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
# The overall idea is that when the real forward executes, we are going to build an autograd graph
# and save it.  When the real backward executes, we are going to invoke the autograd graph.
# We need to build these "real_forward" and "real_backward" functions from the "real_fn".
#
# Inputs:
# real_fn_callable/fake_fn_callable are _LeafCallable objects that wrap real_fn and fake_fn.
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
    def forward(
        ctx,
        real_fn_callable,
        fake_fn_callable,
        input_spec,
        mutated_arg_indices,
        *flat_args,
    ):
        include_keys = torch._C._dispatch_tls_local_include_set()
        exclude_keys = torch._C._dispatch_tls_local_exclude_set()

        requires_grad_indices = tuple(
            i
            for i, arg in enumerate(flat_args)
            if isinstance(arg, torch.Tensor) and arg.requires_grad
        )

        real_forward, real_state = _make_forward(
            real_fn_callable, include_keys, exclude_keys
        )

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

        new_real_fn_callable = _LeafCallable(real_forward)

        with torch._C._AutoDispatchBelowAutograd():
            fw_outputs = invoke_leaf_function(
                new_real_fn_callable,
                fake_fn_callable,
                input_spec,
                mutated_arg_indices,
                *flat_args,
                requires_grad_indices=requires_grad_indices,
            )

        ctx.real_backward = real_backward
        ctx.fake_backward = fake_backward

        return fw_outputs

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(ctx, *grads):
        real_bw_callable = _LeafCallable(ctx.real_backward)
        fake_bw_callable = _LeafCallable(ctx.fake_backward)
        _, bw_input_spec = pytree.tree_flatten((grads, {}))
        fw_grads = invoke_leaf_function(
            real_bw_callable, fake_bw_callable, bw_input_spec, "", *grads
        )
        return None, None, None, None, *fw_grads


@invoke_leaf_function.py_autograd_impl
def invoke_leaf_function_autograd(
    real_fn_callable,
    fake_fn_callable,
    input_spec,
    mutated_arg_indices,
    *flat_args,
    requires_grad_indices=(),
):
    return InvokeLeafFunctionAutogradOp.apply(
        real_fn_callable, fake_fn_callable, input_spec, mutated_arg_indices, *flat_args
    )


# TODO: aliasing is not allowed
@invoke_leaf_function.py_functionalize_impl
def invoke_leaf_function_functionalization(ctx, *all_args, **kwargs):
    from torch._higher_order_ops.auto_functionalize import (
        can_auto_functionalize,
        do_auto_functionalize_v2,
    )
    from torch._higher_order_ops.utils import HopInstance

    unwrapped_args = ctx.unwrap_tensors(all_args)
    hop_instance = HopInstance.create(invoke_leaf_function, *unwrapped_args, **kwargs)
    if can_auto_functionalize(hop_instance):
        return do_auto_functionalize_v2(ctx.mode, hop_instance, all_args, kwargs)

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


def _parse_mutated_arg_indices(s: str) -> set[int]:
    return {int(x) for x in s.split(",") if x}


def _check_no_input_mutation(
    flat_args: tuple[Any, ...],
    version_before: list[int],
    mutated_arg_indices: str = "",
) -> None:
    mutated_set = _parse_mutated_arg_indices(mutated_arg_indices)
    for i, arg in enumerate(flat_args):
        if isinstance(arg, torch.Tensor) and arg._version != version_before[i]:
            if i not in mutated_set:
                raise RuntimeError(
                    f"Undeclared in-place mutation on input tensor at position {i}. "
                    f"Declare it in @leaf_function(mutates_args=...) or avoid mutating inputs."
                )


@register_fake(invoke_leaf_function)
def invoke_leaf_function_fake(
    real_fn_callable,
    fake_fn_callable,
    input_spec,
    mutated_arg_indices,
    *flat_args,
    requires_grad_indices=(),
):
    with unflatten_args_with_modules(flat_args, input_spec) as (args, kwargs):
        return fake_fn_callable(*args, **kwargs)


@invoke_leaf_function.py_impl(DispatchKey.CompositeExplicitAutograd)
def invoke_leaf_function_dense(
    real_fn_callable,
    fake_fn_callable,
    input_spec,
    mutated_arg_indices,
    *flat_args,
    requires_grad_indices=(),
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

    with unflatten_args_with_modules(flat_args, input_spec) as (args, kwargs):
        real_output = real_fn_callable(*args, **kwargs)

        _check_no_input_mutation(flat_args, version_before, mutated_arg_indices)

        if dynamo_config.leaf_function_validate_outputs:
            fake_output = fake_fn_callable(*args, **kwargs)
            _validate_outputs_match(fake_output, real_output)

    return real_output
