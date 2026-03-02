"""
This module provides decorators and utilities for controlling TorchDynamo's behavior during compilation.
"""

import functools
import inspect
import weakref
from collections.abc import Callable
from dataclasses import dataclass
from types import TracebackType
from typing import Any, overload, TYPE_CHECKING, TypeVar
from typing_extensions import ParamSpec

import torch
import torch.utils._pytree as pytree
from torch.compiler import is_compiling
from torch.utils._contextlib import _DecoratorContextManager
from torch.utils._python_dispatch import is_traceable_wrapper_subclass

from .._utils_internal import justknobs_check
from . import trace_rules, variables
from .comptime import comptime
from .eval_frame import (
    _set_stance,
    DisableContext,
    DynamoStance,
    innermost_fn,
    RunOnlyContext,
    skip_code,
)
from .external_utils import (
    get_nonrecursive_disable_wrapper,
    wrap_dunder_call_ctx_manager,
)
from .utils import _get_error_on_graph_break, _set_error_on_graph_break, is_function


justknobs_check._dynamo_marked_constant = True  # type: ignore[attr-defined]

if TYPE_CHECKING:
    from types import FunctionType

    from torch._C._dynamo.eval_frame import (  # noqa: F401
        reset_code,
        set_eval_frame,
        set_guard_complete_hook,
        set_guard_error_hook,
        unsupported,
    )

    from .variables import VariableTracker
else:
    for name in dir(torch._C._dynamo.eval_frame):
        if name.startswith("__"):
            continue
        globals()[name] = getattr(torch._C._dynamo.eval_frame, name)


_P = ParamSpec("_P")
_R = TypeVar("_R")
FuncType = Callable[..., Any]
F = TypeVar("F", bound=FuncType)


def run(fn: Callable[_P, _R] | None = None) -> Any:
    """Don't do any dynamic compiles, just use prior optimizations"""
    if fn is not None:
        fn = innermost_fn(fn)
        assert callable(fn)
        return RunOnlyContext()(fn)
    return RunOnlyContext()


def disable(fn=None, recursive=True, *, reason=None, wrapping=True):  # type: ignore[no-untyped-def]
    """
    Decorator to disable TorchDynamo

    If recursive=True, Dynamo is completely skipped on the decorated function
    frame as well as the recursively invoked functions.

    If recursive=False, Dynamo skips frames associated with the function code,
    but still process recursively invoked frames.

    If reason is provided, it will be printed when Dynamo attempts to trace the disabled function.
    """
    if recursive:
        if fn is not None:
            fn = innermost_fn(fn)
            assert callable(fn)
            return DisableContext(msg=reason, wrapping=wrapping)(fn)
        return DisableContext(msg=reason, wrapping=wrapping)
    else:

        def wrap(fn: Callable[_P, _R]) -> Callable[_P, _R]:
            fn = innermost_fn(fn)
            assert callable(fn)

            nonrecursive_disable_wrapper = get_nonrecursive_disable_wrapper(fn)
            nonrecursive_disable_wrapper._torchdynamo_disable = True  # type: ignore[attr-defined]
            nonrecursive_disable_wrapper._torchdynamo_disable_msg = reason  # type: ignore[attr-defined]
            nonrecursive_disable_wrapper._torchdynamo_orig_callable = fn  # type: ignore[attr-defined]
            nonrecursive_disable_wrapper._torchdynamo_wrapper_id = id(  # type: ignore[attr-defined]
                nonrecursive_disable_wrapper
            )
            nonrecursive_disable_wrapper._torchdynamo_disable_recursive = False  # type: ignore[attr-defined]
            return nonrecursive_disable_wrapper

        if fn is None:
            return wrap
        return wrap(fn)


_nonrecursive_disable_wrapper_code = disable(lambda: None, recursive=False).__code__  # type: ignore[attr-defined]
skip_code(_nonrecursive_disable_wrapper_code)


def skip(fn: Callable[_P, _R] | None = None) -> Callable[..., Any]:
    """
    Skip frames associated with the function code, but still process recursively
    invoked frames
    """
    if fn is None:
        return skip
    fn = innermost_fn(fn)
    assert callable(fn)
    skip_code(fn.__code__)
    fn._torchdynamo_disable = True  # type: ignore[attr-defined]
    return fn


class set_stance(_DecoratorContextManager):
    """
    Decorator, context manager, function to set the current stance of the compiler.

    Stances documented in corresponding function in torch/compiler/__init__.py
    """

    _dynamo_forbidden = True

    def __init__(
        self,
        stance: str = "default",
        *,
        skip_guard_eval_unsafe: bool = False,
        force_backend: str | Callable[..., Any] | None = None,
    ) -> None:
        if force_backend is not None and stance != "default":
            raise RuntimeError("non-default stance cannot have force_backend set")

        self.stance = DynamoStance(stance, skip_guard_eval_unsafe, force_backend)
        self.prev = _set_stance(self.stance)

    def __call__(self, fn: F) -> F:
        _set_stance(self.prev)
        wrapper = super().__call__(fn)
        # forbid wrapper in graph
        wrapper._dynamo_forbidden = True  # type: ignore[attr-defined]
        return wrapper

    def __enter__(self) -> None:
        _set_stance(self.stance)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        _set_stance(self.prev)

    def clone(self) -> "set_stance":
        return self.__class__(self.stance.stance, force_backend=self.stance.backend)


def assume_constant_result(fn):  # type: ignore[no-untyped-def]
    fn._dynamo_marked_constant = True  # type: ignore[attr-defined]
    return fn


def allow_in_graph(fn):  # type: ignore[no-untyped-def]
    """
    Tells the compiler frontend (Dynamo) to skip symbolic introspection of the function
    and instead directly write it to the graph when encountered.

    See :func:`torch.compiler.allow_in_graph`'s docstring for the full documentation

    WARNING: this API can be a footgun, please read the documentation carefully.
    """
    if isinstance(fn, (list, tuple)):
        return [allow_in_graph(x) for x in fn]
    assert callable(fn), "allow_in_graph expects a callable"
    if trace_rules.lookup_callable(fn) != variables.TorchInGraphFunctionVariable:
        fn_id = id(fn)
        trace_rules._disallowed_callable_ids.remove(fn_id)
        trace_rules._allowed_callable_ids.add(fn_id)

        # Avoid id reuse which creates subtle bugs.
        def deregister() -> None:
            trace_rules._allowed_callable_ids.remove(fn_id)

        weakref.finalize(fn, deregister)
    return fn


# pyrefly: ignore [implicit-any]
def _check_mutually_exclusive_decorators(fn: Callable, decorator_name: str) -> None:
    mutually_exclusive = {
        "leaf_function": trace_rules.is_leaf_function,
        "nonstrict_trace": trace_rules.is_nonstrict_trace_callable,
    }

    for other_name, check_fn in mutually_exclusive.items():
        if other_name != decorator_name and check_fn(fn):
            first, second = sorted([decorator_name, other_name])
            raise ValueError(
                f"Function {fn} cannot be both marked as @{first} and "
                f"@{second}. Please use only one decorator."
            )


def nonstrict_trace(traceable_fn: Callable[_P, _R]) -> Callable[_P, _R]:
    """
    Decorator to mark a function as nonstrict-traceable for dynamo.

    A nonstrict-traced function appears as an opaque call in the dynamo graph.
    Dynamo does not trace into the function body (hence the "nonstrict"), but
    aot_autograd will trace into it.

    This is similar to ``allow_in_graph`` but with enhanced support for:
    - User-defined classes as inputs (must be registered with pytree)
    - ``nn.Module`` as input arguments (parameters and buffers are tracked for autograd)
    - Global/captured tensors treated as constants (assumed not updated during execution)

    Note:
        - With ``backend="eager"``, the original Python function runs directly.
          With ``backend="aot_eager"``, the graph traced by aot_autograd runs.
          With ``backend="inductor"``, the traced graph is compiled with inductor.

        - Training is supported: you can call ``.backward()`` on outputs and gradients
          will flow through the nonstrict-traced function.

    Dangerous patterns (may cause silent incorrectness):
        - Side effects between nonstric_traced fn and compiled region: The function should
          not depend on variables mutated by other code inside the compiled function, and code
          after the call should not depend on mutations made by it.

        - Implicit inputs (closures/globals): Tensors captured from enclosing scopes
          are treated as constants. Gradients will NOT flow back to them. Pass tensors
          as explicit arguments if gradients are needed.

    Restrictions:
        - Both inputs and outputs must use pytree-compatible types. User-defined classes
          must be registered via :func:`torch.utils._pytree.register_pytree_node`,
          :func:`torch.utils._pytree.register_dataclass`, or
          :func:`torch.utils._pytree.register_constant`. Tensors, Python primitives (int, float, bool, str),
          symbolic types (SymInt, SymFloat, SymBool), and built-in containers (list,
          tuple, dict) are already handled by default.
        - Primitive values and container structure are specialized per call site:
          each call site expects the same primitives and structure on every execution.

    Example::

        >>> import torch
        >>> @torch._dynamo.nonstrict_trace
        ... def traced_forward(model, x):
        ...     # It's OK to have dynamo graph break within nonstrict_trace region
        ...     torch._dynamo.graph_break()
        ...     return model(x) + x
        ...
        >>> class MyModule(torch.nn.Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.inner = torch.nn.Linear(10, 10)
        ...
        ...     def forward(self, x):
        ...         return traced_forward(self.inner, x)
        ...
        >>> # Compile and run
        >>> model = MyModule()
        >>> opt_model = torch.compile(model, backend="aot_eager", fullgraph=True)
        >>> out = opt_model(torch.randn(10, 10))
        >>> out.sum().backward()  # Gradients flow through traced_forward

    """
    assert callable(traceable_fn), "nonstrict_trace expects a callable"

    _check_mutually_exclusive_decorators(traceable_fn, "nonstrict_trace")

    @functools.wraps(traceable_fn)
    def wrapped(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        return traceable_fn(*args, **kwargs)

    wrapped_id = id(wrapped)

    # This line allows us to reuse much of the `allow_in_graph` impl.
    trace_rules._allowed_callable_ids.add(wrapped_id)

    # This line allows us to diverge the impl from `allow_in_graph`.
    trace_rules._nonstrict_trace_callable_ids.add(wrapped_id)

    # Avoid id reuse which creates subtle bugs.
    def deregister() -> None:
        trace_rules._allowed_callable_ids.remove(wrapped_id)
        trace_rules._nonstrict_trace_callable_ids.remove(wrapped_id)

    weakref.finalize(wrapped, deregister)

    return wrapped


def _invoke_leaf_function_python(
    real_impl: Callable[..., Any],
    fake_impl: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    mutates_args: frozenset[str] | None = None,
) -> Any:
    """Call invoke_leaf_function HOP directly from Python.

    This enables @leaf_function to work with make_fx
    without relying on Dynamo to intercept the call.
    """
    from torch._higher_order_ops.invoke_leaf_function import (
        _LeafCallable,
        convert_modules_to_states,
        invoke_leaf_function,
        make_leaf_function_wrappers,
        store_makefx_modules,
    )

    captured_modules: list[torch.nn.Module] = []
    seen_module_ids: dict[int, int] = {}  # id(module) -> position in captured_modules
    for val in pytree.tree_flatten(
        (args, kwargs), is_leaf=lambda x: isinstance(x, torch.nn.Module)
    )[0]:
        if isinstance(val, torch.nn.Module) and id(val) not in seen_module_ids:
            seen_module_ids[id(val)] = len(captured_modules)
            captured_modules.append(val)

    global_indices = store_makefx_modules(captured_modules)
    module_to_index = {
        mod_id: global_indices[pos] for mod_id, pos in seen_module_ids.items()
    }

    processed = convert_modules_to_states((args, kwargs), module_to_index)
    flat_args, input_spec = pytree.tree_flatten(processed)

    # Single-element mutable list so the wrappers can write back the output
    # TreeSpec. Read captured_out_spec[0] after the wrappers have been called.
    captured_out_spec: list[pytree.TreeSpec | None] = [None]
    wrapped_real, wrapped_fake = make_leaf_function_wrappers(
        real_impl, fake_impl, captured_out_spec
    )

    real_fn_callable = _LeafCallable(wrapped_real)
    fake_fn_callable = _LeafCallable(wrapped_fake)

    mutated_flat_indices = ""
    if mutates_args:
        from torch._higher_order_ops.invoke_leaf_function import (
            _resolve_mutated_flat_indices,
        )

        mutated_flat_indices = _resolve_mutated_flat_indices(
            real_impl, mutates_args, len(flat_args), input_spec
        )

    flat_out = invoke_leaf_function(
        real_fn_callable, fake_fn_callable, input_spec, mutated_flat_indices, *flat_args
    )

    assert captured_out_spec[0] is not None
    return pytree.tree_unflatten(flat_out, captured_out_spec[0])


@overload
def leaf_function(fn: Callable[_P, _R]) -> Callable[_P, _R]: ...


@overload
def leaf_function(
    *, mutates_args: set[str]
) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]: ...


def leaf_function(
    fn: Callable[_P, _R] | None = None, *, mutates_args: set[str] | None = None
) -> Callable[_P, _R] | Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    """
    Decorator to mark a function as a leaf function for :func:`torch.compile`.

    A leaf function appears as an opaque operation in the compiled graph. During
    compilation, Dynamo and AOT autograd do not trace into it. At runtime, the
    original eager Python code is executed directly.

    Quick Start:
        Suppose we want to log per-sample statistics during the forward pass::

            def compute_and_log_stats(x):
                stats = x.mean(dim=1)
                logger.info(f"Per-sample means: {stats}")
                return (stats,)

        With ``torch.compile(..., fullgraph=True)``, this fails because
        ``logger.info`` causes a graph break. To fix it, follow the steps below:

        1. Decorate your function with ``@leaf_function``
        2. Define a shape-inference function using ``@your_fn.register_fake``

        Concrete implementation::

            >>> import logging
            >>> import torch
            >>> from torch._dynamo.decorators import leaf_function
            >>>
            >>> logging.basicConfig(level=logging.INFO)
            >>> logger = logging.getLogger(__name__)
            >>>
            >>> @leaf_function
            ... def compute_and_log_stats(x):
            ...     stats = x.mean(dim=1)  # Shape: (x.shape[0],)
            ...     logger.info(f"Per-sample means: {stats}")
            ...     return (stats,)
            ...
            >>> @compute_and_log_stats.register_fake
            ... def compute_and_log_stats_fake(x):
            ...     # Match the output shape: (x.shape[0],)
            ...     return (x.new_empty(x.shape[0]),)
            ...
            >>> def fn(x):
            ...     return compute_and_log_stats(x)
            ...
            >>> x = torch.randn(3, 4)
            >>> compiled_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
            >>> out = compiled_fn(x)[0]

    When to Use:
        Use ``leaf_function`` when you need eager execution semantics that tracing
        cannot preserve:

        - **Non-traceable code**: The function consists of code that Dynamo or
          AOT Autograd cannot trace.
        - **Runtime side effects**: Operations that must happen exactly once per call
          at runtime (e.g., logging, external library calls).

        If your function has a static computation graph and no runtime side effects,
        prefer :func:`allow_in_graph` or :func:`nonstrict_trace` instead. They allow AOT autograd
        to trace through and potentially optimize the code.

    Usage:
        **Inputs and Outputs**:
        - Both inputs and outputs must use pytree-compatible types.
        Tensors, Python primitives (int, float, bool, str), and built-in containers
        (list, tuple, dict) are supported by default.

        Note: We recommend leaf_functions only accept and return tensors. Though primitive
        types (int, float, bool, str) are supported in inputs and outputs, they may cause
        surprising behavior: primitive inputs are guarded and may trigger recompilation,
        while primitive output values are captured from the fake implementation at compile time and
        remain fixed for all subsequent executions. Even though the real function runs
        at runtime, its primitive return values are silently replaced with the compile-time
        values from the fake implementation.

        Example::

            # BAD: primitive output varies at runtime
            counter = 0


            @leaf_function
            def count_calls(x):
                global counter
                counter += 1
                return (x, counter)


            @count_calls.register_fake
            def count_calls_fake(x):
                return (x, 999)  # placeholder value


            # At runtime: real function runs (counter increments to 1, 2, 3, ...)
            # But returned count is always 999 (from fake implementation at compile time)

        - User-defined classes must be registered via :func:`torch.utils._pytree.register_pytree_node`
        or :func:`torch.utils._pytree.register_dataclass`.

        - :class:`torch.nn.Module` can also be passed as input; its parameters and buffers
        are tracked for autograd. The module must exist outside the compile region.
        Running the module multiple times must not modify its parameters, buffers, or attributes.

        **register_fake (required)**:
        Since the function body is not traced, you must
        provide a shape-inference function via ``@fn.register_fake``. It runs at compile
        time with FakeTensor inputs (tensors with no data, only metadata) and must
        satisfy the following requirements:

        - Must have the same input and output signature (e.g., same pytree structure, same tensor metadata
          such as shapes, dtypes, device, strides) as the real function
        - Must be runnable with FakeTensor inputs
        - Must only use its explicit arguments (no closures over tensors or modules)

        Note: The input and output signature must be determinable at compile time.
        If your function's output structure depends on runtime values, adjust the leaf function
        boundary until outputs become predictable.

        To validate that your fake implementation matches the real function's outputs, set
        ``torch._dynamo.config.leaf_function_validate_outputs = True``.

    Limitations:
        Currently, inductor backend and :func:`torch.export.export` are not yet supported.

    Training / Autograd:
        Training is supported automatically if your leaf function is differentiable
        in eager mode (e.g., it's implemented with PyTorch ops, :class:`torch.autograd.Function`,
        or differentiable custom ops).

        **Restriction**: Calling ``.backward()`` *inside* the leaf function is not
        supported.

        **Escaped gradients check**: If the leaf function closes over a tensor with
        ``requires_grad=True``, gradients will not flow back to it (only explicit inputs
        receive gradients). To detect such cases, set
        ``torch._dynamo.config.leaf_function_check_escaped_gradients = True``.
        When enabled, a ``RuntimeError`` is raised with details about the escaped tensors.

        Internally, inputs and outputs are detached from the outer autograd graph at the
        leaf function boundary. The leaf function builds its own local autograd
        graph. In backward, gradients propagate through this local graph to the
        leaf function's inputs. If a :class:`torch.nn.Module` is passed in as input, the gradients
        will also flow back to its parameters and buffers.

    How leaf_function Works:
        Understanding this helps avoid pitfalls:

        1. **Compilation**: Dynamo and AOT autograd do not trace into the leaf function.
           Only your fake implementation runs during compilation to determine output signatures,
           shapes, and dtypes. The real function runs at runtime as eager Python.

        2. **Isolation**: Mutations to shared state (globals, closures) may not be
           visible across the boundary between leaf functions and compiled regions. Pass data
           explicitly as function arguments and return results as outputs.

    Dangerous Patterns:
        These patterns may cause silent incorrectness or errors:

        - **Side effects between leaf functions and compiled regions**: Mutations to
          Python state inside a leaf function may not be visible to the compiled
          region, and vice versa. The compiled graph may reorder or eliminate
          operations in ways that break assumptions about side effect ordering.

          Bad::

            # Compiled region
            self.counter += 1
            y = my_leaf_fn(self, x)  # Don't depend on self.counter inside leaf_fn

            y = my_leaf_fn(self, x)
            result = self.state  # Don't depend on state mutated by leaf_fn

          Logging/printing inside the leaf function is fine since it doesn't affect
          correctness.

        - **In-place mutations on inputs**: Undeclared in-place mutations on input
          tensors are detected and will raise an error. Either declare mutations via
          ``mutates_args`` or clone inputs before mutating.

          Bad::

            @leaf_function
            def my_leaf_fn(x):
                x.add_(1)  # Will raise: "Undeclared in-place mutation"
                return (x,)

          Good::

            @leaf_function(mutates_args={"buf"})
            def my_leaf_fn(x, buf):
                buf.add_(1)  # OK: declared in mutates_args
                return (x + buf,)

          For pytree args (lists, tuples, dicts of tensors), use the parameter name
          to declare mutation on all contained tensors::

            @leaf_function(mutates_args={"buffers"})
            def my_leaf_fn(x, buffers):
                for buf in buffers:
                    buf.add_(1)  # OK: 'buffers' declared in mutates_args
                return (x + sum(buffers),)

          Or use bracket notation for fine-grained control::

            @leaf_function(mutates_args={"buffers[0]"})
            def my_leaf_fn(x, buffers):
                buffers[0].add_(1)  # OK: 'buffers[0]' declared
                return (x + buffers[1],)  # buffers[1] is not cloned

        - **Closures in fake implementation**: Tensors or modules captured from enclosing scopes
          in the fake implementation will cause compilation errors. The real function can
          close over them if they don't require gradient, but the fake implementation must
          only use its arguments.

          Bad::

            weight = torch.randn(3, 3)


            @leaf_function
            def my_leaf_fn(x):
                return (x @ weight,)


            @my_leaf_fn.register_fake
            def my_leaf_fn_fake(x):
                return (x @ weight,)  # Error: weight used in fake implementation

          Good::

            weight = torch.randn(3, 3)


            @leaf_function
            def my_leaf_fn(x):
                return (x @ weight,)  # OK: real function can use closure


            @my_leaf_fn.register_fake
            def my_leaf_fn_fake(x):
                return (x @ torch.empty_like(x),)  # OK: uses only args

    Example::
        Wrapping a linear function with logging as a leaf_function. Gradients
        flow back because the inner operations are differentiable::

            >>> import torch
            >>> import torch.nn.functional as F
            >>> from torch._dynamo.decorators import leaf_function
            >>>
            >>> @leaf_function
            ... def custom_forward(linear, x):
            ...     # Logging at runtime
            ...     print(f"Input: shape={x.shape}, mean={x.mean().item():.4f}")
            ...     out = F.linear(x, linear.weight, linear.bias)
            ...     print(f"Output: shape={out.shape}, norm={out.norm().item():.4f}")
            ...     return (out,)
            ...
            >>> @custom_forward.register_fake
            ... def custom_forward_fake(linear, x):
            ...     # Return tensor with correct shape: (batch, out_features)
            ...     return (x.new_empty(x.shape[0], linear.weight.shape[0]),)
            ...
            >>> class MyModel(torch.nn.Module):
            ...     def __init__(self):
            ...         super().__init__()
            ...         self.linear = torch.nn.Linear(10, 20)
            ...     def forward(self, x):
            ...         return custom_forward(self.linear, x)
            ...
            >>> model = MyModel()
            >>> compiled = torch.compile(model, backend="aot_eager", fullgraph=True)
            >>> x = torch.randn(32, 10, requires_grad=True)
            >>> out = compiled(x)[0]
            >>> out.sum().backward()  # Gradients flow to model.linear.weight/bias and x

    Args:
        fn: The function being decorated.
        mutates_args: Set of Python expressions (as strings) identifying arguments
            that the function mutates in-place. Each string is evaluated against
            the function's parameters. Examples: ``'buf'`` for a plain tensor,
            ``'model.running_mean'`` for an nn.Module buffer,
            ``'buffers'`` to mark all tensors in a list, ``'buffers[0]'`` for a
            specific element, ``'state["key"]'`` for a dict entry.
    """
    if fn is None:
        return functools.partial(leaf_function, mutates_args=mutates_args)

    from . import trace_rules

    _check_mutually_exclusive_decorators(fn, "leaf_function")

    if mutates_args:
        import inspect
        import re

        params = set(inspect.signature(fn).parameters)
        for expr in mutates_args:
            root = re.split(r"[.\[]", expr, maxsplit=1)[0]
            if root not in params:
                raise ValueError(
                    f"mutates_args expression '{expr}' refers to parameter '{root}', "
                    f"which is not a parameter of '{fn.__name__}'. "
                    f"Available parameters: {', '.join(params)}"
                )

    @functools.wraps(fn)
    def inner(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        if inner._torchdynamo_leaf_fake_fn is None:  # type: ignore[attr-defined]
            raise ValueError(
                f"leaf_function '{getattr(fn, '__name__', fn)}' "
                "requires a fake implementation. Please provide one using the @<func>.register_fake "
                "decorator. See the leaf_function docstring for details."
            )

        return _invoke_leaf_function_python(
            fn,  # pyrefly: ignore [bad-argument-type]
            # pyrefly: ignore [bad-argument-type]
            inner._torchdynamo_leaf_fake_fn,
            args,
            kwargs,
            mutates_args=inner._torchdynamo_leaf_mutates_args,  # pyrefly: ignore [missing-attribute]
        )  # type: ignore[attr-defined]

    inner._torchdynamo_leaf_real_fn = fn  # type: ignore[attr-defined]
    inner._torchdynamo_leaf_fake_fn = None  # type: ignore[attr-defined]
    inner._torchdynamo_leaf_mutates_args = (  # pyrefly: ignore [missing-attribute]
        frozenset(mutates_args) if mutates_args else frozenset()
    )  # type: ignore[attr-defined]

    # Follow nonstrict_trace implementation
    wrapped_id = id(inner)
    trace_rules._allowed_callable_ids.add(wrapped_id)
    trace_rules._leaf_function_ids.add(wrapped_id)

    def deregister() -> None:
        trace_rules._allowed_callable_ids.remove(wrapped_id)
        trace_rules._leaf_function_ids.remove(wrapped_id)

    weakref.finalize(inner, deregister)

    def register_fake_setter(fake_fn: Callable[..., Any]) -> Callable[..., Any]:
        inner._torchdynamo_leaf_fake_fn = fake_fn  # type: ignore[attr-defined]
        return inner

    inner.register_fake = register_fake_setter  # type: ignore[attr-defined]

    return inner


def _disallow_in_graph_helper(throw_if_not_allowed: bool) -> Callable[..., Any]:
    def inner(fn: Any) -> Any:
        if isinstance(fn, (list, tuple)):
            return [disallow_in_graph(x) for x in fn]
        assert callable(fn), "disallow_in_graph expects a callable"
        if (
            throw_if_not_allowed
            and trace_rules.lookup_callable(fn)
            != variables.TorchInGraphFunctionVariable
            and trace_rules.lookup(fn) != variables.TorchInGraphFunctionVariable
        ):
            raise RuntimeError(
                "disallow_in_graph is expected to be used on an already allowed callable (like torch.* ops). "
                "Allowed callables means callables that TorchDynamo puts as-is in the extracted graph."
            )
        trace_rules._allowed_callable_ids.remove(id(fn))
        trace_rules._nonstrict_trace_callable_ids.remove(id(fn))
        trace_rules._disallowed_callable_ids.add(id(fn))
        return fn

    return inner


def disallow_in_graph(fn: Callable[..., Any]) -> Any:
    """
    Customize which functions TorchDynamo will exclude in the generated
    graph and force a graph break on.
    ::

        torch._dynamo.disallow_in_graph(torch.sub)


        @torch._dynamo.optimize(...)
        def fn(a):
            x = torch.add(x, 1)
            x = torch.sub(x, 1)
            x = torch.add(x, 1)
            return x


        fn(...)

    Will break the graph on `torch.sub`, and give two graphs each with a
    single `torch.add()` op.
    """
    return _disallow_in_graph_helper(throw_if_not_allowed=True)(fn)


@_disallow_in_graph_helper(throw_if_not_allowed=False)
def graph_break(msg: str = "") -> None:
    """Force a graph break"""


# NOTE: primarily used for internal debugging purposes!
@_disallow_in_graph_helper(throw_if_not_allowed=False)
def skip_frame(msg: str = "") -> None:
    """Force a skipped frame"""


@_disallow_in_graph_helper(throw_if_not_allowed=False)
def step_unsupported(msg: str = "") -> None:
    """Force a step unsupported graph break, which results in compiling
    the traced FX graph so far, then skipping the rest of the frame.
    In order to get expected behavior, there should be at least 2 ops
    and a part of the code not contained in any try/with blocks."""


def forbid_in_graph(fn: Any) -> Any:
    """
    Customize which functions TorchDynamo will assert are not present while tracing.

    If you want a graph break on this function instead, use disallow_in_graph.
    TODO(voz): We now have allow_in_graph, disallow_in_graph, forbid_in_graph - some more robust
    documentation would not be amiss.
    """
    if isinstance(fn, (list, tuple)):
        return [forbid_in_graph(x) for x in fn]
    assert callable(fn), "forbid_in_graph applies only to callables"
    # pyrefly: ignore [missing-attribute]
    fn._dynamo_forbidden = True
    return fn


def substitute_in_graph(
    original_fn: Callable[_P, _R],
    *,
    can_constant_fold_through: bool = False,
    skip_signature_check: bool = False,
    # type that is embedded in the Python interpreter
    is_embedded_type: bool = False,  # internal use only
) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    """
    Register a polyfill handler for a function, usually a C function from the C extension, to be
    used in place of the original function when inlining the original function in the graph.

    .. note::

        The polyfill handler is only used when inlining the original function. It is not used when
        the original function is called directly. In the eager mode, the decorated function calls
        the performant C function rather than the polyfill handler.

    The polyfill handler is a function that will be called in place of the original function when
    inlining the original function. The polyfill handler should have the same signature and the same
    behavior as the original function.

    Args:
        original_fn (callable): The original function, usually a C function, to register a polyfill
            handler for.
        can_constant_fold_through (bool, optional): Whether the polyfill handler can be constant
            folded through. That is, if the polyfill handler is a pure function and its arguments
            are constant, the result of the polyfill handler can be constant folded during the
            compilation. Defaults to ``False``.
        skip_signature_check (bool, optional): Whether to skip the signature check between the
            original function and the polyfill handler. Defaults to ``False``.

    Returns:
        A decorator that registers the polyfill handler for the original function.

    Example::

        >>> # xdoctest: +SKIP("conflict with the tests: duplicate polyfill handlers")
        >>> import operator
        >>> operator.indexOf([1, 2, 3, 4, 5], 3)
        2
        >>> torch.compile(operator.indexOf, fullgraph=True)([1, 2, 3, 4, 5], 3)
        Traceback (most recent call last):
        ...
        torch._dynamo.exc.Unsupported: ...

        >>> @torch.compiler.substitute_in_graph(operator.indexOf)
        ... def indexOf(a, b, /):
        ...     for i, item in enumerate(a):
        ...         if item is b or item == b:
        ...             return i
        ...     raise ValueError("sequence.index(x): x not in sequence")
        >>>
        >>> torch.compile(operator.indexOf, fullgraph=True)([1, 2, 3, 4, 5], 3)
        2
    """
    if not is_function(original_fn) and not (
        is_embedded_type and inspect.isclass(original_fn)
    ):
        raise TypeError(
            f"substitute_in_graph expects a function but got {type(original_fn)!r}"
        )
    if is_embedded_type:
        if not inspect.isclass(original_fn):
            raise TypeError(
                f"substitute_in_graph expects a class but got {type(original_fn)!r}"
            )

        from .variables.builder import ITERTOOLS_POLYFILLED_TYPE_IDS, ITERTOOLS_TYPE_IDS

        if id(original_fn) in ITERTOOLS_TYPE_IDS:
            ITERTOOLS_POLYFILLED_TYPE_IDS.add(id(original_fn))

    def wrapper(traceable_fn: Callable[_P, _R]) -> Callable[_P, _R]:
        if not is_function(traceable_fn):
            raise TypeError(
                f"@substitute_in_graph(...) expects a function but got {type(traceable_fn)!r}"
            )

        if not skip_signature_check:
            try:
                original_sig = inspect.signature(original_fn)
            except ValueError:
                pass
            else:
                traceable_sig = inspect.signature(traceable_fn)

                def sig_ident(
                    sig: inspect.Signature,
                ) -> tuple[tuple[str, ...], set[str], dict[str, Any]]:
                    # Ignore annotations for parameters and return type
                    return (
                        tuple(
                            p.name
                            for p in sig.parameters.values()
                            if (
                                p.kind
                                not in {
                                    p.KEYWORD_ONLY,
                                    # the name of *args and **kwargs is not important
                                    p.VAR_POSITIONAL,
                                    p.VAR_KEYWORD,
                                }
                            )
                        ),
                        {
                            p.name
                            for p in sig.parameters.values()
                            if p.kind == p.KEYWORD_ONLY
                        },
                        {
                            p.name: p.default
                            for p in sig.parameters.values()
                            # the name of *args and **kwargs is not important
                            if p.kind not in {p.VAR_POSITIONAL, p.VAR_KEYWORD}
                        },
                    )

                wildcard_sig = inspect.signature(lambda *args, **kwargs: None)

                if (
                    sig_ident(original_sig) != sig_ident(traceable_sig)
                    and sig_ident(original_sig) != sig_ident(wildcard_sig)
                    and sig_ident(traceable_sig) != sig_ident(wildcard_sig)
                ):
                    raise TypeError(
                        f"Signature mismatch between {original_fn} and {traceable_fn}: "
                        f"{original_sig} != {traceable_sig}"
                    )

        from torch._dynamo.guards import GuardBuilder
        from torch._dynamo.trace_rules import (
            _polyfilled_function_ids,
            get_torch_obj_rule_map,
        )
        from torch._dynamo.variables import PolyfilledFunctionVariable
        from torch._dynamo.variables.builder import VariableBuilder

        id_dispatch_map = VariableBuilder._id_dispatch()
        if id(original_fn) in id_dispatch_map:
            raise ValueError(
                f"Duplicate dispatch rule for {original_fn}: "
                "already registered in VariableBuilder's id dispatch map"
            )

        if id(original_fn) in _polyfilled_function_ids:
            raise ValueError(f"Duplicate polyfilled object {original_fn}")

        rule_map: dict[Any, type[VariableTracker]] = get_torch_obj_rule_map()
        if original_fn in rule_map:
            raise ValueError(
                f"Duplicate object {original_fn} with different rules: "
                f"{PolyfilledFunctionVariable}, {rule_map[original_fn]}"
            )

        polyfill_handlers: dict[Callable[..., Any], FunctionType]
        polyfill_handlers = PolyfilledFunctionVariable._get_polyfill_handlers()
        if original_fn in polyfill_handlers:
            raise ValueError(
                f"Duplicate polyfill handlers for {original_fn}: "
                f"already handled by {polyfill_handlers[original_fn]}"
            )

        # Need to wrap the function because we may cannot assign __torch_dynamo_polyfill__ to a
        # C++ function.
        @functools.wraps(traceable_fn)
        def wrapped(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            return original_fn(*args, **kwargs)

        def dispatch_fn(
            self: VariableBuilder, value: Callable[_P, _R]
        ) -> PolyfilledFunctionVariable:
            if inspect.isclass(value):
                guard_type = GuardBuilder.CLASS_MATCH
            elif inspect.ismodule(value):
                guard_type = GuardBuilder.MODULE_MATCH
            else:
                guard_type = GuardBuilder.ID_MATCH
            guards = self.install_guards(guard_type)
            assert guards is not None
            return PolyfilledFunctionVariable(
                value,
                source=self.source,
                **guards,
            )

        id_dispatch_map[id(original_fn)] = id_dispatch_map[id(wrapped)] = dispatch_fn
        _polyfilled_function_ids.add(id(original_fn))
        _polyfilled_function_ids.add(id(wrapped))
        rule_map[original_fn] = rule_map[wrapped] = PolyfilledFunctionVariable
        polyfill_handlers[original_fn] = polyfill_handlers[wrapped] = wrapped  # type: ignore[assignment]

        wrapped.__torch_dynamo_original__ = original_fn  # type: ignore[attr-defined]
        wrapped.__torch_dynamo_polyfill__ = traceable_fn  # type: ignore[attr-defined]
        wrapped.__torch_dynamo_can_constant_fold_through__ = can_constant_fold_through  # type: ignore[attr-defined]

        return wrapped  # type: ignore[return-value]

    return wrapper


# Helper function to flatten a tensor subclass and apply a function to
# all inner tensors that match the outer dim. Used to reduce duplication
# across the various marking APIs.
def _apply_func_to_inner_tensors_of_same_dim(
    func: Callable[..., Any], t: object, *args: Any, **kwargs: Any
) -> None:
    assert is_traceable_wrapper_subclass(t)

    attrs, _ctx = t.__tensor_flatten__()
    assert isinstance(t, torch.Tensor)
    for attr in attrs:
        inner = getattr(t, attr)
        if inner.dim() == t.dim():
            func(inner, *args, **kwargs)


@dataclass(frozen=True, slots=True)
class _DimRange:
    """
    This represents an dimension of a tensor and the corresponding
    min and max values it can take.  Don't create this
    class directly; instead, use :func:`mark_dynamic`.
    """

    dim: int
    min: int
    max: int


@forbid_in_graph
def mark_unbacked(
    t: Any,
    index: int | list[Any] | tuple[Any],
    hint_override: int | None = None,
    strict: bool = False,
    specialize_on: list[Any] | None = None,
    shape_id: str | None = None,
) -> None:
    """
    Mark a tensor as having an unbacked dimension. This changes the semantics of operations:
    - The size of the specified dimension will always be reported as not equal to zero or one.
    - Assertions on this index will be turned into runtime asserts.
    - Attempting to get the real value of this dimension will raise an exception.
    - In effect, this dimension is treated as data-dependent (its value is unknown).

    Args:
        t (Any): The tensor to mark as having an unbacked dimension.
        index (int or list/tuple of int): The dimension(s) to mark as unbacked. Can be a single integer or a list/tuple of integers.
        hint_override (Optional[int], default=None): An optional integer to override the size hint for this dimension.
            This is only used by the inductor backend for size hint queries, such as during autotuning.
            NOTE: changing hint_override values will cause FxGraphCache misses, since hint overrides
            affect inductor codegen decisions and are included in the cache key via
            ShapeEnv.var_to_hint_override.
        strict (bool, default=False): If True, an error will be raised if the unbacked dimension is specialized.
            By default (strict=False), specialization is allowed and will proceed without error.
        specialize_on (Optional[list[Any]], default=None): A list of specialization criteria (e.g., lambdas) for this dimension.
            If provided, Dynamo will generate specialized compiled regions for each criterion in addition to a generic trace.
        shape_id (Optional[str], default=None): An optional identifier to group unbacked dimensions together.
            All unbacked dimensions with the same shape_id will share the same unbacked symbol. This is useful when multiple tensors
            are known to have the same batch size at runtime. A runtime assertion is added
            to ensure this property at runtime.
    """
    if torch.distributed.is_available() and isinstance(
        t, torch.distributed.tensor.DTensor
    ):
        # apply on inner tensor sizes/strides
        mark_unbacked(t._local_tensor, index, shape_id=shape_id)
    else:
        # You could have copied the mark_dynamic behavior but I'm not convinced
        # it's what you want
        assert not is_traceable_wrapper_subclass(t), "not implemented yet"

    if isinstance(index, int):
        if strict:
            if not hasattr(t, "_dynamo_strict_unbacked_indices"):
                t._dynamo_strict_unbacked_indices = set()

            t._dynamo_strict_unbacked_indices.add(index)
            return

        if not hasattr(t, "_specialized_on"):
            # pyrefly: ignore [implicit-any]
            t._specialize_on = {}

        if not hasattr(t, "_dynamo_unbacked_indices"):
            t._dynamo_unbacked_indices = set()

        if not hasattr(t, "_dynamo_hint_overrides"):
            # pyrefly: ignore [implicit-any]
            t._dynamo_hint_overrides = {}

        if hint_override:
            t._dynamo_hint_overrides[index] = hint_override

        if shape_id is not None:
            if not hasattr(t, "_dynamo_shape_ids"):
                # pyrefly: ignore [implicit-any]
                t._dynamo_shape_ids = {}
            t._dynamo_shape_ids[index] = shape_id

        # FX tracers don't respect @forbid_in_graph and choke on the following error since it passes in proxies:
        # TypeError: 'Attribute' object does not support item assignment

        if isinstance(t._specialize_on, dict):
            t._specialize_on[index] = specialize_on if specialize_on is not None else []

        t._dynamo_unbacked_indices.add(index)
        return

    assert isinstance(index, (list, tuple))
    for i in index:
        mark_unbacked(t, i, shape_id=shape_id)


@forbid_in_graph
def mark_dynamic(
    t: Any,
    index: int | list[Any] | tuple[Any],
    *,
    hint_override: int | None = None,
    min: int | None = None,
    max: int | None = None,
    specialize_on: list[Any] | None = None,
) -> None:
    """
    Mark a tensor as having a dynamic dim and set corresponding min and max range for the dim.

    [Note - on the state of mark_dynamic]

    The behavior of having a dynamic dimension on a tensor is governed by a few factors:

    1) torch._dynamo.config dynamic_shapes True or False.
        a) dynamic_shapes=True - dynamic_shapes must be True for mark_dynamic to work.
        a) dynamic_shapes=False - This config will raise an exception when used in conjunction with
        mark_dynamic. We will eventually support this.

    2) If the dimension is fully constrained - as in, it does not allow more than a single value
    in both eager (torch.compile, torch._dynamo.optimize) mode and export mode (torch._dynamo.export),
    we will raise an error

    3) If the dimension is partially constrained - allowing at least 2 values but not the full unbounded
    range of shapes, in eager we will pass it through, but export will raise an error.

    4) Attempts to trace this function will explicitly raise. As such, all calls to mark_dynamic must be made
    before torch.compile.

    5) If hint_override is passed, the hint_override for the specified dimension will replace the provided value
    from the first example input as the official size hint. Note: changing hint_override values will cause
    FxGraphCache misses, since hint overrides affect inductor codegen decisions (autotuning, reduction
    strategy, etc.) and are included in the cache key via ShapeEnv.var_to_hint_override.

    6) If specialize_on is passed in, we will perform a single generic Dynamo trace followed by
    multiple specialized compilations in addition to a single generic compilation. NB: For now we only support
    per dimension specialization, or in other words we do not generate a cross product of specializations.
    At runtime, we will dispatch to a specialized compiled region if the input matches the specialization criteria.

    For example:
        mark_dynamic(..., specialize_on=[
            lambda x: x == 8,
            lambda x: x == 16
        ])

    This approach results in one Dynamo trace and two backend compilations. When the input dimension equals 8 or 16
    at runtime, execution will be directed to the specialized compiled region. Performance measurements indicate
    2-8x speedups depending on the specific specialization and model architecture.

    """
    if is_traceable_wrapper_subclass(t):
        # default behavior: mirror mark_dynamic() on all inner tensors with same dim as t
        # TODO: Make this configurable via a supported public API
        _apply_func_to_inner_tensors_of_same_dim(
            mark_dynamic, t, index, min=min, max=max
        )

    if isinstance(index, int):
        if not hasattr(t, "_dynamo_dynamic_indices"):
            t._dynamo_dynamic_indices = set()

            t._dynamo_dynamic_range = set()

            # pyrefly: ignore [implicit-any]
            t._dynamo_hint_overrides = {}

        if not hasattr(t, "_specialize_on"):
            # pyrefly: ignore [implicit-any]
            t._specialize_on = {}

        if hint_override:
            t._dynamo_hint_overrides[index] = hint_override
        # TODO(voz): Should we bounds check?

        t._dynamo_dynamic_indices.add(index)
        t._dynamo_dynamic_range.add(_DimRange(index, min, max))  # type: ignore[arg-type]

        # FX tracers don't respect @forbid_in_graph and choke on the following error since it passes in proxies:
        # TypeError: 'Attribute' object does not support item assignment

        if isinstance(t._specialize_on, dict):
            t._specialize_on[index] = specialize_on if specialize_on is not None else []

        return

    assert isinstance(index, (list, tuple))
    for i in index:
        mark_dynamic(t, i, min=min, max=max)
        mark_dynamic(t, i, min=min, max=max, specialize_on=specialize_on)


@forbid_in_graph
def maybe_mark_dynamic(t: Any, index: int | list[Any] | tuple[Any]) -> None:
    """
    Mark a tensor as having a dynamic dim, but don't enforce it (i.e., if this
    dimension ends up getting specialized, don't error).
    """
    if is_traceable_wrapper_subclass(t):
        # default behavior: mirror maybe_mark_dynamic() on all inner tensors with same dim as t
        # TODO: Make this configurable via a supported public API
        _apply_func_to_inner_tensors_of_same_dim(maybe_mark_dynamic, t, index)

    if isinstance(index, int):
        if not hasattr(t, "_dynamo_weak_dynamic_indices"):
            t._dynamo_weak_dynamic_indices = set()
        # TODO(voz): Should we bounds check?

        t._dynamo_weak_dynamic_indices.add(index)
        return

    assert isinstance(index, (list, tuple))
    for i in index:
        maybe_mark_dynamic(t, i)


def mark_static(t: Any, index: int | list[Any] | tuple[Any] | None = None) -> None:
    """
    Mark a tensor as having a static dim or mark a nn module class as static.

    For tensors
    ===========
    This will prevent us from attempting to compile it dynamically
    when dynamic=True; this can improve trace-time performance.

    This has lower precedence than mark_dynamic.

    Unlike mark_dynamic, this can be done inside a graph, in which case it
    induces specialization on the tensor.

    For nn.Module classes
    =====================
    For static nn.Module classes, TorchDynamo assumes that the module instance
    attributes will not be modified after compilation. This will ensure that
    TorchDynamo keeps integer attributes CONSTANT and not symints.

    From TorchDynamo implementation side, the instances of static-marked
    nn.Module class will be converted to UnspecializedBuiltinNNModuleVariable,
    which have the same properties.

    Note that we still have to guard on the attributes, because different
    instances of the nn.Module can have different values of the attributes. The
    key point here is that the attributes are static.
    """
    if is_compiling():
        if index is None:
            for s in t.size():
                comptime.force_static(s)
        else:
            comptime.force_static(t.size(index))
        return

    if is_traceable_wrapper_subclass(t):
        # default behavior: mirror mark_static() on all inner tensors with same dim as t
        # TODO: Make this configurable via a supported public API
        _apply_func_to_inner_tensors_of_same_dim(mark_static, t, index)

    if not isinstance(t, torch.Tensor) and issubclass(t, torch.nn.Module):
        t._dynamo_marked_static = True
        # pyrefly: ignore [bad-return]
        return t

    if not isinstance(t, torch.Tensor):
        raise TypeError(
            f"mark_static expects a tensor/nn.Module class but received {type(t)}"
        )

    if isinstance(index, int):
        if not hasattr(t, "_dynamo_static_indices"):
            t._dynamo_static_indices = set()  # type: ignore[attr-defined]
        # TODO(voz): Should we bounds check?
        t._dynamo_static_indices.add(index)  # type: ignore[attr-defined]
    elif index is None:
        for i in range(t.dim()):
            mark_static(t, i)
    else:
        assert isinstance(index, (list, tuple))
        for i in index:
            mark_static(t, i)


@forbid_in_graph
def mark_static_address(t: Any, guard: bool = False) -> None:
    """
    Marks an input tensor whose address should be treated as constant across calls to the
    same dynamo-compiled function. This indicates to cudagraphs that an extra allocation
    is not needed for this input. The data_ptr will be guarded if guard=True, and cause a full
    recompile if the data_ptr changes. Note: If this address changes, cudagraphs will re-record
    if guard=False.
    """
    if not isinstance(t, torch.Tensor):
        raise TypeError(f"mark_static_address expects a tensor but received {type(t)}")

    if guard:
        t._dynamo_static_input_type = "guarded"  # type: ignore[attr-defined]
    else:
        t._dynamo_static_input_type = "unguarded"  # type: ignore[attr-defined]


# One day, Dynamo will support tracing into einops directly (no allow_in_graph needed)
# Note that PyTorch supports multiple versions of einops, so when that day comes,
# we still need to be really careful about version matches.
def _allow_in_graph_einops() -> None:
    import einops

    # There is a lru_cache logspam issue with einops when allow_in_graph is not
    # used. Disabling this for now until the lru_cache issue is resolved.
    # if einops.__version__ >= "0.8.2":
    #     if hasattr(einops, "einops") and hasattr(einops.einops, "get_backend"):
    #         # trigger backend registration up front to avoid a later guard failure
    #         # that would otherwise cause a recompilation
    #         einops.rearrange(torch.randn(1), "i -> i")
    #     # einops 0.8.2+ don't need explicit allow_in_graph calls
    #     return

    try:
        # requires einops > 0.6.1, torch >= 2.0
        from einops._torch_specific import (  # type: ignore[attr-defined]  # noqa: F401
            _ops_were_registered_in_torchdynamo,
        )

        # einops > 0.6.1 will call the op registration logic as it is imported.
    except ImportError:
        # einops <= 0.6.1
        allow_in_graph(einops.rearrange)
        allow_in_graph(einops.reduce)
        if hasattr(einops, "repeat"):
            allow_in_graph(einops.repeat)  # available since einops 0.2.0
        if hasattr(einops, "einsum"):
            allow_in_graph(einops.einsum)  # available since einops 0.5.0
        if hasattr(einops, "pack"):
            allow_in_graph(einops.pack)  # available since einops 0.6.0
        if hasattr(einops, "unpack"):
            allow_in_graph(einops.unpack)  # available since einops 0.6.0


# Note: this carefully avoids eagerly import einops.
trace_rules.add_module_init_func("einops", _allow_in_graph_einops)


# Proxy class for torch._dynamo.config patching - so dynamo can identify context managers/decorators
# created by patch_dynamo_config, compared to ones created by a raw torch._dynamo.config.patch.
class DynamoConfigPatchProxy:
    def __init__(self, config_patch: Any) -> None:
        self.config_patch = config_patch

    @property
    def changes(self) -> dict[str, Any]:
        return self.config_patch.changes

    # Decorator implementation that simply sets up `self` as a context manager.
    # Placed in external_utils so that we can trace through it.
    __call__ = wrap_dunder_call_ctx_manager

    def __enter__(self) -> None:
        return self.config_patch.__enter__()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        return self.config_patch.__exit__(exc_type, exc_val, exc_tb)


# Criteria for patchable config:
# - Config values must be constants (i.e. int, float, str, bool, None).
#     - in particular, NO list, set, dict.
# - Traceable config patches are only useful for configs that change dynamo behavior
#   from symbolic_convert and below.
#     - e.g. patching recompile_limit won't really do anything.
# - For patching configs that affect Dynamo behavior above symbolic_convert,
#   ensure that Dynamo behaves soundly even if tracing is done with different config.
#     - e.g. be careful if patching guard-related configs as configs may have changed
#       between guard creation and evaluation.
_allowed_config_patches = (
    "verbose",
    "verify_correctness",
    "rewrite_assert_with_torch_assert",
    "capture_scalar_outputs",
    "allow_unspec_int_on_nn_module",
    "skip_torchrec",
    "dont_skip_tracing",
    "nested_graph_breaks",
)

from . import config


for name in _allowed_config_patches:
    assert hasattr(config, name), "nonexistent config"
del config


def _patch_dynamo_config_check(changes: dict[str, Any]) -> None:
    for k, v in changes.items():
        if k not in _allowed_config_patches:
            raise ValueError(
                f"patch_dynamo_config does not support patching config {k}"
            )
        if not torch._dynamo.utils.is_safe_constant(v):
            raise ValueError(
                f"patch_dynamo_config does not support patching config {k} "
                f"with non-safe-constant value {v}"
            )


# TODO: also implement nonrecursive patch_dynamo_config/dont_skip_tracing.
# Unlike config.patch, we also need to accept tuple as input in order to
# deal with context manager reconstruction.
def patch_dynamo_config(
    arg1: str | dict[str, Any] | tuple[tuple[str, Any], ...] | None = None,
    arg2: Any = None,
    **kwargs: Any,
) -> DynamoConfigPatchProxy:
    """
    A wrapper around torch._dynamo.config.patch that can be traced by Dynamo to
    temporarily change config values DURING tracing.

    See _allowed_config_patches for the list of allowed config patches.

    Arguments are the same as with torch._dynamo.config.patch.

    Can be used as a decorator or a context manager.

    User code SHOULD NOT MODIFY the return value of this function.

    WARNING: changing Dynamo config during tracing can lead to unpredictable tracing behavior!
        Proceed only as advised!
    """
    if isinstance(arg1, tuple):
        arg1 = dict(arg1)
    config_patch = torch._dynamo.config.patch(arg1, arg2, **kwargs)
    _patch_dynamo_config_check(config_patch.changes)
    # check for valid patching using config_patch.changes
    return DynamoConfigPatchProxy(config_patch)


@overload
def dont_skip_tracing(fn: None = None) -> DynamoConfigPatchProxy: ...


@overload
def dont_skip_tracing(fn: Callable[_P, _R]) -> Callable[_P, _R]: ...


def dont_skip_tracing(fn: Any | None = None) -> Any:
    """
    Context manager/decorator to trace into functions intentionally marked by developers to be skipped
    when tracing.

    This decorator will also apply to recursively invoked functions.
    """
    ctx = patch_dynamo_config(dont_skip_tracing=True)
    if fn:
        return ctx(fn)
    return ctx


@overload
def disable_nested_graph_breaks(fn: None = None) -> DynamoConfigPatchProxy: ...


@overload
def disable_nested_graph_breaks(fn: Callable[_P, _R]) -> Callable[_P, _R]: ...


def disable_nested_graph_breaks(fn: Any | None = None) -> Any:
    """
    Context manager/decorator to disable nested graph breaks when tracing
    this function and any nested functions. Used when nested graph breaks
    is causing problems.
    """
    ctx = patch_dynamo_config(nested_graph_breaks=False)
    if fn:
        return ctx(fn)
    return ctx


class ErrorOnGraphBreakDecoratorContextManager:
    def __init__(self, error_on_graph_break: bool) -> None:
        self.error_on_graph_break = error_on_graph_break

    __call__ = wrap_dunder_call_ctx_manager

    def __enter__(self) -> None:
        self.prev_error_on_graph_break = _get_error_on_graph_break()
        _set_error_on_graph_break(self.error_on_graph_break)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        _set_error_on_graph_break(self.prev_error_on_graph_break)


def error_on_graph_break(
    error_on_graph_break: bool,
) -> ErrorOnGraphBreakDecoratorContextManager:
    """
    Context manager/decorator to toggle torch.compile's `error_on_graph_break` setting at compile time.

    If `fullgraph` is set, then `error_on_graph_break` does nothing
    (i.e. `fullgraph = True` takes higher precedence). If `fullgraph` is False, then
    `error_on_graph_break` determines whether `torch.compile` throws an error upon
    encountering a graph break, or attempts to continue tracing.

    `error_on_graph_break` can be toggled during compile time with this decorator to allow graph breaks in some
    compiled regions but not others. One key difference from `fullgraph` is that `error_on_graph_break = True`
    does NOT guarantee that a single graph is captured from the compiled function.

    The default value of torch.compile's `error_on_graph_break` setting is False.
    """
    return ErrorOnGraphBreakDecoratorContextManager(error_on_graph_break)


class CudagraphOverrideContextManager:
    """Context manager that overrides cudagraph recording during tracing."""

    def __init__(self, fwd: bool | None = None, bwd: bool | None = None) -> None:
        self.fwd = fwd
        self.bwd = bwd

    __call__ = wrap_dunder_call_ctx_manager

    def __enter__(self) -> None:
        pass

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        pass


def override_cudagraphs(
    fwd: bool | None = None, bwd: bool | None = None
) -> CudagraphOverrideContextManager:
    """
    Context manager/decorator to override cudagraph recording for compiled graphs.

    When used as a context manager, overrides cudagraphs for all graph segments
    within the block (including across graph breaks).

    When used as a decorator, marks a function so that any compiled graph
    inlining it will have cudagraphs overridden.

    Args:
        fwd: If False, disable cudagraphs for forward. If True, force enable.
             If None, don't override.
        bwd: If False, disable cudagraphs for backward. If True, force enable.
             If None, don't override.
    """
    return CudagraphOverrideContextManager(fwd=fwd, bwd=bwd)


def is_dynamo_disable_recursive(method: Callable[[Any], Any]) -> bool | None:
    """
    Check if a method is marked as `dynamo_disable` recursively. It returns:
    - True if disable(recursive=True)
    - False if disable(recursive=False)
    - None if method is not a disable decorator
    """
    return getattr(method, "_torchdynamo_disable_recursive", None)
