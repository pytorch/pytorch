# mypy: allow-untyped-defs
# ruff: noqa: TCH004

"""
This module provides decorators and utilities for controlling TorchDynamo's behavior during compilation.
"""

import functools
import inspect
import sys
import weakref
from dataclasses import dataclass
from typing import Any, Callable, Optional, TYPE_CHECKING, TypeVar, Union
from typing_extensions import ParamSpec

import torch
from torch._environment import is_fbcode
from torch._vendor.packaging.version import Version
from torch.utils._contextlib import _DecoratorContextManager
from torch.utils._python_dispatch import is_traceable_wrapper_subclass

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
from .exc import IncorrectUsage
from .external_utils import (
    _dynamo_config_patch_proxy_dunder_call,
    get_nonrecursive_disable_wrapper,
    is_compiling,
)
from .utils import is_function


if TYPE_CHECKING:
    from types import FunctionType

    from torch._C._dynamo.eval_frame import (  # noqa: F401
        reset_code,
        set_eval_frame,
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


def run(fn=None):
    """Don't do any dynamic compiles, just use prior optimizations"""
    if fn is not None:
        fn = innermost_fn(fn)
        assert callable(fn)
        return RunOnlyContext()(fn)
    return RunOnlyContext()


def disable(fn=None, recursive=True, *, reason=None):
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
            return DisableContext(msg=reason)(fn)
        return DisableContext(msg=reason)
    else:

        def wrap(fn):
            fn = innermost_fn(fn)
            assert callable(fn)

            nonrecursive_disable_wrapper = get_nonrecursive_disable_wrapper(fn)
            nonrecursive_disable_wrapper._torchdynamo_disable = True  # type: ignore[attr-defined]
            nonrecursive_disable_wrapper._torchdynamo_disable_msg = reason  # type: ignore[attr-defined]
            nonrecursive_disable_wrapper._torchdynamo_orig_callable = fn  # type: ignore[attr-defined]
            return nonrecursive_disable_wrapper

        if fn is None:
            return wrap
        return wrap(fn)


_nonrecursive_disable_wrapper_code = disable(lambda: None, recursive=False).__code__  # type: ignore[attr-defined]
skip_code(_nonrecursive_disable_wrapper_code)


def skip(fn=None):
    """
    Skip frames associated with the function code, but still process recursively
    invoked frames
    """
    if fn is None:
        return skip
    fn = innermost_fn(fn)
    assert callable(fn)
    skip_code(fn.__code__)
    fn._torchdynamo_disable = True
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
        force_backend=None,
    ) -> None:
        if force_backend is not None and stance != "default":
            raise RuntimeError("non-default stance cannot have force_backend set")

        self.stance = DynamoStance(stance, skip_guard_eval_unsafe, force_backend)
        self.prev = _set_stance(self.stance)

    def __call__(self, fn):
        _set_stance(self.prev)
        wrapper = super().__call__(fn)
        # forbid wrapper in graph
        wrapper._dynamo_forbidden = True  # type: ignore[attr-defined]
        return wrapper

    def __enter__(self):
        _set_stance(self.stance)

    def __exit__(self, exc_type, exc_val, exc_tb):
        _set_stance(self.prev)

    def clone(self):
        return self.__class__(self.stance.stance, force_backend=self.stance.backend)


def assume_constant_result(fn):
    fn._dynamo_marked_constant = True
    return fn


def allow_in_graph(fn):
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
        def deregister():
            trace_rules._allowed_callable_ids.remove(fn_id)

        weakref.finalize(fn, deregister)
    return fn


def nonstrict_trace(traceable_fn):
    # Like `allow_in_graph`, but with the following enhancements/differences:
    #
    # 1. Supports user-defined class as inputs, as long as the class has been
    #    registered with pytree.
    # 2. Reads to global/captured tensors forces the underlying graph to treat
    #    those tensors as constant, and we _assume_ they will not be updated. This
    #    is similar to FX tracing.
    # 3. In the resulting Dynamo graph, the call to a `nonstrict_trace`-ed function
    #    will be represented as a call to `torch._higher_order_ops.flat_apply`,
    #    which takes in the `nonstrict_trace`-ed function and pytree-flattened
    #    inputs.
    # 4. Only the returned function is traceable, and the original function will
    #    not be. Moreover, `nonstrict_trace` can be used inside a `torch.compile`
    #    region.
    #
    # NOTE: like `allow_in_graph`, aliasing information is neither preserved
    # between inputs themselves, nor between inputs and outputs.
    assert callable(traceable_fn), "nonstrict_trace expects a callable"

    @functools.wraps(traceable_fn)
    def wrapped(*args, **kwargs):
        return traceable_fn(*args, **kwargs)

    wrapped_id = id(wrapped)

    # This line allows us to reuse much of the `allow_in_graph` impl.
    trace_rules._allowed_callable_ids.add(wrapped_id)

    # This line allows us to diverge the impl from `allow_in_graph`.
    trace_rules._nonstrict_trace_callable_ids.add(wrapped_id)

    # Avoid id reuse which creates subtle bugs.
    def deregister():
        trace_rules._allowed_callable_ids.remove(wrapped_id)
        trace_rules._nonstrict_trace_callable_ids.remove(wrapped_id)

    weakref.finalize(wrapped, deregister)

    return wrapped


def _disallow_in_graph_helper(throw_if_not_allowed):
    def inner(fn):
        if isinstance(fn, (list, tuple)):
            return [disallow_in_graph(x) for x in fn]
        assert callable(fn), "disallow_in_graph expects a callable"
        if (
            throw_if_not_allowed
            and trace_rules.lookup_callable(fn)
            != variables.TorchInGraphFunctionVariable
            and trace_rules.lookup(fn) != variables.TorchInGraphFunctionVariable
        ):
            raise IncorrectUsage(
                "disallow_in_graph is expected to be used on an already allowed callable (like torch.* ops). "
                "Allowed callables means callables that TorchDynamo puts as-is in the extracted graph."
            )
        trace_rules._allowed_callable_ids.remove(id(fn))
        trace_rules._nonstrict_trace_callable_ids.remove(id(fn))
        trace_rules._disallowed_callable_ids.add(id(fn))
        return fn

    return inner


def disallow_in_graph(fn):
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
def graph_break(msg=""):
    """Force a graph break"""


def forbid_in_graph(fn):
    """
    Customize which functions TorchDynamo will assert are not present while tracing.

    If you want a graph break on this function instead, use disallow_in_graph.
    TODO(voz): We now have allow_in_graph, disallow_in_graph, forbid_in_graph - some more robust
    documentation would not be amiss.
    """
    if isinstance(fn, (list, tuple)):
        return [forbid_in_graph(x) for x in fn]
    assert callable(fn), "forbid_in_graph applies only to callables"
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

                def sig_ident(sig):
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

        def dispatch_fn(self, value: Callable[_P, _R]) -> PolyfilledFunctionVariable:
            return PolyfilledFunctionVariable(
                value,
                source=self.source,
                **self.install_guards(GuardBuilder.FUNCTION_MATCH),
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
def _apply_func_to_inner_tensors_of_same_dim(func, t, *args, **kwargs):
    assert is_traceable_wrapper_subclass(t)

    attrs, _ctx = t.__tensor_flatten__()
    assert isinstance(t, torch.Tensor)
    for attr in attrs:
        inner = getattr(t, attr)
        if inner.dim() == t.dim():
            func(inner, *args, **kwargs)


@dataclass(frozen=True)
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
def mark_unbacked(t, index, strict=False):
    """
    Mark a tensor as having an unbacked dim.  This changes the semantics of operations,
    we will always report the size does not equal zero/one, we will turn asserts
    on this index into runtime asserts, and if you try to get the real value we will
    raise an exception.  In other words, we will treat this dimension as if it was
    data dependent (we do not know anything about its value.)

    For historical reasons, by default if an unbacked dim is specialized, we will
    happily specialize it and continue. If you want to error in these cases, pass
    strict=True.
    """
    # You could have copied the mark_dynamic behavior but I'm not convinced
    # it's what you want
    assert not is_traceable_wrapper_subclass(t), "not implemented yet"

    if isinstance(index, int):
        if strict:
            if not hasattr(t, "_dynamo_strict_unbacked_indices"):
                t._dynamo_strict_unbacked_indices = set()
            t._dynamo_strict_unbacked_indices.add(index)
            return

        if not hasattr(t, "_dynamo_unbacked_indices"):
            t._dynamo_unbacked_indices = set()
        t._dynamo_unbacked_indices.add(index)
        return

    assert isinstance(index, (list, tuple))
    for i in index:
        mark_unbacked(t, i)


@forbid_in_graph
def mark_dynamic(t, index, *, min=None, max=None):
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
        # TODO(voz): Should we bounds check?
        t._dynamo_dynamic_indices.add(index)
        t._dynamo_dynamic_range.add(_DimRange(index, min, max))
        return

    assert isinstance(index, (list, tuple))
    for i in index:
        mark_dynamic(t, i, min=min, max=max)


@forbid_in_graph
def maybe_mark_dynamic(t, index):
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


def mark_static(t, index=None):
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
        return t

    if not isinstance(t, torch.Tensor):
        raise TypeError(
            f"mark_static expects a tensor/nn.Module class but recieved {type(t)}"
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
def mark_static_address(t, guard=True):
    """
    Marks an input tensor whose data_ptr will not change across multiple calls
    to a dynamo-compiled function. This indicates to cudagraphs that an extra allocation
    is not needed for this input. The data_ptr will be guarded if guard=True. Note:
    Tensors marked in this way will be kept alive until `torch._dynamo.reset()` is called.
    """
    if not isinstance(t, torch.Tensor):
        raise TypeError(f"mark_static_address expects a tensor but recieved {type(t)}")

    if guard:
        t._dynamo_static_input_type = "guarded"  # type: ignore[attr-defined]
    else:
        t._dynamo_static_input_type = "unguarded"  # type: ignore[attr-defined]


# Note: this carefully avoids eagerly import einops.
# TODO: we should delete this whole _allow_in_graph_einops logic by approximately 2024 Q2
def _allow_in_graph_einops():
    mod = sys.modules.get("einops")
    if mod is None:
        return
    else:
        # version > 0.7.0 does allow_in_graph out of tree
        # for BC we need to keep this in fbcode
        # internal xref https://fb.workplace.com/groups/1026248852325474/permalink/1107135774236781/
        if Version(mod.__version__) < Version("0.7.0") or is_fbcode():
            import einops

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


trace_rules.add_module_init_func("einops", _allow_in_graph_einops)


# Proxy class for torch._dynamo.config patching - so dynamo can identify context managers/decorators
# created by patch_dynamo_config, compared to ones created by a raw torch._dynamo.config.patch.
class DynamoConfigPatchProxy:
    def __init__(self, config_patch):
        self.config_patch = config_patch

    @property
    def changes(self):
        return self.config_patch.changes

    # Decorator implementation that simply sets up `self` as a context manager.
    # Placed in external_utils so that we can trace through it.
    __call__ = _dynamo_config_patch_proxy_dunder_call

    def __enter__(self):
        return self.config_patch.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
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
)

for name in _allowed_config_patches:
    assert hasattr(torch._dynamo.config, name), "nonexistent config"


def _patch_dynamo_config_check(changes: dict[str, Any]):
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
    arg1: Optional[Union[str, dict[str, Any], tuple[tuple[str, Any], ...]]] = None,
    arg2: Any = None,
    **kwargs: Any,
) -> DynamoConfigPatchProxy:
    """
    A wrapper around torch._dynamo.config.patch that can be traced by Dynamo to
    temporarily change config values DURING tracing.

    See _allowed_config_patches for the list of allowed config patches.

    Arguments are the same as with torch._dynamo.confing.patch.

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


def dont_skip_tracing(fn=None):
    """
    Context manager/decorator to trace into functions intentionally marked by developers to be skipped
    when tracing.

    This decorator will also apply to recursively invoked functions.
    """
    ctx = patch_dynamo_config(dont_skip_tracing=True)
    if fn:
        return ctx(fn)
    return ctx
