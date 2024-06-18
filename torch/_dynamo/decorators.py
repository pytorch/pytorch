# mypy: allow-untyped-defs
# ruff: noqa: TCH004
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch.utils._python_dispatch import is_traceable_wrapper_subclass
from . import trace_rules, variables
from .comptime import comptime
from .eval_frame import DisableContext, innermost_fn, RunOnlyContext
from .exc import IncorrectUsage
from .external_utils import is_compiling

if TYPE_CHECKING:
    from torch._C._dynamo.eval_frame import (  # noqa: F401
        reset_code,
        set_eval_frame,
        set_guard_error_hook,
        skip_code,
        unsupported,
    )
else:
    for name in dir(torch._C._dynamo.eval_frame):
        if name.startswith("__"):
            continue
        globals()[name] = getattr(torch._C._dynamo.eval_frame, name)


def run(fn=None):
    """Don't do any dynamic compiles, just use prior optimizations"""
    if fn is not None:
        fn = innermost_fn(fn)
        assert callable(fn)
        return RunOnlyContext()(fn)
    return RunOnlyContext()


def disable(fn=None, recursive=True):
    """
    Decorator and context manager to disable TorchDynamo

    If recursive=True, Dynamo is completely skipped on the decorated function
    frame as well as the recursively invoked functions.

    If recursive=False, Dynamo skips frames associated with the function code,
    but still process recursively invoked frames.
    """
    if recursive:
        if fn is not None:
            fn = innermost_fn(fn)
            assert callable(fn)
            return DisableContext()(fn)
        return DisableContext()
    else:
        return skip(fn)


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
        trace_rules._disallowed_callable_ids.remove(id(fn))
        trace_rules._allowed_callable_ids.add(id(fn))
    return fn


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
def graph_break():
    """Force a graph break"""
    pass


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


# Helper function to flatten a tensor subclass and apply a function to
# all inner tensors that match the outer dim. Used to reduce duplication
# across the various marking APIs.
def _apply_func_to_inner_tensors_of_same_dim(func, t, *args, **kwargs):
    assert is_traceable_wrapper_subclass(t)

    attrs, ctx = t.__tensor_flatten__()
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
def mark_unbacked(t, index):
    """
    Mark a tensor as having an unbacked dim.  This changes the semantics of operations,
    we will always report the size does not equal zero/one, we will turn asserts
    on this index into runtime asserts, and if you try to get the real value we will
    raise an exception.  In other words, we will treat this dimension as if it was
    data dependent (we do not know anything about its value.)
    """
    # You could have copied the mark_dynamic behavior but I'm not convinced
    # it's what you want
    assert not is_traceable_wrapper_subclass(t), "not implemented yet"

    if isinstance(index, int):
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
    Mark a tensor as having a static dim.

    This will prevent us from attempting to compile it dynamically
    when dynamic=True; this can improve trace-time performance.

    This has lower precedence than mark_dynamic.

    Unlike mark_dynamic, this can be done inside a graph, in which case it
    induces specialization on the tensor.
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

    if isinstance(index, int):
        if not hasattr(t, "_dynamo_static_indices"):
            t._dynamo_static_indices = set()
        # TODO(voz): Should we bounds check?
        t._dynamo_static_indices.add(index)
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
    import einops

    try:
        # requires einops > 0.6.1, torch >= 2.0
        from einops._torch_specific import (  # type: ignore[attr-defined]  # noqa: F401
            _ops_were_registered_in_torchdynamo,
        )

        # einops > 0.6.1 will call the op registration logic as it is imported.
        pass
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
