from collections import defaultdict
from typing import Optional

from torch.fx.experimental.symbolic_shapes import MinMaxConstraint
from . import allowed_functions, convert_frame, eval_frame, resume_execution
from .backends.registry import list_backends, register_backend
from .convert_frame import replay
from .eval_frame import (
    assume_constant_result,
    disable,
    explain,
    export,
    is_dynamo_supported,
    optimize,
    optimize_assert,
    OptimizedModule,
    reset_code,
    run,
    skip,
)
from .external_utils import is_compiling
from .utils import compilation_metrics, guard_failures, orig_code_map, reset_frame_count

__all__ = [
    "allow_in_graph",
    "assume_constant_result",
    "disallow_in_graph",
    "forbid_in_graph",
    "graph_break",
    "mark_dynamic",
    "optimize",
    "optimize_assert",
    "export",
    "explain",
    "run",
    "replay",
    "disable",
    "reset",
    "skip",
    "OptimizedModule",
    "is_compiling",
    "register_backend",
    "list_backends",
    "mark_dynamic_constrain",
    "clear_dynamic",
]


def reset():
    """Clear all compile caches and restore initial state"""
    for weak_code in convert_frame.input_codes.seen + convert_frame.output_codes.seen:
        code = weak_code()
        if code:
            reset_code(code)
    convert_frame.input_codes.clear()
    convert_frame.output_codes.clear()
    orig_code_map.clear()
    guard_failures.clear()
    resume_execution.ContinueExecutionCache.cache.clear()
    eval_frame.most_recent_backend = None
    compilation_metrics.clear()
    reset_frame_count()


def allow_in_graph(fn):
    """
    Customize which functions TorchDynamo will include in the generated
    graph. Similar to `torch.fx.wrap()`.
    ::

        torch._dynamo.allow_in_graph(my_custom_function)

        @torch._dynamo.optimize(...)
        def fn(a):
            x = torch.add(x, 1)
            x = my_custom_function(x)
            x = torch.add(x, 1)
            return x

        fn(...)

    Will capture a single graph containing `my_custom_function()`.
    """
    if isinstance(fn, (list, tuple)):
        return [allow_in_graph(x) for x in fn]
    assert callable(fn), "allow_in_graph expects a callable"
    allowed_functions._allowed_function_ids.add(id(fn))
    allowed_functions._disallowed_function_ids.remove(id(fn))
    return fn


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
    if isinstance(fn, (list, tuple)):
        return [disallow_in_graph(x) for x in fn]
    assert callable(fn), "disallow_in_graph expects a callable"
    allowed_functions._allowed_function_ids.remove(id(fn))
    allowed_functions._disallowed_function_ids.add(id(fn))
    return fn


@disallow_in_graph
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


@forbid_in_graph
def mark_dynamic(t, index):
    """
    Mark a tensor as having a dynamic dim.

    [Note - on the state of mark_dynamic]

    The behavior of having a dynamic dimension on a tensor is governed by a few factors:

    1) torch._dynamo.config dynamic_shapes True or False.
        a) dynamic_shapes=True - dynamic_shapes must be True for mark_dynamic to work.
        a) dynamic_shapes=False - This config will raise an exception when used in conjunction with
        mark_dyamic. We will eventually support this.

    2) If the dimension is fully constrained - as in, it does not allow more than a single value
    in both eager (torch.compile, torch._dynamo.optimize) mode and export mode (torch._dynamo.export),
    we will raise an error

    3) If the dimension is partially constrained - allowing at least 2 values but not the full unbounded
    range of shapes, in eager we will pass it through, but export will raise an error.

    4) Attempts to trace this function will explicitly raise. As such, all calls to mark_dynamic must be made
    before torch.compile.

    """
    mark_dynamic_constrain(t, index, min=None, max=None)


@forbid_in_graph
def mark_dynamic_constrain(
    t, index, *, min: Optional[int] = None, max: Optional[int] = None
):
    """
    To fully understand this API, please read the documentation for mark_dynamic first,
    as this API is an enrichment over that API.

    In its current state, ``mark_dynamic_constrain`` fully subsumes ``mark_dynamic``.

    ``mark_dynamic_constrain`` allows users to provide a directive that the dimension will fall
    within a given range. A range can be unbounded on either ``min``, or ``max``. Multiple calls to this API for
    the same dimension will fail. At guard accumulation time, we verify that the dimension fell within the
    specified range, and raise if it does not.

    Example usage is as follow:

    ::

        x = torch.randn([7, 7, 7])

        def my_dyn_fn(a):
            if a.shape[0] > 5:
                return a.cos()
            return a.sin()

        torch._dynamo.mark_dynamic_constrain(x, 0, min=4, max=10)
        torch._dynamo.optimize("eager")(my_dyn_fn)(x)

    We will get a new guard, ``4 <= a.size()[0] <= 10``.

    If we run it again, with a wider constraint, by adding these 2 lines:

    ::

        torch._dynamo.mark_dynamic_constrain(x, 0, min=4, max=10)
        torch._dynamo.optimize("eager")(my_dyn_fn)(x)

    Nothing happens - ``mark_dynamic_constrain`` is sticky unless reset, so the range is still
    at the narrowest intersection (4, 10).

    If we delete the field first:

    ::

        torch._dynamo.clear_dynamic(x, 0)
        torch._dynamo.mark_dynamic_constrain(x, 0, min=3, max=12)
        torch._dynamo.optimize("eager")(my_dyn_fn)(x)

    We will recompile, and get a new guard, ``3 <= a.size()[0] <= 12``.

    Alternatively, if our directive had been counter to the guards:

    ::

        x = torch.randn([7, 7, 7])

        def my_dyn_fn(a):
            if a.shape[0] > 5:
                return a.cos()
            return a.sin()

        torch._dynamo.optimize("eager")(my_dyn_fn)(x)
        torch._dynamo.mark_dynamic_constrain(x, 0, min=2, max=4)

    We would raise.

    This API behaves identically for eager and export.
    """
    if isinstance(index, int):
        if not hasattr(t, "_dynamo_dynamic_ranges"):
            t._dynamo_dynamic_ranges = defaultdict(MinMaxConstraint.NONE)
        # TODO(voz): Should we bounds check?
        new_range = MinMaxConstraint(min=min, max=max)
        if index in t._dynamo_dynamic_ranges:
            raise RuntimeError(
                f"Attempt to constrain already constrained index {index}"
            )
        t._dynamo_dynamic_ranges[index] = new_range
        return

    assert isinstance(index, (list, tuple))
    for i in index:
        mark_dynamic_constrain(t, i, min=min, max=max)


@forbid_in_graph
def clear_dynamic(t, index):
    """
    Marks a given index or list of indices as not dynamic.

    :param t: The tensor object to operate on.
    :type t: tensor
    :param index: The index or list of indices to mark as not dynamic.
    :type index: int or list[int]
    :raises AssertionError: If the tensor does not have
        any dynamic dimensions.
    """
    if isinstance(index, int):
        assert hasattr(
            t, "_dynamo_dynamic_ranges"
        ), "Illegal call to clear without dynamic dims"
        del t._dynamo_dynamic_ranges[index]
        return

    assert isinstance(index, (list, tuple))
    for i in index:
        clear_dynamic(t, i)


@forbid_in_graph
def has_dynamic_dims(t):
    return hasattr(t, "_dynamo_dynamic_ranges")


@forbid_in_graph
def clear_dynamic_dims(t):
    assert has_dynamic_dims(t)
    delattr(t, "_dynamo_dynamic_ranges")
