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
    if isinstance(index, int):
        if not hasattr(t, "_dynamo_dynamic_indices"):
            t._dynamo_dynamic_indices = set()
        # TODO(voz): Should we bounds check?
        t._dynamo_dynamic_indices.add(index)
        return

    assert isinstance(index, (list, tuple))
    for i in index:
        mark_dynamic(t, i)
