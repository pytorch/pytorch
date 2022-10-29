from . import allowed_functions, convert_frame, eval_frame, resume_execution
from .convert_frame import replay
from .eval_frame import (
    assume_constant_result,
    disable,
    explain,
    export,
    optimize,
    optimize_assert,
    reset_code,
    run,
    skip,
)
from .utils import compilation_metrics, guard_failures, orig_code_map

__all__ = [
    "assume_constant_result",
    "optimize",
    "optimize_assert",
    "export",
    "explain",
    "run",
    "replay",
    "disable",
    "reset",
    "list_backends",
    "skip",
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


def list_backends():
    """
    Return valid strings that can be passed to:
        @torch._dynamo.optimize(<backend>)
        def foo(...):
           ....
    """
    from .optimizations import BACKENDS

    return [*sorted([*BACKENDS.keys(), "inductor"])]


def allow_in_graph(fn):
    """
    Customize which functions TorchDynamo will include in the generated
    graph.  Similar to torch.fx.wrap().

        torch._dynamo.allow_in_graph(my_custom_function)

        @torch._dynamo.optimize(...)
        def fn(a):
            x = torch.add(x, 1)
            x = my_custom_function(x)
            x = torch.add(x, 1)
            return x

        fn(...)

    Will capture a single graph containing my_custom_function().
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

        torch._dynamo.disallow_in_graph(torch.sub)

        @torch._dynamo.optimize(...)
        def fn(a):
            x = torch.add(x, 1)
            x = torch.sub(x, 1)
            x = torch.add(x, 1)
            return x

        fn(...)

    Will break the graph on torch.sub, and give two graphs each with a
    single torch.add() op.
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
