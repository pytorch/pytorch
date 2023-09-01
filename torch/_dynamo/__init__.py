import torch
from . import allowed_functions, convert_frame, eval_frame, resume_execution
from .backends.registry import list_backends, register_backend
from .convert_frame import replay
from .decorators import (
    allow_in_graph,
    assume_constant_result,
    disable,
    disallow_in_graph,
    forbid_in_graph,
    graph_break,
    mark_dynamic,
    mark_static,
    mark_static_address,
    maybe_mark_dynamic,
    run,
)
from .eval_frame import (
    explain,
    export,
    is_dynamo_supported,
    optimize,
    optimize_assert,
    OptimizedModule,
    reset_code,
)
from .external_utils import is_compiling
from .utils import graph_break_reasons, guard_failures, orig_code_map, reset_frame_count

__all__ = [
    "allow_in_graph",
    "assume_constant_result",
    "disallow_in_graph",
    "forbid_in_graph",
    "graph_break",
    "mark_dynamic",
    "maybe_mark_dynamic",
    "mark_static",
    "mark_static_address",
    "optimize",
    "optimize_assert",
    "export",
    "explain",
    "run",
    "replay",
    "disable",
    "reset",
    "OptimizedModule",
    "is_compiling",
    "register_backend",
    "list_backends",
]


def reset() -> None:
    """Clear all compile caches and restore initial state"""
    for weak_code in convert_frame.input_codes.seen + convert_frame.output_codes.seen:
        code = weak_code()
        if code:
            reset_code(code)
    convert_frame.input_codes.clear()
    convert_frame.output_codes.clear()
    orig_code_map.clear()
    guard_failures.clear()
    graph_break_reasons.clear()
    resume_execution.ContinueExecutionCache.cache.clear()
    if hasattr(eval_frame.most_recent_backend, "reset"):
        eval_frame.most_recent_backend.reset()
    eval_frame.most_recent_backend = None
    reset_frame_count()
    torch._C._dynamo.compiled_autograd.clear_cache()
