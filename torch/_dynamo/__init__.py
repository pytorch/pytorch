import torch

from . import convert_frame, eval_frame, resume_execution
from .backends.registry import list_backends, lookup_backend, register_backend
from .callback import callback_handler, on_compile_end, on_compile_start
from .code_context import code_context
from .convert_frame import replay
from .decorators import (
    allow_in_graph,
    assume_constant_result,
    disable,
    disallow_in_graph,
    enable,
    forbid_in_graph,
    graph_break,
    mark_dynamic,
    mark_static,
    mark_static_address,
    maybe_mark_dynamic,
    run,
)
from .eval_frame import (
    _reset_guarded_backend_cache,
    explain,
    export,
    is_dynamo_supported,
    is_inductor_supported,
    optimize,
    optimize_assert,
    OptimizedModule,
    reset_code,
)
from .external_utils import is_compiling
from .mutation_guard import GenerationTracker
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
    "enable",
    "reset",
    "OptimizedModule",
    "is_compiling",
    "register_backend",
    "list_backends",
    "lookup_backend",
]

if torch.manual_seed is torch.random.manual_seed:
    import torch.jit._builtins

    # Wrap manual_seed with the disable decorator.
    # Can't do it at its implementation due to dependency issues.
    torch.manual_seed = torch._disable_dynamo(torch.manual_seed)
    # Add the new manual_seed to the builtin registry.
    torch.jit._builtins._register_builtin(torch.manual_seed, "aten::manual_seed")


def reset() -> None:
    """Clear all compile caches and restore initial state"""
    with convert_frame.compile_lock:
        reset_code_caches()
        convert_frame.input_codes.clear()
        convert_frame.output_codes.clear()
        orig_code_map.clear()
        guard_failures.clear()
        graph_break_reasons.clear()
        resume_execution.ContinueExecutionCache.cache.clear()
        _reset_guarded_backend_cache()
        reset_frame_count()
        torch._C._dynamo.compiled_autograd.clear_cache()
        convert_frame.FRAME_COUNTER = 0
        convert_frame.FRAME_COMPILE_COUNTER.clear()
        callback_handler.clear()
        GenerationTracker.clear()
        torch._dynamo.utils.warn_once_cache.clear()
        torch._C._autograd._saved_tensors_hooks_set_tracing(False)


def reset_code_caches() -> None:
    """Clear compile caches that are keyed by code objects"""
    with convert_frame.compile_lock:
        for weak_code in (
            convert_frame.input_codes.seen + convert_frame.output_codes.seen
        ):
            code = weak_code()
            if code:
                reset_code(code)
        code_context.clear()
