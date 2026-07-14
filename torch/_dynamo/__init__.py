"""
TorchDynamo is a Python-level JIT compiler designed to make unmodified PyTorch programs faster.
TorchDynamo hooks into the frame evaluation API in CPython (PEP 523) to dynamically modify Python
bytecode right before it is executed. It rewrites Python bytecode in order to extract sequences of
PyTorch operations into an FX Graph which is then just-in-time compiled with a customizable backend.
It creates this FX Graph through bytecode analysis and is designed to mix Python execution with
compiled backends to get the best of both worlds: usability and performance. This allows it to
seamlessly optimize PyTorch programs, including those using modern Python features.
"""

import torch

from . import (
    aot_compile,
    config,
    convert_frame,
    eval_frame,
    functional_export,
    resume_execution,
)
from .backends.registry import list_backends, lookup_backend, register_backend
from .callback import callback_handler, on_compile_end, on_compile_start
from .code_context import code_context
from .convert_frame import replay
from .decorators import (
    allow_in_graph,
    assume_constant_result,
    disable,
    disable_nested_graph_breaks,
    disallow_in_graph,
    dont_skip_tracing,
    error_on_graph_break,
    forbid_in_graph,
    graph_break,
    is_dynamo_disable_recursive,
    mark_dynamic,
    mark_static,
    mark_static_address,
    maybe_mark_dynamic,
    nonstrict_trace,
    patch_dynamo_config,
    run,
    set_stance,
    skip_frame,
    step_unsupported,
    substitute_in_graph,
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

# pyrefly: ignore [deprecated]
from .external_utils import is_compiling
from .mutation_guard import GenerationTracker
from .pgo import reset_code_state
from .symbolic_convert import TensorifyState
from .utils import (
    graph_break_reasons,
    guard_failures,
    orig_code_map,
    register_hook_for_recompile_user_context,
    reset_frame_count,
    reset_recompile_user_contexts,
)


# Register polyfill functions
from .polyfills import loader as _  # usort: skip # noqa: F401


__all__ = [
    "allow_in_graph",
    "assume_constant_result",
    "config",
    "disable",
    "disable_nested_graph_breaks",
    "disallow_in_graph",
    "dont_skip_tracing",
    "export",
    "explain",
    "forbid_in_graph",
    "graph_break",
    "is_compiling",
    "is_dynamo_disable_recursive",
    "list_backends",
    "lookup_backend",
    "mark_dynamic",
    "maybe_mark_dynamic",
    "mark_static",
    "mark_static_address",
    "nonstrict_trace",
    "optimize",
    "optimize_assert",
    "OptimizedModule",
    "patch_dynamo_config",
    "register_backend",
    "replay",
    "reset",
    "reset_recompile_user_contexts",
    "run",
    "error_on_graph_break",
    "set_recursion_limit",
    "set_stance",
    "skip_frame",
    "step_unsupported",
    "substitute_in_graph",
]

# allowlist this for weights_only load of NJTs
torch.serialization.add_safe_globals([torch._dynamo.decorators._DimRange])

if torch.manual_seed is torch.random.manual_seed:
    import torch.jit._builtins

    # Wrap manual_seed with the disable decorator.
    # Can't do it at its implementation due to dependency issues.
    torch.manual_seed = torch._disable_dynamo(torch.manual_seed)
    # Add the new manual_seed to the builtin registry.
    torch.jit._builtins._register_builtin(torch.manual_seed, "aten::manual_seed")


def reset() -> None:
    """
    Clear all compile caches and restore initial state.  This function is intended
    to reset Dynamo's state *as if* you had started a fresh process invocation, which
    makes it good for testing scenarios where you want to behave as if you started
    a new process.  It does NOT affect any file system caches.

    NB: this does NOT reset logging state.  Don't use this to test logging
    initialization/reinitialization.
    """
    # TODO: https://github.com/pytorch/pytorch/issues/139200
    import logging

    log = logging.getLogger(__name__)
    log.info("torch._dynamo.reset")
    with convert_frame.compile_lock:
        reset_code_caches()
        convert_frame.input_codes.clear()
        reset_code_state()
        convert_frame.output_codes.clear()
        orig_code_map.clear()
        guard_failures.clear()
        graph_break_reasons.clear()
        resume_execution.ContinueExecutionCache.cache.clear()
        _reset_guarded_backend_cache()
        reset_frame_count()
        torch._dynamo.compiled_autograd.reset()
        convert_frame.FRAME_COUNTER = 0
        convert_frame.FRAME_COMPILE_COUNTER.clear()
        callback_handler.clear()
        GenerationTracker.clear()
        TensorifyState.clear()
        torch._dynamo.utils.warn_once_cache.clear()
        torch._C._autograd._saved_tensors_hooks_set_tracing(False)

        # Reset cudagraph trees unconditionally since they are global state
        # not tied to a specific backend instance
        if torch.cuda.is_available():
            from torch._inductor.cudagraph_trees import reset_cudagraph_trees

            reset_cudagraph_trees()


def reset_code_caches() -> None:
    """
    Clears in-memory code cache, which is what stores compiled products.  This
    resets less state than :func:`reset` and is mostly only used for testing
    purposes.
    """
    # TODO: https://github.com/pytorch/pytorch/issues/139200
    import logging

    log = logging.getLogger(__name__)
    log.info("torch._dynamo.reset_code_caches")
    """Clear compile caches that are keyed by code objects"""
    with convert_frame.compile_lock:
        reset_code_state()
        for weak_code in (
            convert_frame.input_codes.seen + convert_frame.output_codes.seen
        ):
            code = weak_code()
            if code:
                reset_code(code)
        code_context.clear()


def get_recursion_limit() -> int:
    """
    Returns the internal dynamo recursion limit set by `torch._dynamo.set_recursion_limit`.

    Returns -1 if no c recursion limit has been set.
    """
    return torch._C._dynamo.eval_frame.get_c_recursion_limit()


def set_recursion_limit(limit: int) -> None:
    """
    Sets an internal dynamo recursion limit. The limit must be >= 1, or -1 to reset
    to the default (unset) state.

    This is possibly needed in Python 3.12-3.13 since there is a separate C recursion limit
    that is not visible at the Python level. If you are getting RecursionErrors during
    Dynamo compilation and `sys.setrecursionlimit()` doesn't help, this function may alleviate
    the issue.

    NOTE: this function does NOT call `sys.setrecursionlimit()` - the user is expected to manually
        call this if required. This is because the 2 recursion limits are not sync'd up - e.g. in
        Python 3.12, functions can be inline-evaluated, which apparently doesn't use up the C stack.

    WARNING: increasing the recursion limit to an arbitrary large value may cause segfaults
        due to stack overflows! You can try also try to manually increase the stack size, e.g.
        with `$ ulimit -s ...`
    """
    torch._C._dynamo.eval_frame.set_c_recursion_limit(limit)
