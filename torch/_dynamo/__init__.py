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

from . import config, convert_frame, eval_frame, resume_execution
from .backends.registry import list_backends, lookup_backend, register_backend
from .callback import callback_handler, on_compile_end, on_compile_start
from .code_context import code_context
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
    nonstrict_trace,
    run,
    set_stance,
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
from .external_utils import is_compiling
from .mutation_guard import GenerationTracker
from .pgo import reset_code_state
from .symbolic_convert import TensorifyState
from .utils import graph_break_reasons, guard_failures, orig_code_map, reset_frame_count


# Register polyfill functions
from .polyfills import loader as _  # usort: skip # noqa: F401


__all__ = [
    "allow_in_graph",
    "assume_constant_result",
    "disallow_in_graph",
    "forbid_in_graph",
    "substitute_in_graph",
    "graph_break",
    "mark_dynamic",
    "maybe_mark_dynamic",
    "mark_static",
    "mark_static_address",
    "nonstrict_trace",
    "optimize",
    "optimize_assert",
    "export",
    "explain",
    "run",
    "replay",
    "disable",
    "set_stance",
    "reset",
    "OptimizedModule",
    "is_compiling",
    "register_backend",
    "list_backends",
    "lookup_backend",
    "config",
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
        torch._dynamo.utils.user_obj_id_to_weakref.clear()
        torch._C._autograd._saved_tensors_hooks_set_tracing(False)


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
