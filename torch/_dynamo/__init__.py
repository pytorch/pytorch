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
    bytecode_debugger,
    config,
    convert_frame,
    eval_frame,
    functional_export,
    resume_execution,
)
from . import _bytecode_debugger_fix as _bytecode_debugger_fix
from .backends.registry import list_backends, lookup_backend, register_backend
from .callback import callback_handler, on_compile_end, on_compile_start
from .code_context import code_context
from .convert_frame import replay
from .decorators import (
    allow_c_slot,
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
    override_cudagraphs,
    override_optimization_hint,
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
from .polyfills import loader as _  # usort: skip


__all__ = [
    "allow_c_slot",
    "allow_in_graph",
    "assume_constant_result",
    "bytecode_debugger",
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
    "override_optimization_hint",
    "optimize",
    "optimize_assert",
    "OptimizedModule",
    "patch_dynamo_config",
    "register_backend",
    "replay",
    "reset",
    "reset_recompile_user_contexts",
    "run",
    "override_cudagraphs",
    "error_on_graph_break",
    "set_recursion_limit",
    "set_stance",
    "skip_frame",
]
