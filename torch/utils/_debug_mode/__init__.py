# mypy: allow-untyped-defs
"""
DebugMode: a debugging TorchDispatchMode that intercepts and logs runtime calls.

See torch.utils._debug_mode._mode for the full implementation and docstring.
"""

from torch.utils._debug_mode._calls import (
    _AnnotateCall,
    _DebugCall,
    _ExternalCall,
    _get_call_name,
    _OpCall,
    _OutputPlacementCall,
    _RedistributeCall,
    _TritonKernelCall,
)
from torch.utils._debug_mode._mode import (
    _maybe_record_external,
    DebugInterpreter,
    DebugMode,
    get_active_debug_mode,
    register_context_manager_intercept,
    register_function_intercept,
    unregister_context_manager_intercept,
    unregister_function_intercept,
)
from torch.utils._debug_mode._utils import (
    _stringify_shape,
    hash_tensor_fn,
    norm_hash_fn,
    TensorIdTracker,
)


__all__ = [
    "DebugMode",
    "get_active_debug_mode",
    "_maybe_record_external",
    "register_context_manager_intercept",
    "register_function_intercept",
    "unregister_context_manager_intercept",
    "unregister_function_intercept",
]
