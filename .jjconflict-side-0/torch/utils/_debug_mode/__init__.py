# mypy: allow-untyped-defs
"""
DebugMode: a debugging TorchDispatchMode that intercepts and logs runtime calls.

See torch.utils._debug_mode._mode for the full implementation and docstring.
"""

from torch.utils._debug_mode._calls import (
    _AnnotateCall,
    _DebugCall,
    _get_call_name,
    _OpCall,
    _OutputPlacementCall,
    _RedistributeCall,
    _TritonKernelCall,
)
from torch.utils._debug_mode._mode import (
    DebugInterpreter,
    DebugMode,
    get_active_debug_mode,
)
from torch.utils._debug_mode._utils import (
    _stringify_shape,
    hash_tensor_fn,
    norm_hash_fn,
    TensorIdTracker,
)


__all__ = ["DebugMode", "get_active_debug_mode"]
