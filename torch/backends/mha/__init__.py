# Config options to enable/disable C++ kernel for nn.functional.MHA
# and nn.TransformerEncoder
import torch

_is_fastpath_enabled: bool = True

def get_fastpath_enabled() -> bool:
    """Returns whether fast path is enabled, or ``True`` if jit is scripting."""
    if not torch.jit.is_scripting():
        return _is_fastpath_enabled
    return True

def set_fastpath_enabled(value: bool) -> None:
    """Sets whether fast path is enabled"""
    global _is_fastpath_enabled
    _is_fastpath_enabled = value
