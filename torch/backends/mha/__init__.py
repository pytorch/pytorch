# Config options to enable/disable C++ kernel for nn.functional.MHA
# and nn.TransformerEncoder
import torch

_is_fastpath_enabled: bool = True


def _get_fastpath_enabled() -> bool:
    if not torch.jit.is_scripting():
        return _is_fastpath_enabled
    return True


def _set_fastpath_enabled(value: bool) -> None:
    global _is_fastpath_enabled
    _is_fastpath_enabled = value
