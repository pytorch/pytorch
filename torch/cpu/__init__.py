from . import amp
import torch

def _is_cpu_support_vnni() -> bool:
    r"""Returns a bool indicating if CPU supports VNNI."""
    return torch._C._cpu._is_cpu_support_vnni()
