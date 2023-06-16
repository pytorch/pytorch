from . import amp
from torch._C._cpu import is_cpu_support_vnni

def _is_cpu_support_vnni() -> bool:
    r"""Returns a bool indicating if CPU supports VNNI."""
    return is_cpu_support_vnni()
