from typing import Any

from torch.types import _bool

# Defined in torch/csrc/cpu/Module.cpp

def _init_amx() -> _bool: ...
def _get_cpu_capability() -> dict[str, Any]: ...
