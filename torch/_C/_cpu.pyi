from torch.types import _bool

# Defined in torch/csrc/cpu/Module.cpp

def is_cpu_support_vnni() -> _bool: ...
