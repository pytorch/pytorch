import torch

__all__ = []

# error: Module has no attribute "_return_types"
return_types = torch._C._return_types  # type: ignore[attr-defined]

for name in dir(return_types):
    if name.startswith('__'):
        continue
    globals()[name] = getattr(return_types, name)
    __all__.append(name)
