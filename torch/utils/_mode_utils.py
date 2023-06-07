import torch
from typing import TypeVar

T = TypeVar('T')

# returns if all are the same mode
def all_same_mode(modes):
    return all(tuple(mode == modes[0] for mode in modes))

no_dispatch = torch._C._DisableTorchDispatch
