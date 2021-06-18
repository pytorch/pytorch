import torch
from typing import Union, Any, Dict, List

@torch.jit.script
def fn(a):
    # type: (Optional[int]) -> int
    if isinstance(a, int):
        return a + 3
    else:
        return 4

print(fn(None))


"""
change logic for normal type to the logic starting on 1866
"""
