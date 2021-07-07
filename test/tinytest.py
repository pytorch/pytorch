import torch
from typing import Union

@torch.jit.script
def fn(x: Union[int, None]) -> int:
    if x is not None:
        return x
    else:
        return 5
