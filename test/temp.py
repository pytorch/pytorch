import torch
from typing import Optional

def foo(x: Optional[int], y: bool):
    if y == None:
        return x + x
    return None
