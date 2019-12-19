import torch
from typing import Optional

@torch.jit.script
class SimpleNulls:
    a: Optional[int]
    b: Optional[bool]
    s: Optional[str]
    f: Optional[float]

    def __init__(self):
        self.a = 0
        self.b = False
        self.s = torch.jit.annotate(Optional[str], None)
        self.f = 0

@torch.jit.script
def foo() -> str:
    options = SimpleNulls()
    options.s = "bar"
    options.a = 7
    return options.s
