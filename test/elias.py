import torch
from typing import NamedTuple

class Params(NamedTuple):
    p1: float
    p2: int

class MyModule(torch.nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params

    def forward(self):
        print(self.params.p1)

params = Params(1.0, 2)
m = torch.jit.script(MyModule(params))
m()
