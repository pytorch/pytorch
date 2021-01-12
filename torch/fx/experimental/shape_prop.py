import torch
import torch.fx
from torch.fx.node import Node
from typing import Any, Iterator

class ShapeProp(torch.fx.Interpreter):
    def run_node(self, n : Node) -> Any:
        result = super().run_node(n)

        if isinstance(result, torch.Tensor):
            n.shape = result.shape
            n.dtype = result.dtype

        return result

    def propagate(self, *args):
        return super().run(*args)
