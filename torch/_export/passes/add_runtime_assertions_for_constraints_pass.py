import torch
from torch.ao.quantization.fx._decomposed import quantized_decomposed_lib  # noqa: F401
from torch._subclasses.fake_tensor import FakeTensor
from torch._export.pass_base import ExportPassBase, ProxyValue

__all__ = ["AddRuntimeAssertionsForConstraintsPass"]


class AddRuntimeAssertionsForConstraintsPass(ExportPassBase):
    def call_operator(self, op, args, kwargs, meta):
        stuff = super().call_operator(op, args, kwargs, meta)
        print("PRINT", self.current_gm)
        return stuff
