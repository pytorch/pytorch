from torch.ao.quantization.fx._decomposed import quantized_decomposed_lib  # noqa: F401
from torch._export.pass_base import ExportPassBase, ProxyValue

import torch.fx

__all__ = ["AddRuntimeAssertionsForConstraintsPass"]


class AddRuntimeAssertionsForConstraintsPass(ExportPassBase):
    def __init__(self) -> None:
        super().__init__()
        self.input_traker = 0

    def call(self, graph_module: torch.fx.GraphModule) -> None:
        # resets the counter
        self.input_tracker = 0
        super().call(graph_module)

    def placeholder(self, name: str, arg, meta) -> ProxyValue:
        return super().placeholder(name, arg, meta)
    # TODO implement adding inline constraints as assertion
