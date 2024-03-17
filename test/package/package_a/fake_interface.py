import torch
from torch import Tensor


@torch.jit.interface
class ModuleInterface(torch.nn.Module):
    def one(self, inp1: Tensor, inp2: Tensor) -> Tensor:
        pass


class OrigModule(torch.nn.Module):
    """A module that implements ModuleInterface."""

    def one(self, inp1: Tensor, inp2: Tensor) -> Tensor:
        return inp1 + inp2 + 1

    def two(self, input: Tensor) -> Tensor:
        return input + 2

    def forward(self, input: Tensor) -> Tensor:
        return input + self.one(input, input) + 1


class NewModule(torch.nn.Module):
    """A *different* module that implements ModuleInterface."""

    def one(self, inp1: Tensor, inp2: Tensor) -> Tensor:
        return inp1 * inp2 + 1

    def forward(self, input: Tensor) -> Tensor:
        return self.one(input, input + 1)


class UsesInterface(torch.nn.Module):
    proxy_mod: ModuleInterface

    def __init__(self):
        super().__init__()
        self.proxy_mod = OrigModule()

    def forward(self, input: Tensor) -> Tensor:
        return self.proxy_mod.one(input, input)
