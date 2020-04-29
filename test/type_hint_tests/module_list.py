from typing import Iterable

import torch

# ModuleList with elements of a specific type
class FooModule(torch.nn.Module):
    def ten(self) -> int:
        return 10

class FooCollector(torch.nn.Module):
    def __init__(self, ml: Iterable[FooModule]) -> None:
        super(FooCollector, self).__init__()
        self.ml: torch.nn.ModuleList[FooModule] = torch.nn.ModuleList(ml)

    def foo_sum(self) -> int:
        return sum(foo.ten() for foo in self.ml)

collector = FooCollector([FooModule(), FooModule()])
twenty = collector.foo_sum()
twenty == 20

# ModuleList with elements of type Module
class BarModule(torch.nn.Module):
    pass

ml: torch.nn.ModuleList = torch.nn.ModuleList([FooModule(), BarModule()])
ml[0].children() == []
