import sys

from benchmark_base import BenchmarkBase

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch._inductor.utils import fresh_inductor_cache


class ListOfLinears(nn.Module):
    def __init__(self):
        super().__init__()
        self.linears = nn.ModuleList([nn.Linear(10, 10) for i in range(20)])

    def forward(self, x):
        # ModuleList can act as an iterable, or be indexed using ints
        for i, l in enumerate(self.linears):
            x = self.linears[i // 2](x) + l(x)
        return x


class BasicModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear1 = torch.nn.Linear(10, 10)
        self.scale = torch.randn(1, 10)

    def forward(self, x):
        return F.relu(self.linear1(x)) * self.scale


class ModuleForwardHasGraphBreak(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer1 = BasicModule()
        self.layer2 = BasicModule()
        self.layer3 = torch.nn.Sequential(BasicModule(), BasicModule())
        self.layer4 = torch.nn.ModuleList(
            [
                torch.nn.Linear(10, 10),
                torch.nn.ReLU(),
                torch.nn.Linear(10, 10),
                torch.nn.ReLU(),
            ]
        )
        self.layer5 = torch.nn.ModuleDict(
            {
                "0": torch.nn.Linear(10, 10),
            }
        )
        self.scale = torch.randn(1, 10)

    def forward(self, x):
        """
        This is used to test if the results of functions like `named_parameters`
        can be reconstructed correctly after graph break.

        https://github.com/pytorch/torchdynamo/issues/1931
        """
        x = self.layer1(x)
        params1 = dict(self.named_parameters())
        params2 = list(self.parameters())
        buffers1 = dict(self.named_buffers())
        buffers2 = list(self.buffers())
        modules1 = dict(self.named_modules())
        modules2 = list(self.modules())
        torch._dynamo.graph_break()
        y = modules2
        y = modules1
        y = buffers2
        y = buffers1
        y = params2
        y = params1
        x = (
            self.layer2(x)
            + y["layer3.1.linear1.weight"]
            + y["layer4.2.weight"]
            + y["layer5.0.weight"]
        )
        return x * self.scale


class SequentialWithDuplicatedModule(torch.nn.Module):
    # Sequential module(self.layer) contains three duplicated ReLU module.
    def __init__(self) -> None:
        super().__init__()
        self.relu = torch.nn.ReLU()
        self.layer = torch.nn.Sequential(
            torch.nn.Linear(10, 20),
            self.relu,
            torch.nn.Linear(20, 20),
            self.relu,
            torch.nn.Linear(20, 10),
            self.relu,
        )

    def forward(self, x):
        return self.layer(x)


class ModuleComparison(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer0 = torch.nn.Linear(10, 10)
        self.layer1 = torch.nn.Linear(10, 10)
        self.layer2 = torch.nn.Linear(10, 10)

    @property
    def encoder_layers(self):
        return [self.layer0, self.layer1, self.layer2]

    def forward(self, x):
        for layer in self.encoder_layers:
            output = layer(x)
            if layer is None or layer == self.layer0:
                output = F.relu6(output)
            else:
                output = F.relu(output)
        return output


class Benchmark(BenchmarkBase):
    def __init__(self, ModuleClass, backend):
        self.ModuleClass = ModuleClass
        self.backend = backend
        self._name = ModuleClass.__name__

    def name(self):
        return f"basic_modules_{self._name}_{self.backend}"

    def _prepare_once(self):
        self.m = self.ModuleClass()
        self.input = torch.ones(10)

    def _prepare(self):
        torch._dynamo.reset()

    def _work(self):
        with fresh_inductor_cache():
            opt_m = torch.compile(backend=self.backend)(self.m)
            opt_m(self.input)


def main():
    result_path = sys.argv[1]
    benchmarks = [
        Benchmark(ListOfLinears, "inductor"),
        Benchmark(ListOfLinears, "eager"),
        Benchmark(ModuleForwardHasGraphBreak, "inductor"),
        Benchmark(ModuleForwardHasGraphBreak, "eager"),
        Benchmark(SequentialWithDuplicatedModule, "inductor"),
        Benchmark(SequentialWithDuplicatedModule, "eager"),
        Benchmark(ModuleComparison, "inductor"),
        Benchmark(ModuleComparison, "eager"),
    ]
    for b in benchmarks:
        b.enable_compile_time_instruction_count().collect_all().append_results(
            result_path
        )


if __name__ == "__main__":
    main()
