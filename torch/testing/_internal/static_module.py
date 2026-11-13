# mypy: allow-untyped-defs
# Owner(s): ["module: unknown"]

import torch


class StaticModule:
    def __init__(self, scripted):
        # this is an nn.Module
        if hasattr(scripted, "_c"):
            self.static_module = torch._C._jit_to_static_module(scripted._c)
        else:
            self.static_module = torch._C._jit_to_static_module(scripted.graph)

    def __call__(self, *args, **kwargs):
        return self.static_module(*args, **kwargs)

    def benchmark(self, args, kwargs, warmup_runs, main_runs):
        self.static_module.benchmark(args, kwargs, warmup_runs, main_runs)

    def runAsync(self, args, kwargs):
        return self.static_module.runAsync(args, kwargs)

    def benchmark_individual_ops(self, args, kwargs, warmup_runs, main_runs):
        return self.static_module.benchmark_individual_ops(
            args, kwargs, warmup_runs, main_runs
        )
