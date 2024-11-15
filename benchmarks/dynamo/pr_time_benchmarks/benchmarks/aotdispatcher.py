import json
import os
import sys

from benchmark_base import BenchmarkBase

import torch
from torch.testing._internal.two_tensor import TwoTensor


class Benchmark(BenchmarkBase):
    def __init__(self, *, training, subclass):
        self._training = training
        self._subclass = subclass
        self._device = "cpu"

    def name(self):
        prefix = "aotdispatcher"
        if self._training:
            prefix += "_training"
        else:
            prefix += "_inference"
        if self._subclass:
            prefix += "_subclass"
        else:
            prefix += "_nosubclass"
        if self._device == "cpu":
            prefix += "_cpu"
        return prefix

    def description(self):
        return "100 inputs, 100 outputs, each input is added once"

    def _prepare_once(self):
        _args = [
            torch.ones(100, requires_grad=self._training, device=self._device)
            for _ in range(100)
        ]
        if self._subclass:
            _args = [
                TwoTensor(x, x.clone().detach().requires_grad_(self._training))
                for x in _args
            ]
        self._args = _args

    def _prepare(self):
        torch._dynamo.reset()

    def _work(self):
        @torch.compile(backend="aot_eager_decomp_partition", fullgraph=True)
        def f(*args):
            outs = [torch.add(x, x) for x in args]
            return outs

        f(*self._args)

    def _write_to_json(self, output_dir: str):
        records = []
        for entry in self.results:
            metric_name = entry[1]
            value = entry[2]

            if not metric_name or value is None:
                continue

            records.append(
                {
                    "benchmark": {
                        "name": "pr_time_benchmarks",
                        "mode": "training" if self._training else "inference",
                        "extra_info": {
                            "subclass": self._subclass,
                            "device": self._device,
                            "description": self.description(),
                        },
                    },
                    "model": {
                        "name": self.name(),
                        "type": "aotdispatcher",
                        "backend": "aot_eager_decomp_partition",
                    },
                    "metric": {
                        "name": metric_name,
                        "benchmark_values": [value],
                    },
                }
            )

        with open(os.path.join(output_dir, f"{self.name()}.json"), "w") as f:
            json.dump(records, f)


def main():
    result_path = sys.argv[1]
    all = [
        Benchmark(training=False, subclass=False),
        Benchmark(training=True, subclass=False),
        Benchmark(training=False, subclass=True),
        Benchmark(training=True, subclass=True),
    ]

    for benchmark in all:
        benchmark.enable_compile_time_instruction_count().collect_all().append_results(
            result_path
        )


if __name__ == "__main__":
    main()
