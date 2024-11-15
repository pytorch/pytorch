import json
import os
import sys

from benchmark_base import BenchmarkBase

import torch


class Benchmark(BenchmarkBase):
    def name(self):
        return "aotdispatcher_partitioner_cpu"

    def description(self):
        return "partitioner benchmark 1 input and 100 weights, mix of recompute and non-recompute ops"

    def _prepare_once(self):
        self.weights = [torch.randn(16, 16, requires_grad=True) for _ in range(100)]
        self.inp = torch.randn(16, 16)

    def _prepare(self):
        torch._dynamo.reset()

    def _work(self):
        @torch.compile(backend="aot_eager_decomp_partition", fullgraph=True)
        def f(inp, *weights):
            x = inp
            for w in weights:
                x = torch.matmul(w, x).sin().sin()
            return x

        f(self.inp, *self.weights)

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
                        "extra_info": {
                            "device": "cpu",
                            "description": self.description(),
                        },
                    },
                    "model": {
                        "name": self.name(),
                        "type": "aotdispatcher_partitioner",
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
        Benchmark(),
    ]

    for benchmark in all:
        benchmark.enable_compile_time_instruction_count().collect_all().append_results(
            result_path
        )


if __name__ == "__main__":
    main()
