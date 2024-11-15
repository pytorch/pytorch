import json
import os
import sys

from benchmark_base import BenchmarkBase

import torch


class Benchmark(BenchmarkBase):
    N = 100

    def name(self):
        return "sum_floordiv_regression"

    def description(self):
        return "information at https://github.com/pytorch/pytorch/issues/134133"

    def _prepare_once(self):
        class M(torch.nn.Module):
            def forward(self, x):
                total = sum(t.item() for t in x)
                return total // 2

        self.m = M()
        self.input = [torch.tensor(i + 2) for i in range(self.N)]

    def _prepare(self):
        torch._dynamo.reset()

    def _work(self):
        torch.export.export(self.m, (self.input,))

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
                        "type": "sum_floordiv",
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
    Benchmark().enable_compile_time_instruction_count().collect_all().append_results(
        result_path
    )


if __name__ == "__main__":
    main()
