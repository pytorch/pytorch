import operator_benchmark as op_bench
import torch


"""Microbenchmarks for arange operator"""

# Configs for PT stack operator
configs_short = op_bench.config_list(
    attr_names=["start", "end", "step"],
    attrs=[
        [0, 1000, 2.5],
        [-1024, 2048, 1],
    ],
    cross_product_configs={"device": ["cpu"], "dtype": [torch.float]},
    tags=["short"],
)

configs_long = op_bench.cross_product_configs(
    start=[-1024, 8],
    end=[16, 2048],
    step=[8, 0.1],
    device=["cpu", "cuda"],
    dtype=[torch.float, torch.bfloat16],
    tags=["long"],
)


class ArangeBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, start, end, step, dtype, device):
        self.inputs = {
            "start": start,
            "end": end,
            "step": step,
            "dtype": dtype,
            "device": device,
        }

        self.set_module_name("arange")

    def forward(self, start, end, step, dtype, device):
        return torch.arange(start=start, end=end, step=step, dtype=dtype, device=device)


op_bench.generate_pt_test(configs_short + configs_long, ArangeBenchmark)

if __name__ == "__main__":
    op_bench.benchmark_runner.main()
