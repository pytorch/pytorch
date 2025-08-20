import operator_benchmark as op_bench

import torch


"""Microbenchmarks for torch.mm."""

# Specify additional metrics to compute for this benchmark. Supported values
# include "flops" and "bytes". Set this to a list like ["flops", "bytes"] to
# enable TFLOP/s and memory bandwidth reporting. Leave as None to run the
# benchmark with default latency-only metrics.
metrics = None

# Benchmark ops performance without broadcast
ops_list = op_bench.op_list(
    attr_names=["op_name", "op_func"],
    attrs=[["mm", torch.mm]],
)

mm_short_configs = op_bench.config_list(
    attr_names=["M", "N", "K"],
    attrs=[
        [1, 1, 1],
        [64, 64, 64],
        [64, 64, 128],
    ],
    cross_product_configs={
        "device": ["cpu"],
        "dtype": [torch.float],
    },
    tags=["short"],
)

mm_long_configs = op_bench.cross_product_configs(
    M=[8, 128],
    N=[32, 64],
    K=[256, 512],
    device=["cpu", "cuda"],
    dtype=[torch.float, torch.bfloat16],
    tags=["long"],
)


class MmOpBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, M, N, K, device, dtype, op_func):
        self.inputs = {
            "input_one": torch.randn(M, N, device=device).to(dtype=dtype),
            "input_two": torch.randn(N, K, device=device).to(dtype=dtype),
        }
        self.op_func = op_func
        if metrics:
            self.extra_metrics = {}
            if "flops" in metrics:
                self.extra_metrics["flops"] = 2 * M * N * K
            if "bytes" in metrics:
                element_size = torch.tensor([], dtype=dtype).element_size()
                self.extra_metrics["bytes"] = (
                    (M * N + N * K + M * K) * element_size
                )

    def forward(self, input_one, input_two):
        return self.op_func(input_one, input_two)


op_bench.generate_pt_tests_from_op_list(
    ops_list, mm_short_configs + mm_long_configs, MmOpBenchmark
)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
