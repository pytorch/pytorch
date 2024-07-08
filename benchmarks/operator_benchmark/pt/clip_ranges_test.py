import operator_benchmark as op_bench
import torch


"""Microbenchmarks for ClipRanges operator."""
torch.ops.load_library("//caffe2/torch/fb/sparsenn:sparsenn_operators")

# Configs for C2 ClipRanges operator
clip_ranges_long_configs = op_bench.cross_product_configs(
    LENGTH=range(1, 100),
    M=[1],
    N=[2],
    MAX_LENGTH=range(1, 100),
    device=["cpu", "cuda"],
    dtype=[torch.int32],
    tags=["long"],
)


clip_ranges_short_configs = op_bench.config_list(
    attrs=[
        [6, 1, 2, 1, torch.int32],
        [7, 1, 2, 2, torch.int32],
        [8, 1, 2, 3, torch.int32],
        [9, 1, 2, 4, torch.int32],
        [10, 1, 2, 5, torch.int32],
    ],
    attr_names=["LENGTH", "M", "N", "MAX_LENGTH", "dtype"],
    cross_product_configs={
        "device": ["cpu", "cuda"],
    },
    tags=["short"],
)


class ClipRangesBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, LENGTH, M, N, MAX_LENGTH, device, dtype):
        self.inputs = {
            "input": torch.rand(LENGTH, M, N, device=device).type(dtype),
            "max_length": MAX_LENGTH,
        }
        self.set_module_name("clip_ranges")

    def forward(self, input, max_length: int):
        return torch.ops.fb.clip_ranges(input, max_length)


op_bench.generate_pt_test(
    clip_ranges_long_configs + clip_ranges_short_configs, ClipRangesBenchmark
)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
