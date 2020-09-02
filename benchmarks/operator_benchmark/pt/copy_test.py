import operator_benchmark as op_bench
import torch


"""Microbenchmarks for copy_."""
copy_short_configs = op_bench.config_list(
    attr_names=['M', 'N', 'K'],
    attrs=[
        [1, 1, 1],
        [64, 64, 64],
        [64, 64, 128],
    ],
    cross_product_configs={
        'device': ['cpu', 'cuda'],
        'dtype_one' : [torch.int32],
        'dtype_two' : [torch.int32],
    },
    tags=['short'],
)

copy_long_configs = op_bench.cross_product_configs(
    M=[8, 128],
    N=[32, 64],
    K=[256, 512],
    device=['cpu', 'cuda'],
    dtype_one=[torch.int8, torch.int32],
    dtype_two=[torch.int8, torch.int32],
    tags=['long']
)


class CopyBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, M, N, K, device, dtype_one, dtype_two):
        self.input_one = torch.randn(M, N, K, device=device).to(dtype=dtype_one)
        self.input_two = torch.randn(M, N, K, device=device).to(dtype=dtype_two)

    def forward(self):
        return self.input_one.copy_(self.input_two)

op_bench.generate_pt_test(copy_short_configs + copy_long_configs, CopyBenchmark)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
