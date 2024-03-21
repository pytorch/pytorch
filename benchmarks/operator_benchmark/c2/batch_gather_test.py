import benchmark_caffe2 as op_bench_c2
import numpy
from benchmark_caffe2 import Caffe2BenchmarkBase  # noqa: F401
from caffe2.python import core

import operator_benchmark as op_bench


"""Microbenchmarks for element-wise BatchGather operator."""

# Configs for C2 BatherGather operator
batch_gather_configs_short = op_bench.config_list(
    attr_names=["M", "N", "K"],
    attrs=[
        [8, 8, 1],
        [256, 512, 1],
        [512, 512, 1],
        [8, 8, 2],
        [256, 512, 2],
        [512, 512, 2],
    ],
    cross_product_configs={
        "device": ["cpu", "cuda"],
    },
    tags=["short"],
)

batch_gather_configs_long = op_bench.cross_product_configs(
    M=[128, 1024], N=[128, 1024], K=[1, 2], device=["cpu", "cuda"], tags=["long"]
)


class BatchGatherBenchmark(op_bench_c2.Caffe2BenchmarkBase):
    def init(self, M, N, K, device):
        self.input_one = self.tensor([M, N, K], device=device)
        max_val = N
        numpy.random.seed((1 << 32) - 1)
        index_dim = numpy.random.randint(0, N)
        self.index = self.feed_tensor(
            numpy.random.randint(0, max_val, index_dim), device=device
        )
        self.output = self.tensor([M, index_dim, K], device=device)
        self.set_module_name("batch_gather")

    def forward(self):
        op = core.CreateOperator(
            "BatchGather", [self.input_one, self.index], self.output
        )
        return op


op_bench_c2.generate_c2_test(
    batch_gather_configs_long + batch_gather_configs_short, BatchGatherBenchmark
)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
