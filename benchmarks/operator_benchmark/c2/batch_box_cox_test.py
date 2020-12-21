import benchmark_caffe2 as op_bench_c2
import operator_benchmark as op_bench
from benchmark_caffe2 import Caffe2BenchmarkBase  # noqa
from caffe2.python import core


"""Microbenchmarks for BatchBoxCox operator."""

# Configs for C2 BatchBoxCox operator
batch_box_cox_long_configs = op_bench.cross_product_configs(
    M=[32, 64, 128], N=range(32, 128, 32), dtype=["float", "double"], tags=["long"]
)


batch_box_cox_short_configs = op_bench.config_list(
    attrs=[
        [16, 16, "float"],
        [16, 16, "double"],
        [64, 64, "float"],
        [64, 64, "double"],
    ],
    attr_names=["M", "N", "dtype"],
    tags=["short"],
)


class BatchBoxCoxBenchmark(op_bench_c2.Caffe2BenchmarkBase):
    def init(self, M, N, dtype):
        self.data = self.tensor([M, N], dtype)
        self.lambda1 = self.tensor([N], dtype)
        self.lambda2 = self.tensor([N], dtype)
        self.output = self.tensor([1, 1], dtype)
        self.set_module_name("batch_box_cox")

    def forward(self):
        op = core.CreateOperator("BatchBoxCox", [self.data, self.lambda1, self.lambda2], self.output)
        return op


op_bench_c2.generate_c2_test(
    batch_box_cox_long_configs + batch_box_cox_short_configs, BatchBoxCoxBenchmark
)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
