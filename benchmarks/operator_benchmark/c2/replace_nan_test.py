import benchmark_caffe2 as op_bench_c2
import operator_benchmark as op_bench
from benchmark_caffe2 import Caffe2BenchmarkBase  # noqa: F401
from caffe2.python import core


"""Microbenchmarks for element-wise ReplaceNaN operator."""

# Configs for C2 ReplaceNaN operator
replace_nan_long_configs = op_bench.cross_product_configs(
    M=[32, 64, 128], N=range(32, 128, 32), dtype=["float", "double"], tags=["long"]
)


replace_nan_short_configs = op_bench.config_list(
    attrs=[
        [16, 16, "float"],
        [16, 16, "double"],
        [64, 64, "float"],
        [64, 64, "double"],
    ],
    attr_names=["M", "N", "dtype"],
    tags=["short"],
)


class ReplaceNaNBenchmark(op_bench_c2.Caffe2BenchmarkBase):
    def init(self, M, N, dtype):
        self.input = self.tensor([M, N], dtype)
        self.set_module_name("replace_nan")

    def forward(self):
        op = core.CreateOperator("ReplaceNaN", self.input, self.input, value=1.0)
        return op


op_bench_c2.generate_c2_test(
    replace_nan_long_configs + replace_nan_short_configs, ReplaceNaNBenchmark
)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
