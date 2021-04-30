import benchmark_caffe2 as op_bench_c2
import operator_benchmark as op_bench
from benchmark_caffe2 import Caffe2BenchmarkBase  # noqa: F401
from caffe2.python import core


"""Microbenchmarks for QuantileOp operator."""

# Configs for C2 QuantileOp operator
quantile_op_long_configs = op_bench.cross_product_configs(
    M=[32, 64, 128], N=range(32, 128, 32), dtype=["float", "double"], tags=["long"]
)


quantile_op_short_configs = op_bench.config_list(
    attrs=[
        [16, 16, "float"],
        [16, 16, "double"],
        [64, 64, "float"],
        [64, 64, "double"],
    ],
    attr_names=["M", "N", "dtype"],
    tags=["short"],
)


class QuantileOpBenchmark(op_bench_c2.Caffe2BenchmarkBase):
    def init(self, M, N, dtype):
        self.data = [self.tensor([N], dtype) for _ in range(M)]
        self.quantile = 0.3
        self.output = self.tensor([1], dtype)
        self.set_module_name("quantile_op")

    def forward(self):
        op = core.CreateOperator(
            "Quantile", inputs=self.data, outputs=self.output, quantile=self.quantile
        )
        return op


op_bench_c2.generate_c2_test(
    quantile_op_long_configs + quantile_op_short_configs, QuantileOpBenchmark
)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
