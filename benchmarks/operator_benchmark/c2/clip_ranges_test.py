import benchmark_caffe2 as op_bench_c2
import operator_benchmark as op_bench
from benchmark_caffe2 import Caffe2BenchmarkBase  # noqa
from caffe2.python import core, dyndep

dyndep.InitOpsLibrary("@/caffe2/caffe2/fb/operators:clip_ranges_op")

"""Microbenchmarks for ClipRanges operator."""

# Configs for C2 ClipRanges operator
clip_ranges_long_configs = op_bench.cross_product_configs(
    LENGTH=range(1, 100),
    M=[1],
    N=[2],
    MAX_LENGTH=range(1, 100),
    dtype=["int32"],
    tags=["long"]
)


clip_ranges_short_configs = op_bench.config_list(
    attrs=[
        [6, 1, 2, 1, "int32"],
        [7, 1, 2, 2, "int32"],
        [8, 1, 2, 3, "int32"],
        [9, 1, 2, 4, "int32"],
        [10, 1, 2, 5, "int32"],
    ],
    attr_names=["LENGTH", "M", "N", "MAX_LENGTH", "dtype"],
    tags=["short"],
)


class ClipRangesBenchmark(op_bench_c2.Caffe2BenchmarkBase):
    def init(self, LENGTH, M, N, MAX_LENGTH, dtype):
        self.input = self.tensor([LENGTH, M, N], dtype)
        self.max_length = MAX_LENGTH
        self.set_module_name("clip_ranges")

    def forward(self):
        op = core.CreateOperator("ClipRanges", self.input, self.input, max_length=self.max_length)
        return op


op_bench_c2.generate_c2_test(
    clip_ranges_long_configs + clip_ranges_short_configs, ClipRangesBenchmark
)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
