from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import operator_benchmark as op_bench
import benchmark_caffe2 as op_bench_c2
from benchmark_caffe2 import Caffe2BenchmarkBase # noqa
from caffe2.python import core 

"""Microbenchmarks for MatMul operator"""

# Configs for C2 Matmul operator
mm_long_configs = op_bench.cross_product_configs(
    M=[8, 64, 128],
    N=range(2, 10, 3),
    K=[2 ** x for x in range(0, 3)], 
    trans_a=[True, False],
    trans_b=[True, False],
    tags=["long"]
)


mm_short_configs = op_bench.config_list(
    attrs=[
        [128, 128, 128, False, True],
        [1024, 1024, 256, True, False],
        [8192, 8192, 1024, True, False],
    ],
    attr_names=["M", "N", "K", "trans_a", "trans_b"], 
    tags=["short"], 
)


class MatMulBenchmark(op_bench_c2.Caffe2BenchmarkBase):
    def init(self, M, N, K, trans_a, trans_b): 
        self.input_one = self.tensor([N, M]) if trans_a else self.tensor([M, N])
        self.input_two = self.tensor([K, N]) if trans_b else self.tensor([N, K])
        self.args = {'trans_a': trans_a, 'trans_b': trans_b}
        self.output = self.tensor([M, K])
        self.set_module_name("matmul")

    def forward(self):
        op = core.CreateOperator(
            "MatMul", [self.input_one, self.input_two], self.output, **self.args 
        )
        return op


op_bench_c2.generate_c2_test(mm_long_configs + mm_short_configs, MatMulBenchmark)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
