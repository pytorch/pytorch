from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import operator_benchmark as op_bench
from pt import ( # noqa
    add_test, batchnorm_test, cat_test, chunk_test, conv_test,  # noqa
    gather_test, linear_test, matmul_test, pool_test,  # noqa
    softmax_test, split_test, unary_test, fill_test, as_strided_test,  # noqa
)

# Quantized tests
from benchmark_all_quantized_test import *  # noqa


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
