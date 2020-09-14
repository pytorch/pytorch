from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import operator_benchmark as op_bench
from pt import ( # noqa
    add_test, as_strided_test, batchnorm_test, binary_test, cat_test,  # noqa
    channel_shuffle_test, chunk_test, conv_test, diag_test, embeddingbag_test,  # noqa
    fill_test, gather_test, linear_test, matmul_test, pool_test,  # noqa
    softmax_test, hardsigmoid_test, hardswish_test, layernorm_test,  # noqa
    groupnorm_test, instancenorm_test # noqa
)

if __name__ == "__main__":
    op_bench.benchmark_runner.main()
