from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import operator_benchmark as op_bench
from ops.pt import ( # noqa
    add_test, batchnorm_test, conv_test, linear_test, matmul_test # noqa
)
from ops.c2 import ( # noqa
    add_test, matmul_test # noqa
)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
