from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import operator_benchmark as op_bench
from pt import ( # noqa
    unary_test,  # noqa
)
import benchmark_all_other_test  # noqa
import benchmark_all_quantized_test  # noqa

if __name__ == "__main__":
    op_bench.benchmark_runner.main()
