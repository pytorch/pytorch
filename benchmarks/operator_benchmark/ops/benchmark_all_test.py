from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from operator_benchmark import benchmark_runner
from operator_benchmark.ops import ( # noqa
    add_test, # noqa
    matmul_test) # noqa


if __name__ == "__main__":
    benchmark_runner.main()
