import benchmark_all_other_test  # noqa: F401
import benchmark_all_quantized_test  # noqa: F401
from pt import unary_test  # noqa: F401

import operator_benchmark as op_bench


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
