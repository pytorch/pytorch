import operator_benchmark as op_bench
from pt import (  # noqa: F401
    unary_test,
)
import benchmark_all_other_test  # noqa: F401
import benchmark_all_quantized_test  # noqa: F401

if __name__ == "__main__":
    op_bench.benchmark_runner.main()
