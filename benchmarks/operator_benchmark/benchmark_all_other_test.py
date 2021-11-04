import operator_benchmark as op_bench
from pt import (  # noqa: F401
    add_test, as_strided_test, batchnorm_test, binary_test, cat_test,
    channel_shuffle_test, chunk_test, conv_test, diag_test, embeddingbag_test,
    fill_test, gather_test, linear_test, matmul_test, nan_to_num_test, pool_test,
    softmax_test, hardsigmoid_test, hardswish_test, layernorm_test,
    groupnorm_test, interpolate_test, instancenorm_test, remainder_test,
    split_test, sum_test, tensor_to_test
)

if __name__ == "__main__":
    op_bench.benchmark_runner.main()
