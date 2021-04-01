import operator_benchmark as op_bench
from pt import ( # noqa
    add_test, as_strided_test, batchnorm_test, binary_test, cat_test,  # noqa
    channel_shuffle_test, chunk_test, conv_test, diag_test, embeddingbag_test,  # noqa
    fill_test, gather_test, linear_test, matmul_test, nan_to_num_test, pool_test,  # noqa
    softmax_test, hardsigmoid_test, hardswish_test, layernorm_test,  # noqa
    groupnorm_test, interpolate_test, instancenorm_test, remainder_test, softmax_test,  # noqa
    split_test, sum_test, tensor_to_test  # noqa
)

if __name__ == "__main__":
    op_bench.benchmark_runner.main()
