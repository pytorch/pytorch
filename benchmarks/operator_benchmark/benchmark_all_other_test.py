from pt import (  # noqa: F401  # noqa: F401
    add_test,
    ao_sparsifier_test,
    as_strided_test,
    batchnorm_test,
    binary_test,
    cat_test,
    channel_shuffle_test,
    chunk_test,
    conv_test,
    diag_test,
    embeddingbag_test,
    fill_test,
    gather_test,
    groupnorm_test,
    hardsigmoid_test,
    hardswish_test,
    instancenorm_test,
    interpolate_test,
    layernorm_test,
    linear_test,
    matmul_test,
    nan_to_num_test,
    pool_test,
    remainder_test,
    softmax_test,
    split_test,
    sum_test,
    tensor_to_test,
)

import operator_benchmark as op_bench

if __name__ == "__main__":
    op_bench.benchmark_runner.main()
