import benchmark_fuzz_utils as fuzz_utils
import operator_benchmark as op_bench

"""
Configs shared by multiple benchmarks
"""

def remove_cuda(config_list):
    cuda_config = {'device': 'cuda'}
    return [config for config in config_list if cuda_config not in config]


conv1d_fuzzed_configs_short = fuzz_utils.make_fuzzed_config(
    fuzz_utils.Fuzzers.CONV1D,
    fuzz_utils.Scale.SMALL,
    n=10,
    seed="Conv1D",
    cross_product_configs={"device": ["cpu", "cuda"]},
    tags=["short"],
    checksum=570,
)

conv1d_fuzzed_configs_long = fuzz_utils.make_fuzzed_config(
    fuzz_utils.Fuzzers.CONV1D,
    fuzz_utils.CPU_MEDIUM_CUDA_LARGER,
    n=10,
    seed="Conv1D",
    tags=["long"],
    checksum=(2341, 9275),
)

for i in conv1d_fuzzed_configs_long:
    print(i)

conv2d_fuzzed_configs_short = fuzz_utils.make_fuzzed_config(
    fuzz_utils.Fuzzers.CONV2D,
    fuzz_utils.Scale.SMALL,
    n=10,
    fuzzer_kwargs={"groups": {1: 0.5, 2: 0.5}},
    seed="Conv2D",
    cross_product_configs={"device": ["cpu", "cuda"]},
    tags=["short"],
    checksum=517,
)

conv2d_fuzzed_configs_long = fuzz_utils.make_fuzzed_config(
    fuzz_utils.Fuzzers.CONV2D,
    fuzz_utils.CPU_MEDIUM_CUDA_LARGER,
    n=10,
    fuzzer_kwargs={"groups": {1: 0.5, 2: 0.5}},
    seed="Conv2D",
    tags=["long"],
    checksum=(1803, 6358),
)


# Configs for Batch/Instance/Layer Norm.
norm_fuzzed_configs_short = fuzz_utils.make_fuzzed_config(
    fuzz_utils.Fuzzers.UNARY,
    fuzz_utils.Scale.SMALL,
    n=10,
    seed="Norm",
    fuzzer_kwargs={"dim": {3: 0.5, 4: 0.5}, "pow_2_fraction": 0.8},
    cross_product_configs={
        'device': ['cpu', 'cuda'],
    },
    tags=["short"],
    checksum=648,
)

norm_fuzzed_configs_long = fuzz_utils.make_fuzzed_config(
    fuzz_utils.Fuzzers.UNARY,
    fuzz_utils.CPU_MEDIUM_CUDA_LARGER,
    n=10,
    seed="Norm",
    fuzzer_kwargs={"dim": {3: 0.5, 4: 0.5}, "pow_2_fraction": 0.8},
    tags=["long"],
    checksum=(4680, 82871),
)

norm_fuzzed_configs = norm_fuzzed_configs_short + norm_fuzzed_configs_long


linear_configs_short = op_bench.config_list(
    attr_names=["N", "IN", "OUT"],
    attrs=[
        [1, 1, 1],
        [4, 256, 128],
        [16, 512, 256],
    ],
    cross_product_configs={
        'device': ['cpu', 'cuda'],
    },
    tags=["short"]
)


linear_configs_long = op_bench.cross_product_configs(
    N=[32, 64],
    IN=[128, 512],
    OUT=[64, 128],
    device=['cpu', 'cuda'],
    tags=["long"]
)

embeddingbag_short_configs = op_bench.cross_product_configs(
    embeddingbags=[10, 120, 1000, 2300],
    dim=[64],
    mode=['sum'],
    input_size=[8, 16, 64],
    offset=[0],
    sparse=[True],
    include_last_offset=[True, False],
    device=['cpu'],
    tags=['short']
)
