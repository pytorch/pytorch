import operator_benchmark as op_bench

"""
Configs shared by multiple benchmarks
"""

# Configs for conv-1d ops
conv_1d_configs_short = op_bench.config_list(
    attr_names=[
        'IC', 'OC', 'kernel', 'stride', 'N', 'L'
    ],
    attrs=[
        [128, 256, 3, 1, 1, 64],
        [256, 256, 3, 2, 4, 64],
    ],
    cross_product_configs={
        'device': ['cpu', 'cuda'],
    },
    tags=['short']
)

conv_1d_configs_long = op_bench.cross_product_configs(
    IC=[128, 512],
    OC=[128, 512],
    kernel=[3],
    stride=[1, 2],
    N=[8],
    L=[128],
    device=['cpu', 'cuda'],
    tags=["long"]
)

# Configs for Conv2d and ConvTranspose1d
conv_2d_configs_short = op_bench.config_list(
    attr_names=[
        'IC', 'OC', 'kernel', 'stride', 'N', 'H', 'W', 'G', 'pad',
    ],
    attrs=[
        [256, 256, 3, 1, 1, 16, 16, 1, 0],
    ],
    cross_product_configs={
        'device': ['cpu', 'cuda'],
    },
    tags=['short']
)

conv_2d_configs_long = op_bench.cross_product_configs(
    IC=[128, 256],
    OC=[128, 256],
    kernel=[3],
    stride=[1, 2],
    N=[4],
    H=[32],
    W=[32],
    G=[1],
    pad=[0],
    device=['cpu', 'cuda'],
    tags=["long"]
)

# Configs for Conv3d and ConvTranspose3d
conv_3d_configs_short = op_bench.config_list(
    attr_names=[
        'IC', 'OC', 'kernel', 'stride', 'N', 'D', 'H', 'W'
    ],
    attrs=[
        [64, 64, 3, 1, 8, 4, 16, 16],
    ],
    cross_product_configs={
        'device': ['cpu', 'cuda'],
    },
    tags=['short']
)
