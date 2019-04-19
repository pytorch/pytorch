from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from benchmarks.operator_benchmark import benchmark_core, benchmark_runner
from benchmarks.operator_benchmark.benchmark_test_generator import *

import torch


"""Microbenchmarks for MatMul operator. Supports both Caffe2/PyTorch."""
# Long config
long_config = generate_configs(
    M=get_n_rand_nums(min_val=1, max_val=128, n=2),
    N=get_n_rand_nums(min_val=1, max_val=128, n=2),
    K=get_n_rand_nums(min_val=1, max_val=128, n=2),
    trans_a=[False, True],
    trans_b=[True, False],
    mode=['long'],
    sample_func=cross_product
)

# Short config
short_config = generate_configs(
    M=[8, 16],
    N=[32, 64],
    K=[64, 128],
    trans_a=[False, True],
    trans_b=[True, False],
    mode=['short'],
    sample_func=cross_product
)


@torch.jit.script
def torch_matmul(a, b, iterations):
    # type: (Tensor, Tensor, int)
    result = torch.jit.annotate(torch.Tensor, None)
    for _ in range(iterations):
        result = torch.matmul(a, b)
    return result


@benchmark_core.register_test
def test_matmul():
    generate_pt_test(
        [long_config, short_config],
        map_pt_config_matmul,
        [('matmul', torch_matmul)]
    )
    generate_c2_test(
        [long_config, short_config],
        map_c2_config_matmul,
        [('matmul', 'MatMul')],
    )


if __name__ == "__main__":
    benchmark_runner.main()
