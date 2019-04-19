from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from benchmarks.operator_benchmark import benchmark_core, benchmark_runner
from benchmarks.operator_benchmark.benchmark_test_generator import *

import torch


"""Microbenchmarks for element-wise Add operator. Supports both Caffe2/PyTorch."""

# Input shapes that we test and the run mode for each shape.
# Sum up two tensors with the same shape

# Long config
long_config = generate_configs(
    M=get_n_rand_nums(min_val=1, max_val=128, n=2),
    N=get_n_rand_nums(min_val=1, max_val=128, n=2),
    K=get_n_rand_nums(min_val=1, max_val=128, n=2),
    mode=['long'],
    sample_func=cross_product,
)

# Short config
short_config = generate_configs(
    M=[8, 16],
    N=[32, 64],
    K=[64, 128],
    mode=['short'],
    sample_func=cross_product
)


@torch.jit.script
def torch_add(a, b, iterations):
    # type: (Tensor, Tensor, int)
    result = torch.jit.annotate(torch.Tensor, None)
    for _ in range(iterations):
        result = torch.add(a, b)
    return result


@benchmark_core.register_test
def test_add():
    generate_pt_test(
        [long_config, short_config],
        map_pt_config_add,
        [('add', torch_add)]
    )
    generate_c2_test(
        [long_config, short_config],
        map_c2_config_add,
        [('add', 'Add')],
    )


if __name__ == "__main__":
    benchmark_runner.main()
