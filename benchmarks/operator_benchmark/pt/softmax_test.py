from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import operator_benchmark as op_bench
import torch
import torch.nn as nn


"""
Microbenchmarks for the softmax operators.
"""


# Configs for softmax ops
softmax_configs_short = op_bench.config_list(
    attrs=[
        [4, 3, 128, 128],
        [8, 3, 256, 256],
    ],
    attr_names=[
        'N', 'C', 'H', 'W'
    ],
    tags=['short']
)

softmax_configs_long = op_bench.config_list(
    attrs=[
        [8, 3, 128, 128],
        [16, 512, 14, 14],
        [16, 256, 28, 28],
    ],
    attr_names=[
        'N', 'C', 'H', 'W'
    ],
    tags=['long']
)

softmax_ops_list = op_bench.op_list(
    attr_names=['op_name', 'op_func'],
    attrs=[
        ['Softmax', nn.Softmax],
        ['Softmax2d', nn.Softmax2d],
        ['LogSoftmax', nn.LogSoftmax],
    ],
)


class SoftmaxBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, N, C, H, W, op_func):
        self.input_one = torch.rand(N, C, H, W)
        self.op_func = op_func()

    def forward(self):
        return self.op_func(self.input_one)


op_bench.generate_pt_tests_from_op_list(softmax_ops_list,
                                        softmax_configs_short + softmax_configs_long,
                                        SoftmaxBenchmark)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
