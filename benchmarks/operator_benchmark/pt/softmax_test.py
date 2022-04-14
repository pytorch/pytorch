
import operator_benchmark as op_bench
import torch
import torch.nn as nn


"""
Microbenchmarks for the softmax operators.
"""


# Configs for softmax ops
softmax_configs_short = op_bench.config_list(
    attr_names=[
        'N', 'C', 'H', 'W'
    ],
    attrs=[
        [1, 3, 256, 256],
        [4, 3, 256, 256],
    ],
    cross_product_configs={
        'device': ['cpu', 'cuda'],
    },
    tags=['short']
)


softmax_configs_long = op_bench.cross_product_configs(
    N=[8, 16],
    C=[3],
    H=[256, 512],
    W=[256, 512],
    device=['cpu', 'cuda'],
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
    def init(self, N, C, H, W, device, op_func):
        self.inputs = {
            "input": torch.rand(N, C, H, W, device=device)
        }
        self.op_func = op_func()

    def forward(self, input):
        return self.op_func(input)


op_bench.generate_pt_tests_from_op_list(softmax_ops_list,
                                        softmax_configs_short + softmax_configs_long,
                                        SoftmaxBenchmark)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
