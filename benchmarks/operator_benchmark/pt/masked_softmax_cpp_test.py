
import operator_benchmark as op_bench
import torch


"""
Microbenchmarks for the masked_softmax operators.
"""


# Configs for masked_softmax ops
masked_softmax_configs_short = op_bench.config_list(
    attr_names=[
        'N', 'C', 'H', 'W'
    ],
    attrs=[
        [1, 3, 256, 256],
        [4, 3, 256, 256],
    ],
    cross_product_configs={
        'device': ['cpu'],
        'sparsity': [0., 0.01, 0.05, 0.075, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.92, 0.94, 0.96, 0.98, 0.99, 1],
    },
    tags=['short'],
)


masked_softmax_configs_long = op_bench.cross_product_configs(
    N=[8, 16],
    C=[3],
    H=[256, 512],
    W=[256, 512],
    device=['cpu'],
    tags=['long'],
    sparsity=[0., 0.01, 0.05, 0.075, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.92, 0.94, 0.96, 0.98, 0.99, 1],
)


masked_softmax_ops_list = op_bench.op_list(
    attr_names=['op_name', 'op_func'],
    attrs=[
        ['masked_softmax', lambda x, mask: x.masked_softmax(mask, dim=-1)],
    ],
)


class MaskedSoftmaxCppBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, N, C, H, W, device, op_func, sparsity):
        self.input_one = torch.rand(N, C, H, W, device=device)
        self.input_two = torch.multinomial(
            torch.Tensor([sparsity, 1. - sparsity]),
            num_samples=self.input_one.numel(),
            replacement=True,
        ).view(self.input_one.shape).to(torch.bool)
        self.op_func = op_func
        self.set_module_name("masked_softmax_cpp")

    def forward(self):
        return self.op_func(self.input_one, self.input_two)


op_bench.generate_pt_tests_from_op_list(masked_softmax_ops_list,
                                        masked_softmax_configs_short + masked_softmax_configs_long,
                                        MaskedSoftmaxCppBenchmark)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
