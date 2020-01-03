from __future__ import absolute_import, division, print_function, unicode_literals
import operator_benchmark as op_bench
import torch


configs = op_bench.random_sample_configs(
    M=[1, 2, 3, 4, 5, 6],
    N=[7, 8, 9, 10, 11, 12],
    K=[13, 14, 15, 16, 17, 18],
    # probs saves the weights of each value
    probs=op_bench.attr_probs( 
        M=[0.5, 0.2, 0.1, 0.05, 0.03, 0.1],
        N=[0.1, 0.3, 0.4, 0.02, 0.03, 0.04],
        K=[0.03, 0.6, 0.04, 0.02, 0.03, 0.01],
    ),
    # this is the number of returned inputs 
    total_samples=10, 
    tags=["short"],
)


class AddBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, M, N, K): 
        self.input_one = torch.rand(M, N, K)
        self.input_two = torch.rand(M, N, K)
        self.set_module_name("add")

    def forward(self):
        return torch.add(self.input_one, self.input_two)


op_bench.generate_pt_test(configs, AddBenchmark)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
