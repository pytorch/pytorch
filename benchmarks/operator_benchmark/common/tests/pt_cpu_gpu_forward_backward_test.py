import operator_benchmark as op_bench

import torch


add_configs = op_bench.cross_product_configs(
    M=[8], N=[8], K=[8], device=["cuda", "cpu"], tags=["short"]
)


class AddBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, M, N, K, device):
        self.input_one = torch.rand(M, N, K, device=device, requires_grad=True)
        self.input_two = torch.rand(M, N, K, device=device, requires_grad=True)
        self.set_module_name("add")

    def forward(self):
        return torch.add(self.input_one, self.input_two)


op_bench.generate_pt_test(add_configs, AddBenchmark)
op_bench.generate_pt_gradient_test(add_configs, AddBenchmark)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
