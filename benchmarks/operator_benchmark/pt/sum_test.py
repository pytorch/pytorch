import operator_benchmark as op_bench
import torch

"""Microbenchmarks for sum reduction operator."""

# Configs for PT add operator
sum_configs = op_bench.cross_product_configs(
    R=[64, 256],  # Length of reduced dimension
    V=[32, 512],  # Length of other dimension
    dim=[0, 1],
    contiguous=[True, False],
    device=["cpu", "cuda"],
    tags=["short"],
) + op_bench.cross_product_configs(
    R=[1024, 8192],
    V=[512, 1024],
    dim=[0, 1],
    contiguous=[True, False],
    device=["cpu", "cuda"],
    tags=["long"],
)


class SumBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, R, V, dim, contiguous, device):
        shape = (R, V) if dim == 0 else (V, R)
        tensor = torch.rand(shape, device=device)

        if not contiguous:
            storage = torch.empty([s * 2 for s in shape], device=device)
            storage[::2, ::2] = tensor
            self.input_tensor = storage[::2, ::2]
        else:
            self.input_tensor = tensor

        self.inputs = {"input_tensor": self.input_tensor, "dim": dim}
        self.set_module_name("sum")

    def forward(self, input_tensor, dim: int):
        return input_tensor.sum(dim=dim)


op_bench.generate_pt_test(sum_configs, SumBenchmark)

if __name__ == "__main__":
    op_bench.benchmark_runner.main()
