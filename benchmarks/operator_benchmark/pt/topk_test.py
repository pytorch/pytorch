import operator_benchmark as op_bench
import torch


"""Microbenchmarks for topk operator"""


topk_configs_short = op_bench.config_list(
    attr_names=["shape", "k", "dim"],
    attrs=[
        [(16, 4), 4, 1],
        [(1024 * 1024,), 16, 0],
    ],
    cross_product_configs={"device": ["cpu"], "dtype": [torch.float]},
    tags=["short"],
)

topk_configs_long = op_bench.cross_product_configs(
    shape=[(64, 2), (1024 * 1024,), (128,)],
    k=[1, 2, 4, 16, 32],
    dim=[0],
    device=["cpu", "cuda"],
    dtype=[torch.float, torch.bfloat16],
    tags=["long"],
)


class TopkBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, shape, k, dim, dtype, device):
        self.inputs = {
            "input": torch.randn(shape, device=device, dtype=dtype),
            "k": k,
            "dim": dim,
        }

        self.set_module_name("topk")

    def forward(self, input, k, dim):
        return torch.topk(input, k=k, dim=dim)


op_bench.generate_pt_test(topk_configs_short + topk_configs_long, TopkBenchmark)

if __name__ == "__main__":
    op_bench.benchmark_runner.main()
