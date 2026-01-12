import operator_benchmark as op_bench
import torch


"""Microbenchmarks for where operator."""


configs_short = op_bench.config_list(
    attr_names=["cond_shape", "input_shape", "other_shape"],
    attrs=[
        [(8, 16, 1), (1,), (1,)],
        [(8, 16, 1), (16, 1), (8, 16, 1)],
        [(8, 16, 1), (8, 1, 1), (1,)],
    ],
    cross_product_configs={"device": ["cpu"], "dtype": [torch.float]},
    tags=["short"],
)


configs_long = op_bench.cross_product_configs(
    cond_shape=[(64, 16, 1), (64, 16, 8), (1024, 64, 16, 128)],
    input_shape=[(1,), (16, 1), (64, 16, 1)],
    other_shape=[(1,), (16, 1), (64, 16, 1)],
    device=["cpu", "cuda"],
    dtype=[torch.float],
    tags=["long"],
)


class WhereBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, cond_shape, input_shape, other_shape, dtype, device):
        def _create_tensor(shape):
            return torch.randn(*shape, dtype=dtype, device=device)

        self.inputs = {
            "condition": _create_tensor(cond_shape) > 0,
            "input": _create_tensor(input_shape),
            "other": _create_tensor(other_shape),
        }
        self.set_module_name("where")

    def forward(self, condition, input, other):
        return torch.where(condition, input, other)


op_bench.generate_pt_test(configs_short + configs_long, WhereBenchmark)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
