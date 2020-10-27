import operator_benchmark as op_bench
import torch
import math


"""Microbenchmarks for torch.nan_to_num / nan_to_num_ operators"""

# Configs for PT torch.nan_to_num / nan_to_num_ operators
nan_to_num_long_configs = op_bench.cross_product_configs(
    M=[32, 64, 128],
    N=range(32, 128, 32),
    dtype=[torch.float, torch.double],
    op=["nan_to_num", "nan_to_num_"],
    replace_inf=[True, False],
    tags=["long"],
)


nan_to_num_short_configs = op_bench.cross_product_configs(
    M=[16, 64],
    N=[64, 64],
    dtype=[torch.float, torch.double],
    op=["nan_to_num", "nan_to_num_"],
    replace_inf=[True, False],
    tags=["short"],
)


class ReplaceNaNBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, M, N, dtype, op, replace_inf):
        self.input = torch.randn(M, N, dtype=dtype)
        self.input[0][0] = float("nan")
        self.op = op
        self.replace_inf = replace_inf
        self.set_module_name("nan_to_num")

    def forward(self):
        # compare inplace
        if self.op == "nan_to_num":
            if self.replace_inf:
                output = torch.nan_to_num(self.input, nan=1.0)
            else:
                output = torch.nan_to_num(self.input, nan=1.0, posinf=math.inf, neginf=-math.inf)
        else:
            if self.replace_inf:
                output = torch.nan_to_num_(self.input, nan=1.0)
            else:
                output = torch.nan_to_num_(self.input, nan=1.0, posinf=math.inf, neginf=-math.inf)
        return output


op_bench.generate_pt_test(
    nan_to_num_long_configs + nan_to_num_short_configs,
    ReplaceNaNBenchmark,
)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
