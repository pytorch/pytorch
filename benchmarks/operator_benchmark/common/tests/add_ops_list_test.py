import operator_benchmark as op_bench
import torch


# Configs for pointwise unary ops
unary_ops_configs = op_bench.config_list(
    attrs=[
        [128, 128],
    ],
    attr_names=["M", "N"],
    tags=["short"]
)


unary_ops_list = op_bench.op_list(
    attr_names=["op_name", "op_func"],
    attrs=[
        ["abs", torch.abs],
        ["acos", torch.acos],
    ],
)


class UnaryOpBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, M, N, op_func):
        self.input_one = torch.rand(M, N)
        self.op_func = op_func

    def forward(self):
        return self.op_func(self.input_one)


op_bench.generate_pt_tests_from_op_list(unary_ops_list, unary_ops_configs, UnaryOpBenchmark)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
