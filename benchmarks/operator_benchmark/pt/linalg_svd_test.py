import operator_benchmark as op_bench

import torch


"""Microbenchmarks for torch.linalg.svd and torch.linalg.svdvals."""


def linalg_svd(input):
    return torch.linalg.svd(input, full_matrices=False)


ops_list = op_bench.op_list(
    attr_names=["op_name", "op_func"],
    attrs=[
        ["linalg_svd", linalg_svd],
        ["linalg_svdvals", torch.linalg.svdvals],
    ],
)

linalg_svd_short_configs = op_bench.config_list(
    attr_names=["M", "N", "batch"],
    attrs=[
        [2, 3, 1],
        [64, 32, 1],
        [128, 64, 1],
        [64, 128, 1],
        [96, 48, 2],
        [128, 128, 1],
    ],
    cross_product_configs={"device": ["cpu", "mps"], "dtype": [torch.float32]},
    tags=["short"],
)


class LinalgSvdBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, M, N, batch, device, dtype, op_func):
        shape = (M, N) if batch == 1 else (batch, M, N)
        self.inputs = {
            "input": torch.randn(
                *shape, device=device, dtype=dtype, requires_grad=self.auto_set()
            ),
        }
        self.op_func = op_func

    def forward(self, input):
        return self.op_func(input)


op_bench.generate_pt_tests_from_op_list(
    ops_list, linalg_svd_short_configs, LinalgSvdBenchmark
)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
