import operator_benchmark as op_bench
import torch


"""Microbenchmarks for diag operator"""


# Configs for PT diag operator
diag_configs_short = op_bench.config_list(
    attr_names=['dim', 'M', 'N', 'diagonal', 'out'],
    attrs=[
        [1, 64, 64, 0, True],
        [2, 128, 128, -10, False],
        [1, 256, 256, 20, True],
    ],
    cross_product_configs={
        'device': ['cpu', 'cuda'],
    },
    tags=['short'],
)


class DiagBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, dim, M, N, diagonal, out, device):
        self.inputs = {
            "input": torch.rand(M, N, device=device) if dim == 2 else torch.rand(M, device=device),
            "diagonal": diagonal,
            "out": out,
            "out_tensor": torch.tensor((),)
        }
        self.set_module_name('diag')

    def forward(self, input, diagonal: int, out: bool, out_tensor):
        if out:
            return torch.diag(input, diagonal=diagonal, out=out_tensor)
        else:
            return torch.diag(input, diagonal=diagonal)


op_bench.generate_pt_test(diag_configs_short, DiagBenchmark)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
