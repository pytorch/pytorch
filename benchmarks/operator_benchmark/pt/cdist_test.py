import operator_benchmark as op_bench

import torch


"""Microbenchmarks for torch.cdist (pairwise distances)."""


cdist_configs = op_bench.cross_product_configs(
    B=[1, 8],
    P=[64, 512],
    R=[64, 512],
    M=[16, 64],
    p=[1.0, 2.0, 3.0, 4.0],
    device=["cpu", "cuda", "mps"],
    tags=["long"],
)


class CdistBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, device, B, P, R, M, p):
        self.inputs = {
            "x1": torch.rand(B, P, M, device=device, requires_grad=self.auto_set()),
            "x2": torch.rand(B, R, M, device=device, requires_grad=self.auto_set()),
            "p": p,
        }
        self.set_module_name("cdist")

    def forward(self, x1, x2, p):
        return torch.cdist(x1, x2, p=p)


op_bench.generate_pt_test(
    cdist_configs,
    CdistBenchmark,
)

op_bench.generate_pt_gradient_test(
    cdist_configs,
    CdistBenchmark,
)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
