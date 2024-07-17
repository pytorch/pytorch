import operator_benchmark as op_bench
import torch

"""Microbenchmarks for add_ operator. Supports both Caffe2/PyTorch."""


class BmmBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, B, M, N, K, device, op):
        self.inputs = {
            "batch1": torch.rand(
                (B, M, K), device=device, requires_grad=self.auto_set()
            ),
            "batch2": torch.rand(
                (
                    B,
                    K,
                    N,
                ),
                device=device,
                requires_grad=self.auto_set(),
            ),
        }
        self.set_module_name(f"bmm (actual op={op}")
        self.op = torch.bmm if op == "bmm" else torch.matmul

    def forward(self, batch1, batch2):
        return self.op(batch1, batch2)


bmm_configs = op_bench.cross_product_configs(
    B=[2, 100],
    M=[8, 256],
    N=[256, 16],
    K=[16, 32],
    device=["cpu"],
    tags=["short"],
    op=["bmm", "matmul"],
)

op_bench.generate_pt_test(bmm_configs, BmmBenchmark)

if __name__ == "__main__":
    op_bench.benchmark_runner.main()
