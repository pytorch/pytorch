import operator_benchmark as op_bench

import torch
import torch.optim as optim


"""Microbenchmarks for the Muon optimizer."""


muon_impls = op_bench.op_list(
    attr_names=["op_name", "op_func"],
    attrs=[
        ["muon_forloop", False],
        ["muon_foreach", True],
    ],
)

muon_configs = op_bench.config_list(
    attr_names=["case_name", "num_params", "rows", "cols"],
    attrs=[
        ["one_rect_512x1024", 1, 512, 1024],
        ["one_rect_512x1536", 1, 512, 1536],
        ["one_rect_512x2048", 1, 512, 2048],
        ["four_rect_512x1024", 4, 512, 1024],
        ["four_rect_512x1536", 4, 512, 1536],
        ["four_rect_512x2048", 4, 512, 2048],
        ["sixteen_rect_512x1024", 16, 512, 1024],
        ["sixteen_rect_512x1536", 16, 512, 1536],
        ["sixteen_rect_512x2048", 16, 512, 2048],
        ["four_square_1024", 4, 1024, 1024],
        ["sixteen_square_1024", 16, 1024, 1024],
    ],
    cross_product_configs={
        "device": ["cuda"],
    },
    tags=["long"],
)


class MuonOptimizerBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, op_func, device, case_name, num_params, rows, cols):
        self.params = [
            torch.nn.Parameter(
                torch.randn(rows, cols, device=device, dtype=torch.float32)
            )
            for _ in range(num_params)
        ]
        for p in self.params:
            p.grad = torch.randn_like(p)

        self.optimizer = optim.Muon(self.params, lr=0.001, foreach=op_func)
        # Initialize momentum buffers before benchmarking steady-state step time.
        self.optimizer.step()
        for p in self.params:
            p.grad = torch.randn_like(p)

        self.inputs = {"dummy": self.params[0]}

    def forward(self, dummy):
        self.optimizer.step()
        return dummy

    def get_memory_traffic_bytes(self):
        return None


op_bench.generate_pt_tests_from_op_list(
    muon_impls, muon_configs, MuonOptimizerBenchmark
)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
