from . import benchmark
import numpy as np


class MatMulBench(benchmark.Benchmark):
    def __init__(self, mode, device, dtype, B, M, N, K):
        super().__init__(mode, device, dtype)
        self.B = B
        self.M = M
        self.N = N
        self.K = K
        self.d1 = self.rand([B, M, N], device=device, dtype=dtype, requires_grad=self.requires_grad)
        self.d2 = self.rand([B, N, K], device=device, dtype=dtype, requires_grad=self.requires_grad)
        self.inputs = [self.d1, self.d2]

    def forward(self, d1, d2):
        y = self.matmul(d1, d2)
        return y

    def reference(self):
        return np.matmul(self.numpy(self.d1), self.numpy(self.d2))

    def config(self):
        return [self.B, self.M, self.N, self.K]

    @staticmethod
    def module():
        return "batch_matmul"

    def memory_workload(self):
        if self.mode == "fwd":
            sol_count = 1
            algorithmic_count = 1
        else:
            sol_count = 1 + 1
            algorithmic_count = 1 + (1 + 1)

        buffer_size = (
            self.B * self.M * self.N
            + self.B * self.M * self.N
            + self.B * self.N * self.K
        )
        return {
            "sol": buffer_size * sol_count,
            "algorithmic": buffer_size * algorithmic_count,
        }

    def compute_workload(self):
        if self.mode == "fwd":
            count = 1
        else:
            count = 1 + (1 + 1)

        op_count = 2 * self.B * self.M * self.N * self.K

        return op_count * count

    @staticmethod
    def default_configs():
        return [[128, 64, 128, 256]]


benchmark.register_benchmark_class(MatMulBench)
