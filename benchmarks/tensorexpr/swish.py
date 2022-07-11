from . import benchmark
import torch


class SwishBench(benchmark.Benchmark):
    def __init__(self, mode, device, dtype, M, N):
        super().__init__(mode, device, dtype)
        self.M = M
        self.N = N
        self.data = self.rand([M, N], device=device, dtype=dtype, requires_grad=self.requires_grad)
        self.inputs = [self.data]
        self.zeros = torch.zeros(M, N, device=device)
        self.six = self.zeros + 6.0
        self.three = self.zeros + 3.0
        self.sixth = self.zeros + 1.0 / 6.0

    def forward(self, inp):
        y = inp * (torch.min(torch.relu(inp), self.six) + self.three) * self.sixth
        return y

    def reference(self):
        return self.numpy(self.forward(self.data))

    def config(self):
        return [self.M, self.N]

    @staticmethod
    def module():
        return "swish"

    def memory_workload(self):
        if self.mode == "fwd":
            sol_count = 1 + 1
            algorithmic_count = 3 + 1
        else:
            sol_count = (1 + 1) + (1 + 1)
            algorithmic_count = (3 + 1) + (3 + 1)

        buffer_size = self.M * self.N
        return {
            "sol": buffer_size * sol_count,
            "algorithmic": buffer_size * algorithmic_count,
        }

    @staticmethod
    def default_configs():
        return [[128, 1 << 16]]


benchmark.register_benchmark_class(SwishBench)
