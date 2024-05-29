import scipy.special

from . import benchmark


class SoftmaxBench(benchmark.Benchmark):
    def __init__(self, mode, device, dtype, M, N):
        super().__init__(mode, device, dtype)
        self.M = M
        self.N = N
        self.dtype = dtype
        self.inputs = [
            self.randn(
                [M, N], device=device, dtype=dtype, requires_grad=self.requires_grad
            )
        ]

    def forward(self, inputs):
        x = self.add(inputs, 0.001)
        y = self.softmax(x, dim=-1, dtype=self.dtype)
        return y

    def reference(self):
        return scipy.special.softmax(self.numpy(self.inputs), axis=-1)

    def config(self):
        return [self.M, self.N]

    @staticmethod
    def module():
        return "softmax"

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
        return [
            [480, 20],
            [1 << 15, 32],
            [128, 1 << 16],
        ]


benchmark.register_benchmark_class(SoftmaxBench)
