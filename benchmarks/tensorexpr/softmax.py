import benchmark
import scipy.special


class SoftmaxBench(benchmark.Benchmark):
    def __init__(self, mode, device, M, N):
        super().__init__(mode, device)
        self.M = M
        self.N = N
        self.data = self.rand([M, N], device=device, requires_grad=self.requires_grad)

    def forward(self):
        y = self.softmax(self.data, dim=1)
        return y

    def reference(self):
        return scipy.special.softmax(self.numpy(self.data), axis=1)

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

        buffer_size = self.M * self.N * 4
        return {
            "sol": buffer_size * sol_count,
            "algorithmic": buffer_size * algorithmic_count,
        }

    @staticmethod
    def default_configs():
        return [[128, 1 << 16]]


benchmark.register_benchmark_class(SoftmaxBench)
