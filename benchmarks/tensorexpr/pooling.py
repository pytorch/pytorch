import benchmark


class PoolingBench(benchmark.Benchmark):
    def __init__(self, case, mode, device, kernel_size, N, C, H, W):
        super().__init__(mode, device)
        self.case = case
        self.kernel_size = kernel_size
        self.N = N
        self.C = C
        self.H = H
        self.W = W
        self.data = self.rand(
            [N, C, H, W], device=device, requires_grad=self.requires_grad
        )

    def forward(self):
        if self.case == "maxpool":
            y = self.max_pool2d(self.data, self.kernel_size, stride=1)
        elif self.case == "avgpool":
            y = self.avg_pool2d(self.data, self.kernel_size, stride=1)
        return y

    def config(self):
        return [self.kernel_size, self.N, self.C, self.H, self.W]

    def memory_workload(self):
        if self.mode == "fwd":
            sol_count = 1 + 1
            algorithmic_count = 1 + 1
        else:
            sol_count = (1 + 1) + (1 + 1)
            algorithmic_count = (1 + 1) + (2 + 1)

        buffer_size = self.N * self.C * self.H * self.W * 4
        return {
            "sol": buffer_size * sol_count,
            "algorithmic": buffer_size * algorithmic_count,
        }

    @staticmethod
    def default_configs():
        return [[3, 16, 32, 256, 256]]


class MaxPoolBench(PoolingBench):
    def __init__(self, *args):
        super().__init__("maxpool", *args)

    @staticmethod
    def module():
        return "maxpool"


class AvgPoolBench(PoolingBench):
    def __init__(self, *args):
        super().__init__("avgpool", *args)

    @staticmethod
    def module():
        return "avgpool"


benchmark.register_benchmark_class(MaxPoolBench)
benchmark.register_benchmark_class(AvgPoolBench)
