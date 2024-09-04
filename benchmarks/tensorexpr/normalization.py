from . import benchmark, tensor_engine


class NormalizationBench(benchmark.Benchmark):
    def __init__(self, mode, device, dtype, N, C, H, W):
        super().__init__(mode, device, dtype)
        self.N = N
        self.C = C
        self.H = H
        self.W = W

        self.data = self.nchw_rand(
            [self.N, self.C, self.H, self.W],
            device=device,
            dtype=dtype,
            requires_grad=self.requires_grad,
        )
        self.running_mean = self.rand([self.C], device=device, dtype=dtype)
        self.running_var = self.rand([self.C], device=device, dtype=dtype)
        self.training = self.mode == "both"

    def config(self):
        return [self.N, self.C, self.H, self.W]

    def memory_workload(self):
        if self.mode == "fwd":
            sol_count = 1 + 1
            algorithmic_count = 2 + 1
        else:
            sol_count = (1 + 1) + (1 + 1)
            algorithmic_count = (2 + 1) + (3 + 1)

        buffer_size = self.N * self.C * self.H * self.W * 4
        return {
            "sol": buffer_size * sol_count,
            "algorithmic": buffer_size * algorithmic_count,
        }

    @staticmethod
    def default_configs():
        return [[128, 32, 128, 128]]


class BatchNormBench(NormalizationBench):
    def forward(self):
        y = self.batch_norm(
            self.data, self.running_mean, self.running_var, training=self.training
        )
        return y

    @staticmethod
    def module():
        return "batchnorm"


class InstanceNormBench(NormalizationBench):
    def forward(self):
        y = self.instance_norm(self.data)
        return y

    @staticmethod
    def module():
        return "instance_norm"

    def is_supported(self):
        return tensor_engine.is_supported(self.instance_norm)


class LayerNormBench(NormalizationBench):
    def forward(self):
        y = self.layer_norm(self.data, [self.H, self.W])
        return y

    @staticmethod
    def module():
        return "layernorm"


benchmark.register_benchmark_class(BatchNormBench)
benchmark.register_benchmark_class(InstanceNormBench)
benchmark.register_benchmark_class(LayerNormBench)
