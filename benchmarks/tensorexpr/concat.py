import numpy as np

import torch
from . import benchmark


class Concat2D2InputBench(benchmark.Benchmark):
    def __init__(self, mode, device, dtype, I1_D1, I1_D2, I2_D1, I2_D2, concat_dim):
        super().__init__(mode, device, dtype)
        self.I1_D1 = I1_D1
        self.I1_D2 = I1_D2
        self.I2_D1 = I2_D1
        self.I2_D2 = I2_D2
        self.concat_dim = concat_dim
        self.input1 = self.randn(
            [I1_D1, I1_D2], device=device, dtype=dtype, requires_grad=self.requires_grad
        )
        self.input2 = self.randn(
            [I2_D1, I2_D2], device=device, dtype=dtype, requires_grad=self.requires_grad
        )
        self.inputs = [self.input1, self.input2]

    def forward(self, input1, input2):
        x1 = self.add(input1, 0.00001)
        x2 = self.add(input2, 0.00001)
        y = self.cat((x1, x2), dim=self.concat_dim)
        return y

    def reference(self):
        return np.concatenate(
            (self.numpy(self.input1), self.numpy(self.input2)),
            axis=self.concat_dim,
        )

    def config(self):
        return [self.I1_D1, self.I1_D2, self.I2_D1, self.I2_D2, self.concat_dim]

    @staticmethod
    def module():
        return "concat2d2input"

    def memory_workload(self):
        if self.mode == "fwd":
            sol_count = 1 + 1
            algorithmic_count = 3 + 1
        else:
            sol_count = (1 + 1) + (1 + 1)
            algorithmic_count = (3 + 1) + (3 + 1)

        buffer_size = self.I1_D1 * self.I1_D2 + self.I2_D1 * self.I2_D2
        return {
            "sol": buffer_size * sol_count,
            "algorithmic": buffer_size * algorithmic_count,
        }

    @staticmethod
    def default_configs():
        return [
            [1, 160, 1, 14, 1],
            [1, 580, 1, 174, 1],
            [20, 160, 20, 14, 1],
            [20, 580, 20, 174, 1],
            [8, 512, 8, 512, 1],
            [1 << 13, 1060, 1 << 13, 1040, 1],
            [1 << 13, 2000, 1 << 13, 1074, 1],
            [1 << 15, 1060, 1 << 15, 2670, 1],
            [1 << 15, 5120, 1 << 15, 2512, 1],
        ]


benchmark.register_benchmark_class(Concat2D2InputBench)


class ConcatGraphOptBench(benchmark.Benchmark):
    def __init__(self, mode, device, dtype, I1_D1, I1_D2, I2_D1, I2_D2, concat_dim):
        super().__init__(mode, device, dtype)
        self.I1_D1 = I1_D1
        self.I1_D2 = I1_D2
        self.I2_D1 = I2_D1
        self.I2_D2 = I2_D2
        self.concat_dim = concat_dim
        self.input1 = self.randn(
            [I1_D1, I1_D2], device=device, dtype=dtype, requires_grad=self.requires_grad
        )
        self.input2 = self.randn(
            [I2_D1, I2_D2], device=device, dtype=dtype, requires_grad=self.requires_grad
        )
        self.inputs = [self.input1, self.input2]
        torch._C._jit_override_can_fuse_on_cpu(True)
        torch._C._jit_cat_wo_conditionals(True)

    def forward(self, input1, input2):
        x1 = self.add(input1, 0.00001)
        x2 = self.add(input2, 0.00001)
        y = self.cat((x1, x2), dim=self.concat_dim)
        z = self.relu(y)
        return z

    def reference(self):
        return np.concatenate(
            (self.numpy(self.input1), self.numpy(self.input2)),
            axis=self.concat_dim,
        )

    def config(self):
        return [self.I1_D1, self.I1_D2, self.I2_D1, self.I2_D2, self.concat_dim]

    @staticmethod
    def module():
        return "concatGraphOpt"

    def memory_workload(self):
        if self.mode == "fwd":
            sol_count = 1 + 1
            algorithmic_count = 3 + 1
        else:
            sol_count = (1 + 1) + (1 + 1)
            algorithmic_count = (3 + 1) + (3 + 1)

        buffer_size = self.I1_D1 * self.I1_D2 + self.I2_D1 * self.I2_D2
        return {
            "sol": buffer_size * sol_count,
            "algorithmic": buffer_size * algorithmic_count,
        }

    @staticmethod
    def default_configs():
        return [
            [1 << 13, 1060, 1 << 13, 1040, 1],
            [1 << 13, 2000, 1 << 13, 1074, 1],
            [1 << 15, 1060, 1 << 15, 2670, 1],
            [1 << 15, 5120, 1 << 15, 2512, 1],
        ]


benchmark.register_benchmark_class(ConcatGraphOptBench)
