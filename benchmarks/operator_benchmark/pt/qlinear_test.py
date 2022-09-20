
import operator_benchmark as op_bench

import torch
import torch.ao.nn.quantized as nnq
import torch.ao.nn.quantized.dynamic as nnqd

from pt import configs

"""
Microbenchmarks for Quantized Linear operators.
"""

class _QLinearBenchmarkBase(op_bench.TorchBenchmarkBase):
    def init(self, N, IN, OUT, linear_under_test):
        scale = torch.tensor(1.0 / 255)
        zero_point = torch.tensor(0)
        self.X = torch.randn(N, IN, dtype=torch.float32)
        self.qX = torch.quantize_per_tensor(self.X, scale=scale, zero_point=zero_point, dtype=torch.quint8)
        W = torch.randn(OUT, IN, dtype=torch.float32)
        qW = torch.quantize_per_tensor(W, scale=scale, zero_point=0, dtype=torch.qint8)

        # Assume that the `self.qlinear` is set in the child
        self.qlinear = linear_under_test
        self.qlinear.weight = qW
        self.qlinear.scale = scale
        self.qlinear.zero_point = zero_point

    def forward(self, input):
        # Assume that the `self.input` is set in the child
        return self.qlinear(input)

class QLinearBenchmark(_QLinearBenchmarkBase):
    def init(self, N, IN, OUT, device):
        super(QLinearBenchmark, self).init(N, IN, OUT, nnq.Linear(IN, OUT))
        self.inputs = {
            "input": self.qX
        }
        self.set_module_name("QLinear")


class QDynamicLinearBenchmark(_QLinearBenchmarkBase):
    def init(self, N, IN, OUT, device):
        super(QDynamicLinearBenchmark, self).init(N, IN, OUT, nnqd.Linear(IN, OUT))
        self.inputs = {
            "input": self.X
        }
        self.set_module_name("QDynamicLinear")


op_bench.generate_pt_test(configs.remove_cuda(configs.linear_configs_short + configs.linear_configs_long), QLinearBenchmark)
op_bench.generate_pt_test(configs.remove_cuda(configs.linear_configs_short + configs.linear_configs_long), QDynamicLinearBenchmark)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
