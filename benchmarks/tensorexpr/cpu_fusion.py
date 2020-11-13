from . import benchmark
import numpy as np
import scipy.special
import torch

# A template class for unary ops to run with / without CPU Fusion.
class UnaryOpBench(benchmark.Benchmark):
    def __init__(self, mode, device, dtype, cpu_fusion, fwd_func, ref_func, M):
        super().__init__(mode, device, dtype)
        self.M = M
        self.cpu_fusion = cpu_fusion
        self.fwd_func = fwd_func
        self.ref_func = ref_func
        self.inputs = [self.rand([M], device=device, requires_grad=self.requires_grad)]

    def forward(self, inputs):
        old_cpu_fusion = torch._C._jit_can_fuse_on_cpu()
        torch._C._jit_override_can_fuse_on_cpu(self.cpu_fusion)
        y = self.fwd_func(inputs)
        torch._C._jit_override_can_fuse_on_cpu(old_cpu_fusion)
        return y

    def reference(self):
        return self.ref_func(self.numpy(self.inputs))

    def config(self):
        return [self.M]

    @staticmethod
    def module():
        return "unary_op"

    def memory_workload(self):
        sol_count = 1 + 1
        algorithmic_count = 1 + 1
        buffer_size = self.M
        return {
            "sol": buffer_size * sol_count,
            "algorithmic" : buffer_size * algorithmic_count,
        }

    @staticmethod
    def default_configs():
        return [[164 * 164], [456 * 456], [768 * 768], [1 << 20]]

class SigmoidWithCpuFusionBench(UnaryOpBench):
    def __init__(self, mode, device, dtype, M):
        super(SigmoidWithCpuFusionBench, self).__init__(mode, device, dtype, True, torch.sigmoid, scipy.special.expit, M)

    @staticmethod
    def module():
        return "sigmoid_with_cpu_fusion"

class SigmoidWoCpuFusionBench(UnaryOpBench):
    def __init__(self, mode, device, dtype, M):
        super(SigmoidWoCpuFusionBench, self).__init__(mode, device, dtype, False, torch.sigmoid, scipy.special.expit, M)

    @staticmethod
    def module():
        return "sigmoid_wo_cpu_fusion"

class TanhWithCpuFusionBench(UnaryOpBench):
    def __init__(self, mode, device, dtype, M):
        super(TanhWithCpuFusionBench, self).__init__(mode, device, dtype, True, torch.tanh, np.tanh, M)

    @staticmethod
    def module():
        return "tanh_with_cpu_fusion"

class TanhWoCpuFusionBench(UnaryOpBench):
    def __init__(self, mode, device, dtype, M):
        super(TanhWoCpuFusionBench, self).__init__(mode, device, dtype, False, torch.tanh, np.tanh, M)

    @staticmethod
    def module():
        return "tanh_wo_cpu_fusion"

benchmark.register_benchmark_class(SigmoidWithCpuFusionBench)
benchmark.register_benchmark_class(SigmoidWoCpuFusionBench)
benchmark.register_benchmark_class(TanhWithCpuFusionBench)
benchmark.register_benchmark_class(TanhWoCpuFusionBench)
