from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import operator_benchmark as op_bench
import torch


"""Microbenchmarks for PyTorch CPU intra-op parallelism.
"""

# Configs for intraop test benchmark 
intraop_bench_configs = op_bench.cross_product_configs(
    N=[128, 1024, 4096],
    M=[128, 1024, 4096],
    dtype=[torch.float32, torch.int32],
    contig=[True, False],
    tags=["short"]
)


# TODO(mingzhe0908): instead of using str, choose another way of 
# using JIT in this benchmark
def torch_or(tensor_arg):
    jit_ior_loop_code = """\
def forward(self, a, b, iterations):
    # type: (Tensor, Tensor, int)
    for _ in range(iterations):
        a.__ior__({})
    return a
"""
    jit_ior_loop = torch.jit.ScriptModule()
    jit_ior_loop.define(jit_ior_loop_code.format("b" if tensor_arg else "42"))

#    print("torch_or(", tensor_arg, "):\n", jit_ior_loop.code)
    return jit_ior_loop


def tensor_init_helper(N, M, dtype, is_contig):
    tensor_shape = [N, M]
    if not is_contig:
        tensor_shape = [s * 2 for s in tensor_shape]
    if dtype in [torch.float32, torch.float64]: 
        tensor = torch.rand(tensor_shape, dtype=dtype)
    elif not dtype.is_floating_point:
        tensor = torch.randint(low=0, high=100, size=tensor_shape, dtype=dtype)
    else:
        tensor = torch.ones(tensor_shape, dtype=dtype)

    if not is_contig:
        slices = []
        for dim in tensor_shape:
            slices.append(slice(0, dim, 2))
        tensor = tensor[slices]
        assert not tensor.is_contiguous()
    return tensor


class TorchOrTensorBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, N, M, dtype, contig): 
        if dtype.is_floating_point:
            raise op_bench.InputShapeError

        self.input_one = tensor_init_helper(N, M, dtype, contig)
        self.input_two = tensor_init_helper(N, M, dtype, contig)
        self.set_module_name("or")
        self.or_jit = torch_or(False)

    # this is a temp method and will be removed 
    def jit_forward(self, iters):
        return self.or_jit(self.input_one, self.input_two, iters)


class TorchOrScalarBenchmark(TorchOrTensorBenchmark):
    def init(self, N, M, dtype, contig):
        super(TorchOrScalarBenchmark, self).init(N, M, dtype, contig)
        self.or_jit = torch_or(True)


def torch_unary(op_str):
    jit_op_loop_code = """\
def forward(self, a, iterations):
    # type: (Tensor, int)
    for _ in range(iterations):
        a.{}()
    return a
"""
    jit_op_loop = torch.jit.ScriptModule()
    jit_op_loop.define(jit_op_loop_code.format(op_str))

#    print("torch_unary(", op_str, "):\n", jit_op_loop.code)
    return jit_op_loop


class TorchTanhBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, N, M, dtype, contig): 
        if not dtype.is_floating_point:
            raise op_bench.InputShapeError
        self.input_one = tensor_init_helper(N, M, dtype, contig)
        self.set_module_name("tanh_")
        self.tanh_jit = torch_unary('tanh_')

    # this is a temp method and will be removed 
    def jit_forward(self, iters):
        return self.tanh_jit(self.input_one, iters)


class TorchSigmoidBenchmark(TorchTanhBenchmark):
    def init(self, N, M, dtype, contig): 
        super(TorchSigmoidBenchmark, self).init(N, M, dtype, contig)
        if not dtype.is_floating_point:
            raise op_bench.InputShapeError
        self.set_module_name("sigmoid_")
        self.sigmoid_jit = torch_unary('sigmoid_')

    # this is a temp method and will be removed 
    def jit_forward(self, iters):
        return self.sigmoid_jit(self.input_one, iters)


@torch.jit.script
def torch_sumall(a, iterations):
    # type: (Tensor, int)
    result = 0.0
    for _ in range(iterations):
        result += float(torch.sum(a))
        a[0][0] += 0.01
    return result


class TorchSumBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, N, M, dtype, contig): 
        if not dtype.is_floating_point:
            raise op_bench.InputShapeError
        self.input_one = tensor_init_helper(N, M, dtype, contig)
        self.set_module_name("sum")

    # this is a temp method and will be removed 
    def jit_forward(self, iters):
        return torch_sumall(self.input_one, iters)


op_bench.generate_pt_test(intraop_bench_configs, TorchOrTensorBenchmark)
op_bench.generate_pt_test(intraop_bench_configs, TorchOrScalarBenchmark)
op_bench.generate_pt_test(intraop_bench_configs, TorchTanhBenchmark)
op_bench.generate_pt_test(intraop_bench_configs, TorchSigmoidBenchmark)
op_bench.generate_pt_test(intraop_bench_configs, TorchSumBenchmark)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
