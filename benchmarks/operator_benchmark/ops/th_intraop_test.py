from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from operator_benchmark import benchmark_core, benchmark_runner
from operator_benchmark.benchmark_test_generator import *

import torch


"""Microbenchmarks for TH and ATen (PyTorch) CPU intra-op parallelism.

Tests the following functions:

- bitor, cbitor
    - tensor-scalar and tensor-tensor element-wise function), integer-only
    - explicit OMP pragma, THTensorApply macros

- tahn and sigmoid
    - unary ops, THTensorMoreMath macro and UnaryOps

- sumall
    - basic reduction function
    - pragma OMP 'parallel for' with reduction
"""

# Config
config = generate_configs(
    N=[128, 1024, 4096],
    M=[128, 1024, 4096],
    dtype=[torch.float32, torch.int32],
    contig=[True, False],
    mode=['short'],
    sample_func=cross_product
)


def torch_or(tensor_arg):
    def _ior(x, s):
        return x.__ior__(s if tensor_arg else 42)

    x = torch.ones(64, 64, dtype=torch.int32)
    s = torch.ones(64, 64, dtype=torch.int32)
    traced_ior = torch.jit.trace(_ior, (x, s,))

    @torch.jit.script
    def _jit_ior_loop(a, b, iterations):
        # type: (Tensor, Tensor, int)
        for _ in range(iterations):
            traced_ior(a, b)
        return a

    print("torch_or(", tensor_arg, "):\n", _jit_ior_loop.code)
    return _jit_ior_loop


def torch_unary(op_str):
    def _op(x):
        return getattr(x, op_str)()

    x = torch.ones(64, 64)
    traced_op = torch.jit.trace(_op, (x,))

    @torch.jit.script
    def _jit_op_loop(a, b, iterations):
        # type: (Tensor, Tensor, int)
        for _ in range(iterations):
            traced_op(a)
        return a

    print("torch_unary(", op_str, "):\n", _jit_op_loop.code)
    return _jit_op_loop


@torch.jit.script
def torch_sumall(a, b, iterations):
    # type: (Tensor, Tensor, int)
    result = 0.0
    for _ in range(iterations):
        result += float(torch.sum(a))
        a[0][0] += 0.01
    return result
print("torch_sumall:\n", torch_sumall.code)

@benchmark_core.register_test
def test_th_intraop():
    generate_pt_test(
        [config],
        map_pt_config_intraop,
        [('bitor', torch_or(False)),
         ('cbitor', torch_or(True)),
         ('tanh', torch_unary('tanh_')),
         ('sigmoid', torch_unary('sigmoid_')),
         ('sumall', torch_sumall)]
    )


if __name__ == "__main__":
    benchmark_runner.main()
