from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from operator_benchmark import benchmark_core, benchmark_runner
from operator_benchmark.benchmark_test_generator import *

import torch


"""Microbenchmarks for PyTorch CPU intra-op parallelism.

Tests the following functions:

- bitor, cbitor
    - tensor-scalar and tensor-tensor element-wise function, integer-only

- tahn and sigmoid
    - unary ops

- sumall
    - basic reduction function
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
    jit_ior_loop_code = """\
def forward(self, a, b, iterations):
    # type: (Tensor, Tensor, int)
    for _ in range(iterations):
        a.__ior__({})
    return a
"""
    jit_ior_loop = torch.jit.ScriptModule()
    jit_ior_loop.define(jit_ior_loop_code.format("b" if tensor_arg else "42"))

    print("torch_or(", tensor_arg, "):\n", jit_ior_loop.code)
    return jit_ior_loop


def torch_unary(op_str):
    jit_op_loop_code = """\
def forward(self, a, b, iterations):
    # type: (Tensor, Tensor, int)
    for _ in range(iterations):
        a.{}()
    return a
"""
    jit_op_loop = torch.jit.ScriptModule()
    jit_op_loop.define(jit_op_loop_code.format(op_str))

    print("torch_unary(", op_str, "):\n", jit_op_loop.code)
    return jit_op_loop


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
