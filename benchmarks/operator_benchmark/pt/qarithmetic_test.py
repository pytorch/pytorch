from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch

import operator_benchmark as op_bench

qarithmetic_binary_configs = op_bench.cross_product_configs(
    N=(2, 8, 64, 512),
    dtype=(torch.quint8, torch.qint8, torch.qint32),
    # contig=(False, True),  # TODO: Reenable this after #29435
    contig=(True,),
    tags=('short',)
)

qarithmetic_binary_ops = op_bench.op_list(
    attrs=(
        ('add', 'add'),
        ('add_scalar', 'add_scalar'),
        ('add_relu', 'add_relu'),
        ('mul', 'mul'),
        ('mul_scalar', 'mul_scalar'),
    ),
    attr_names=('op_name', 'op_func'),
)


r"""Base class to use QFunctional.

Children will need to set `self.qop` to the qfunctional op under test.
I.e. `self.qop = 'add'`
"""
class _QFunctionalBinaryArithmeticBenchmarkBase(op_bench.TorchBenchmarkBase):
    def setup(self, N, dtype, contig):
        self.qfunctional = torch.nn.quantized.QFunctional()

        # TODO: Consider more diverse shapes
        f_input = (torch.rand(N, N) - 0.5) * 256
        scale = 1.0
        zero_point = 0

        self.q_input_a = torch.quantize_per_tensor(f_input, scale=scale,
                                                   zero_point=zero_point,
                                                   dtype=dtype)

        if not contig:
            permute_dims = list(range(f_input.ndim))[::-1]
            self.q_input_a = self.q_input_a.permute(permute_dims)

    def forward(self):
        return getattr(self.qfunctional, self.qop)(self.q_input_a,
                                                   self.q_input_b)


class QFunctionalAddBenchmarkBase(_QFunctionalBinaryArithmeticBenchmarkBase):
    def init(self, N, dtype, contig, op_func):
        super(QFunctionalAddBenchmarkBase, self).setup(N, dtype, contig)
        self.qop = op_func
        if self.qop.endswith('_scalar'):
            self.q_input_b = 42
        else:
            self.q_input_b = self.q_input_a


op_bench.generate_pt_tests_from_op_list(qarithmetic_binary_ops,
                                        qarithmetic_binary_configs,
                                        QFunctionalAddBenchmarkBase)


if __name__ == '__main__':
    op_bench.benchmark_runner.main()
