from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch

import operator_benchmark as op_bench

qarithmetic_binary_configs = op_bench.cross_product_configs(
    N=(2, 8, 64, 512),
    dtype=(torch.quint8, torch.qint8, torch.qint32),
    contig=(False, True),
)


r"""Base class to use QFunctional.

Children will need to set `self.qop` to the qfunctional op under test.
I.e. `self.qop = 'add'`
"""
class _QFunctionalBinaryArithmeticBenchmarkBase(op_bench.TorchBenchmarkBase):
    def setup(self, N, dtype, contig):
        # TODO: Consider more diverse shapes
        f_input = (torch.rand(N, N) - 0.5) * 256
        scale = 1.0
        zero_point = 0

        self.q_input = torch.quantize_per_tensor(f_input, scale=scale,
                                                 zero_point=zero_point,
                                                 dtype=dtype)

        if not contig:
            permute_dims = list(range(f_input.ndim))[::-1]
            self.q_input = self.q_input.permute(permute_dims)

        self.qfunctional = QFunctional()

    def forward(self):
        return getattr(self.qfunctional, self.qop)(self.q_input, self.q_input)


class QFunctionalAddBenchmarkBase(_QFunctionalBinaryArithmeticBenchmarkBase):
    def init(self, N, dtype, contig):
        super(QFunctionalAddBenchmarkBase, self).setup(N, dtype, contig)
        self.qop = 'add'


op_bench.generate_pt(qarithmetic_binary_configs, QFunctionalAddBenchmarkBase)


if __name__ == '__main__':
    op_bench.benchmark_runner.main()
