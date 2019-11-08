from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch

import operator_benchmark as op_bench

qcomparators_configs = op_bench.cross_product_configs(
    N=(8, 64),
    dtype=(torch.quint8, torch.qint8, torch.qint32),
    contig=(False, True),
    other_scalar=(False, True),
    out_variant=(False, True),
    tags=('short',)
)

qcomparators_ops = op_bench.op_list(
    attrs=(
        ('eq', torch.eq),
        ('ne', torch.ne),
        ('lt', torch.lt),
        ('gt', torch.gt),
        ('le', torch.le),
        ('ge', torch.ge),
    ),
    attr_names=('op_name', 'op_func'),
)


class QComparatorBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, N, dtype, contig, other_scalar, out_variant, op_func):
        # TODO: Consider more diverse shapes
        f_input = (torch.rand(N, N) - 0.5) * 256
        scale = 1.0
        zero_point = 0

        self.q_input_a = torch.quantize_per_tensor(f_input, scale=scale,
                                                   zero_point=zero_point,
                                                   dtype=dtype)
        if other_scalar:
            self.q_input_b = 42
        else:
            self.q_input_b = self.q_input_a.clone()

        if not contig:
            permute_dims = list(range(f_input.ndim))[::-1]
            self.q_input_a = self.q_input_a.permute(permute_dims)

        self.out = torch.tensor([], dtype=torch.bool) if out_variant else None
        self.qop = op_func

    def forward(self):
        if self.out is not None:
            return self.qop(self.q_input_a, self.q_input_b, out=self.out)
        return self.qop(self.q_input_a, self.q_input_b)


op_bench.generate_pt_tests_from_op_list(qcomparators_ops,
                                        qcomparators_configs,
                                        QComparatorBenchmark)


if __name__ == '__main__':
    op_bench.benchmark_runner.main()
