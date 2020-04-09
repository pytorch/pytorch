from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import operator_benchmark as op_bench

import torch
import torch.nn.quantized as nnq


"""Microbenchmarks for quantized Cat operator"""

# Configs for PT Cat operator
qcat_configs_short = op_bench.config_list(
    attr_names=['M', 'N', 'K', 'L', 'dim'],
    attrs=[
        [256, 512, 1, 2, 0],
        [512, 512, 2, 1, 1],
    ],
    cross_product_configs={
        'contig': ('all', 'one', 'none'),
        'dtype': (torch.quint8, torch.qint8, torch.qint32),
    },
    tags=['short'],
)

qcat_configs_long = op_bench.cross_product_configs(
    M=[128, 1024],
    N=[128, 1024],
    K=[1, 2],
    L=[5, 7],
    dim=[0, 1, 2],
    contig=['all', 'one', 'none'],
    dtype=[torch.quint8],
    tags=['long']
)


class QCatBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, M, N, K, L, dim, contig, dtype):
        f_input = (torch.rand(M, N, K) - 0.5) * 256
        self.qf = nnq.QFunctional()
        scale = 1.0
        zero_point = 0
        self.qf.scale = scale
        self.qf.zero_point = zero_point

        assert(contig in ('none', 'one', 'all'))
        q_input = torch.quantize_per_tensor(f_input, scale, zero_point, dtype)
        permute_dims = tuple(range(q_input.ndim - 1, -1, -1))
        q_input_non_contig = q_input.permute(permute_dims).contiguous()
        q_input_non_contig = q_input_non_contig.permute(permute_dims)
        if contig == 'all':
            self.input = (q_input, q_input)
        elif contig == 'one':
            self.input = (q_input, q_input_non_contig)
        elif contig == 'none':
            self.input = (q_input_non_contig, q_input_non_contig)

        self.dim = dim
        self.set_module_name('qcat')

    def forward(self):
        return self.qf.cat(self.input, dim=self.dim)


op_bench.generate_pt_test(qcat_configs_short + qcat_configs_long,
                          QCatBenchmark)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
