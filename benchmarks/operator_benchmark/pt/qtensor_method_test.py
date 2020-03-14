from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import operator_benchmark as op_bench
import torch

# Configs for pointwise and reduction unary ops
qmethods_configs_short = op_bench.config_list(
    attr_names=['M', 'N'],
    attrs=[
        [32, 32],
    ],
    cross_product_configs={
        'dtype': [torch.quint8],
        'contig': [False, True],
    },
    tags=['short']
)

qmethods_configs_long = op_bench.cross_product_configs(
    M=[256, 1024],
    N=[256, 1024],
    dtype=[torch.qint8, torch.qint32],
    contig=[False, True],
    tags=['long']
)

qmethods_tensor_input_list = op_bench.op_list(
    attr_names=['op_name', 'op_func'],
    attrs=[
        ['q_copy', 'copy_'],
    ],
)


class _QMethodBenchmarkBase(op_bench.TorchBenchmarkBase):
    def init(self, M, N, dtype, contig, op_func):
        f_input = torch.rand(M, N)
        scale = 1.0
        zero_point = 0
        self.q_input = torch.quantize_per_tensor(f_input, scale=scale,
                                                 zero_point=zero_point,
                                                 dtype=dtype)
        if not contig:
            permute_dims = list(range(self.q_input.ndim))[::-1]
            self.q_input = self.q_input.permute(permute_dims)
        self.op_func = op_func


class QMethodTensorInputBenchmark(_QMethodBenchmarkBase):
    def forward(self):
        getattr(self.q_input, self.op_func)(self.q_input)


class QMethodNoInputBenchmark(_QMethodBenchmarkBase):
    def forward(self):
        getattr(self.q_input, self.op_func)()


op_bench.generate_pt_tests_from_op_list(
    qmethods_tensor_input_list,
    qmethods_configs_short + qmethods_configs_long,
    QMethodTensorInputBenchmark
)

if __name__ == "__main__":
    op_bench.benchmark_runner.main()
