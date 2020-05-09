from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import operator_benchmark as op_bench
import torch

'''Microbenchmarks for the quantized interpolate op.

Note: We are not benchmarking `upsample` as it is being depricated, and calls
the `interpolate` anyway.
'''

qinterpolate_long_configs = op_bench.config_list(
    attr_names=['M', 'N', 'K'],
    attrs=[
        [512, 512, 512],
    ],
    cross_product_configs={
        'dtype': [torch.quint8, torch.qint8, torch.qint32],
        'mode': ['nearest', 'bilinear'],
        'scale': [0.5, 1.0, 2.0],
        'contig': [True],  # TODO: Add `False` after #29435
    },
    tags=['long']
)


qinterpolate_short_configs = op_bench.config_list(
    attr_names=['M', 'N', 'K', 'dtype', 'mode', 'scale', 'contig'],
    attrs=[
        [32, 32, 32, torch.quint8, 'nearest', 0.5, True],  # Downsample
        [32, 32, 32, torch.quint8, 'bilinear', 0.5, True],  # Downsample
        [32, 32, 32, torch.quint8, 'nearest', 2.0, True],  # Upsample
        [32, 32, 32, torch.quint8, 'bilinear', 2.0, True],  # Upsample
    ],
    tags=['short'],
)


class QInterpolateBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, M, N, K, dtype, mode, scale, contig):
        f_input = (torch.rand(1, M, N, K) - 0.5) * 256
        scale = 0.1
        zero_point = 42
        self.q_input = torch.quantize_per_tensor(f_input, scale=scale,
                                                 zero_point=zero_point,
                                                 dtype=dtype)
        if not contig:
            permute_dims = list(range(q_input.ndim))[::-1]
            self.q_input_a = self.q_input_a.permute(permute_dims)

        self.mode = mode
        self.scale_factor = scale
        self.set_module_name('q_interpolate')

    def forward(self):
        return torch.nn.quantized.functional.interpolate(
            self.q_input, scale_factor=self.scale_factor, mode=self.mode)


op_bench.generate_pt_test(qinterpolate_short_configs + qinterpolate_long_configs,
                          QInterpolateBenchmark)


if __name__ == '__main__':
    op_bench.benchmark_runner.main()
