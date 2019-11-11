from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch

import operator_benchmark as op_bench

# 2D pooling will have input matrix of rank 3 or 4
qpool2d_long_configs = op_bench.config_list(
    attrs=(
       #  C    H    W   k       s       p
       (  1,   3,   3, (3, 3), (1, 1), (0, 0)),  # dummy        # noqa
       (  3,  64,  64, (3, 3), (1, 1), (0, 0)),  # dummy        # noqa
       (  3,  64,  64, (3, 3), (2, 2), (1, 1)),  # dummy        # noqa
       # VGG16 pools with original input shape: (-1, 3, 224, 224)
       ( 64, 224, 224, (2, 2), (2, 2), (0, 0)),  # MaxPool2d-4  # noqa
       (128, 112, 112, (2, 2), (2, 2), (0, 0)),  # MaxPool2d-9  # noqa
       (256,  56,  56, (2, 2), (2, 2), (0, 0)),  # MaxPool2d-16 # noqa
       (512,  28,  28, (2, 2), (2, 2), (0, 0)),  # MaxPool2d-23 # noqa
       (512,  14,  14, (2, 2), (2, 2), (0, 0)),  # MaxPool2d-30 # noqa
    ),
    attr_names=('C', 'H', 'W',   # Input layout
                'k', 's', 'p'),  # Pooling parameters
    cross_product_configs={
        'N': range(5),  # if N==0, use rank=3
        'ceil': (False, True),
        'contig': (False, True),
        'dtype': (torch.qint32, torch.qint8, torch.quint8),
    },
    tags=('long',)
)

qpool2d_short_configs = op_bench.config_list(
    attrs=((1, 3, 3, (3, 3), (1, 1), (0, 0)),),  # dummy  # noqa
    attr_names=('C', 'H', 'W',        # Input layout
                'k', 's', 'p'),  # Pooling parameters
    cross_product_configs={
        'N': (2,),
        'ceil': (False,),
        'contig': (True,),
        'dtype': (torch.qint32, torch.qint8, torch.quint8),
    },
    tags=('short',)
)


class _QPool2dBenchmarkBase(op_bench.TorchBenchmarkBase):
    def setup(self, N, C, H, W, dtype, contig):
        # Input
        if N == 0:
            f_input = (torch.rand(C, H, W) - 0.5) * 256
        else:
            f_input = (torch.rand(N, C, H, W) - 0.5) * 256

        scale = 1.0
        zero_point = 0

        # Quantize the tensor
        self.q_input = torch.quantize_per_tensor(f_input, scale=scale,
                                                 zero_point=zero_point,
                                                 dtype=dtype)
        if not contig:
            # Permute into NHWC and back to make it non-contiguous
            if N == 0:
                self.q_input = self.q_input.permute(1, 2, 0).contiguous()
                self.q_input = self.q_input.permute(2, 0, 1)
            else:
                self.q_input = self.q_input.permute(0, 2, 3, 1).contiguous()
                self.q_input = self.q_input.permute(0, 3, 1, 2)

    def forward(self):
        return self.pool_op(self.q_input)


class QMaxPool2dBenchmark(_QPool2dBenchmarkBase):
    def init(self, N, C, H, W, k, s, p, ceil, contig, dtype):
        self.pool_op = torch.nn.MaxPool2d(kernel_size=k, stride=s, padding=p,
                                          dilation=(1, 1), ceil_mode=ceil,
                                          return_indices=False)
        super(QMaxPool2dBenchmark, self).setup(N, C, H, W, dtype, contig)


class QAvgPool2dBenchmark(_QPool2dBenchmarkBase):
    def init(self, N, C, H, W, k, s, p, ceil, contig, dtype):
        self.pool_op = torch.nn.AvgPool2d(kernel_size=k, stride=s, padding=p,
                                          ceil_mode=ceil)
        super(QAvgPool2dBenchmark, self).setup(N, C, H, W, dtype, contig)


op_bench.generate_pt_test(qpool2d_short_configs + qpool2d_long_configs,
                          QAvgPool2dBenchmark)
op_bench.generate_pt_test(qpool2d_short_configs + qpool2d_long_configs,
                          QMaxPool2dBenchmark)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
