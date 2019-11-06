from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import random

import torch
import torch.nn.quantized as nnq

import operator_benchmark as op_bench

r"""Microbenchmarks for the quantized activations."""

qactivation_configs = op_bench.cross_product_configs(
    dims=(
        # VGG-16 relu's with original shape: (-1, 3, 224, 224)
        ( 64, 224, 224),  # ReLU-1   # noqa
        (128, 112, 112),  # ReLU-6   # noqa
        (256,  56,  56),  # ReLU-11  # noqa
        (512,  28,  28),  # ReLU-18  # noqa
        (512,  14,  14),  # ReLU-25  # noqa
        # Batch = 16
        (16,  64, 224, 224),  # ReLU-1   # noqa
        (16, 128, 112, 112),  # ReLU-6   # noqa
        (16, 256,  56,  56),  # ReLU-11  # noqa
        (16, 512,  28,  28),  # ReLU-18  # noqa
        (16, 512,  14,  14),  # ReLU-25  # noqa
    ),
    permute_dims=(False, True),
    inplace=(False, True),
    dtype=(torch.quint8, torch.qint8, torch.qint32),
    tags=('long',)
)

qactivation_short_configs = op_bench.cross_product_configs(
    dims=((3, 4, 5),      # Rank=3
          (2, 3, 4, 5)),  # Rank=4,
    permute_dims=(False,),
    inplace=(False,),
    dtype=(torch.quint8, torch.qint8, torch.qint32),
    tags=('short',)
)


class _ActivationBenchmarkBase(op_bench.TorchBenchmarkBase):
    r"""Base class for all the activations."""
    def setup(self, dims, permute_dims, dtype):
        # Input dimensions
        f_input = (torch.rand(*dims) - 0.5) * 256
        scale = 1.0
        zero_point = 0

        # Quantize the tensor
        self.q_input = torch.quantize_per_tensor(f_input, scale=scale,
                                                 zero_point=zero_point,
                                                 dtype=dtype)
        if permute_dims:
            # Make non-contiguous
            new_shape = list(range(len(self.q_input.shape)))
            random.shuffle(new_shape)
            self.q_input = self.q_input.permute(new_shape)

    def forward(self):
        return self.qop(self.q_input)


class QReLUBenchmark(_ActivationBenchmarkBase):
    def init(self, dims, permute_dims, inplace, dtype):
        super(QReLUBenchmark, self).setup(dims, permute_dims, dtype)
        self.qop = nnq.ReLU(inplace=inplace)
        self.set_module_name("QReLU")


class QReLU6Benchmark(_ActivationBenchmarkBase):
    def init(self, dims, permute_dims, inplace, dtype):
        super(QReLU6Benchmark, self).setup(dims, permute_dims, dtype)
        # TODO(z-a-f): Enable `inplace` after #29245
        self.qop = nnq.ReLU6(inplace=False)
        self.set_module_name("QReLU6")


op_bench.generate_pt_test(qactivation_short_configs, QReLUBenchmark)
op_bench.generate_pt_test(qactivation_short_configs, QReLU6Benchmark)
op_bench.generate_pt_test(qactivation_configs, QReLUBenchmark)
op_bench.generate_pt_test(qactivation_configs, QReLU6Benchmark)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
