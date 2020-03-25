from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn.quantized as nnq

import operator_benchmark as op_bench

r"""Microbenchmarks for the quantized activations."""

qactivation_long_configs = op_bench.cross_product_configs(
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
    contig=(False, True),
    inplace=(False, True),
    dtype=(torch.quint8,),
    tags=('long',)
)

qactivation_short_configs = op_bench.cross_product_configs(
    dims=((3, 4, 5),      # Rank=3
          (2, 3, 4, 5)),  # Rank=4,
    contig=(False,),
    inplace=(False,),
    dtype=(torch.quint8, torch.qint8, torch.qint32),
    tags=('short',)
)

qactivation_ops = op_bench.op_list(
    attrs=(
        ('relu', nnq.ReLU),
        ('relu6', nnq.ReLU6),
        ('functional.hardtanh', nnq.functional.hardtanh),
        ('functional.elu', nnq.functional.elu),
        ('functional.hardsigmoid', nnq.functional.hardsigmoid),
    ),
    attr_names=('op_name', 'op_func'),
)


class QActivationBenchmarkBase(op_bench.TorchBenchmarkBase):
    r"""Base class for all the activations."""
    def _setup(self, dims, contig, dtype):
        # Input
        f_input = (torch.rand(*dims) - 0.5) * 256
        scale = 1.0
        zero_point = 0

        # Quantize the tensor
        self.q_input = torch.quantize_per_tensor(f_input, scale=scale,
                                                 zero_point=zero_point,
                                                 dtype=dtype)
        if not contig:
            # Make non-contiguous
            new_shape = list(range(self.q_input.ndim))[::-1]
            self.q_input = self.q_input.permute(new_shape)

    def init(self, dims, contig, inplace, dtype, op_func):
        self._setup(dims, contig, dtype)
        self.qop = op_func

    def forward(self):
        return self.qop(self.q_input)


op_bench.generate_pt_tests_from_op_list(qactivation_ops,
                                        qactivation_short_configs + qactivation_long_configs,
                                        QActivationBenchmarkBase)

if __name__ == "__main__":
    op_bench.benchmark_runner.main()
