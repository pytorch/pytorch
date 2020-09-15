from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import operator_benchmark as op_bench
import torch
import torch.nn.quantized as nnq
import torch.quantization as tq
from torch.quantization._learnable_fake_quantize import (
    _LearnableFakeQuantizePerTensorOp,
    _LearnableFakeQuantizePerChannelOp
)


"""Microbenchmarks for general quantization operations."""

# mode is used to show the direction of the benchmark:
# if 'Q', benchmark quantization, else dequantization

quantize_configs_short_dict = {
    'attr_names': ['C', 'M', 'N', 'dtype', 'mode'],
    'attrs': [
        [3, 512, 512, torch.quint8, 'Q'],
        [3, 512, 512, torch.quint8, 'D'],
    ],
    'tags': ['short'],
}

quantize_configs_long_dict = {
    'C': [3, 5, 8],  # this is reused for per-channel: avoid single channel test
    'M': [256, 1024],
    'N': [256, 1024],
    'dtype': [torch.quint8, torch.qint8, torch.qint32],
    'mode': ['D', 'Q'],
    'tags': ['long'],
}


quantize_per_tensor_configs_short = op_bench.config_list(
    **quantize_configs_short_dict
)

quantize_per_tensor_configs_long = op_bench.cross_product_configs(
    **quantize_configs_long_dict
)


class QuantizePerTensorBenchmark(op_bench.TorchBenchmarkBase):
    r"""Benchmarks both quantization and dequantization."""
    def init(self, C, M, N, dtype, mode):
        assert(mode in ('Q', 'D'))
        self.input = torch.rand(C, M, N)
        self.dtype = dtype
        self.op = nnq.Quantize(scale=1.0, zero_point=0, dtype=dtype)
        self.set_module_name('QuantizePerTensor')

        if mode == 'D':
            self.input = self.op(self.input)
            self.op = nnq.DeQuantize()
            self.set_module_name('DequantizePerTensor')

    def forward(self):
        return self.op(self.input)


op_bench.generate_pt_test(
    quantize_per_tensor_configs_short + quantize_per_tensor_configs_long,
    QuantizePerTensorBenchmark)

# === Per Channel quantization ===

quantize_per_channel_configs_short = op_bench.config_list(
    cross_product_configs={
        'axis': (0,)
    },
    **quantize_configs_short_dict
)

quantize_per_channel_configs_long = op_bench.cross_product_configs(
    axis=(0, 1, 2),
    **quantize_configs_long_dict
)

class QuantizePerChannelBenchmark(op_bench.TorchBenchmarkBase):
    r"""Benchmarks both quantization and dequantization."""
    def init(self, C, M, N, dtype, axis, mode):
        assert(mode in ('Q', 'D'))
        self.input = torch.rand(C, M, N)
        self.op = torch.quantize_per_channel

        channel_len = (C, M, N)[axis]

        self.kwargs = {
            'scales': torch.tensor([1.0] * channel_len),
            'zero_points': torch.tensor([0] * channel_len),
            'dtype': dtype,
            'axis': axis
        }

        self.set_module_name('QuantizePerChannel')

        if mode == 'D':
            self.input = self.op(self.input, **self.kwargs)
            # Dequantize doesn't take any arguments
            self.op = lambda x, **kwargs: x.dequantize()
            self.set_module_name('DequantizePerChannel')

    def forward(self):
        return self.op(self.input, **self.kwargs)


op_bench.generate_pt_test(
    quantize_per_channel_configs_short + quantize_per_channel_configs_long,
    QuantizePerChannelBenchmark)

# === Fake Quantization ===

fake_quantize_configs_short_dict = {
    'attr_names': ['N', 'C', 'H', 'W'],
    'attrs': [
        [1, 3, 512, 512],
        [1, 3, 512, 512]
    ],
    'tags': ['short']
}

fake_quantize_configs_long_dict = {
    'N': [1],
    'C': [1, 3, 8],
    'H': [256, 1024],
    'W': [256, 1024],
    'tags': ['long']
}

fake_quantize_configs_short = op_bench.config_list(
    **fake_quantize_configs_short_dict
)

fake_quantize_configs_long = op_bench.cross_product_configs(
    **fake_quantize_configs_long_dict
)


class FakeQuantizeBenchmark(op_bench.TorchBenchmarkBase):
    r"""Benchmarks fake quantization with default parameters."""
    def init(self, N, C, H, W):
        self.input = torch.rand(N, C, H, W)
        self.op = tq.FakeQuantize()
        self.set_module_name('FakeQuantize')

    def forward(self):
        return self.op(self.input)


op_bench.generate_pt_test(
    fake_quantize_configs_short + fake_quantize_configs_long,
    FakeQuantizeBenchmark)

# op_type is used to describe the type of operator used in benchmarking:
# py_module represents the operator written in Python that can
# backpropagate on scale and zero point.
# learnable_kernel represents the c++ kernel that can backpropagate on
# scale and zero point.
# original_kernel represents the original fake quantize c++ kernel.

fake_quantize_operator_configs_short = op_bench.config_list(
    cross_product_configs={
        'nbits': (4, 8),
        'device': ('cpu', 'cuda'),
        'op_type': ('py_module', 'learnable_kernel', 'original_kernel')
    },
    **fake_quantize_configs_short_dict
)

fake_quantize_operator_configs_long = op_bench.cross_product_configs(
    nbits=(4, 8),
    device=('cpu', 'cuda'),
    op_type=('py_module', 'learnable_kernel', 'original_kernel'),
    **fake_quantize_configs_long_dict
)

class FakeQuantizePerTensorOpBenchmark(op_bench.TorchBenchmarkBase):
    r"""Benchmarks 3 different fake quantize per tensor operators."""
    def init(self, N, C, H, W, nbits, device, op_type):
        self.quant_min = 0
        self.quant_max = 2 ** nbits - 1
        self.quant_range = 2 ** nbits
        self.input = torch.rand(N, C, H, W, dtype=torch.float, device=device)
        self.scale = torch.tensor([1.]).to(device)
        self.zero_point = torch.tensor([0.]).to(device)
        self.input.requires_grad_()
        self.scale.requires_grad_()
        self.zero_point.requires_grad_()
        self.args = [
            self.input, self.scale, self.zero_point,
            self.quant_min, self.quant_max
        ]
        if op_type == 'py_module':
            self.op = _LearnableFakeQuantizePerTensorOp.apply
            self.args.append(1.)
        elif op_type == 'learnable_kernel':
            self.op = torch._fake_quantize_learnable_per_tensor_affine
        else:
            # Replace tensors with float and long types for original per tensor
            # fake quantize kernel.
            self.args[1], self.args[2] = 1., 0
            self.op = torch.fake_quantize_per_tensor_affine

    def forward(self):
        return self.op(*self.args)

op_bench.generate_pt_test(
    fake_quantize_operator_configs_short + fake_quantize_operator_configs_long,
    FakeQuantizePerTensorOpBenchmark
)

op_bench.generate_pt_gradient_test(
    fake_quantize_operator_configs_short + fake_quantize_operator_configs_long,
    FakeQuantizePerTensorOpBenchmark
)

class FakeQuantizePerChannelOpBenchmark(op_bench.TorchBenchmarkBase):
    r"""Benchmarks 3 different fake quantize per channel operators."""
    def init(self, N, C, H, W, nbits, device, op_type):
        self.quant_min = 0
        self.quant_max = 2 ** nbits - 1
        self.quant_range = 2 ** nbits
        # Axis is chosen with respect to the number of channels: C.
        self.axis = 1
        self.input = torch.rand(N, C, H, W, dtype=torch.float, device=device)
        self.scale = torch.ones(C, device=device, dtype=torch.float32)
        self.zero_point = torch.zeros(C, device=device, dtype=torch.float32)
        self.input.requires_grad_()
        self.scale.requires_grad_()
        self.zero_point.requires_grad_()
        self.args = [
            self.input, self.scale, self.zero_point,
            self.axis, self.quant_min, self.quant_max
        ]
        if op_type == 'py_module':
            self.op = _LearnableFakeQuantizePerChannelOp.apply
            self.args.append(1.)
        elif op_type == 'learnable_kernel':
            self.op = torch._fake_quantize_learnable_per_channel_affine
        else:
            self.args[1] = torch.ones(C, device=device, dtype=torch.float32)
            self.args[2] = torch.zeros(C, device=device, dtype=torch.int64)
            self.op = torch.fake_quantize_per_channel_affine

    def forward(self):
        return self.op(*self.args)

op_bench.generate_pt_test(
    fake_quantize_operator_configs_short + fake_quantize_operator_configs_long,
    FakeQuantizePerChannelOpBenchmark
)

op_bench.generate_pt_gradient_test(
    fake_quantize_operator_configs_short + fake_quantize_operator_configs_long,
    FakeQuantizePerChannelOpBenchmark
)

if __name__ == "__main__":
    op_bench.benchmark_runner.main()
