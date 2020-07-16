from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import operator_benchmark as op_bench
import torch

"""Microbenchmarks for learnable fake quantize operators."""

TORCH_RANDOM_SEED = 41

fake_quantize_learnable_dict = {
    'attr_names': ['N', 'C', 'H', 'W', 'nbits'],
    'attrs': [
        [16, 3, 256, 256, 4],
        [16, 3, 256, 256, 8],
    ],
    'tags': ['short'],
}

fake_quantize_learnable_configs = op_bench.config_list(
    **fake_quantize_learnable_dict
)

class FakeQuantizeLearnablePerTensorBenchmark(op_bench.TorchBenchmarkBase):
    r"""Benchmarks learnable fake quantize per tensor."""
    def init(self, N, C, H, W, nbits):
        torch.manual_seed(TORCH_RANDOM_SEED)
        self.quant_min = 0
        self.quant_max = 2 ** nbits - 1
        self.quant_range = 2 ** nbits
        self.input = torch.rand(N, C, H, W, dtype=torch.float)
        self.scale = torch.tensor([1.])
        self.zero_point = torch.tensor([0.])
        self.input.requires_grad_()
        self.scale.requires_grad_()
        self.zero_point.requires_grad_()

    def forward(self):
        return torch._fake_quantize_learnable_per_tensor_affine(
            self.input, self.scale, self.zero_point, self.quant_min, self.quant_max
        )

op_bench.generate_pt_test(
    fake_quantize_learnable_configs,
    FakeQuantizeLearnablePerTensorBenchmark
)
op_bench.generate_pt_gradient_test(
    fake_quantize_learnable_configs,
    FakeQuantizeLearnablePerTensorBenchmark
)

class FakeQuantizeLearnablePerChannelBenchmark(op_bench.TorchBenchmarkBase):
    r"""Benchmarks learnable fake quantize per channel."""
    def init(self, N, C, H, W, nbits):
        torch.manual_seed(TORCH_RANDOM_SEED)
        self.quant_min = 0
        self.quant_max = 2 ** nbits - 1
        self.quant_range = 2 ** nbits
        # Axis is chosen with respect to the number of channels: C.
        self.axis = 1
        self.input = torch.rand(N, C, H, W, dtype=torch.float)
        self.scale = torch.tensor([1.] * C)
        self.zero_point = torch.tensor([0.] * C)
        self.input.requires_grad_()
        self.scale.requires_grad_()
        self.zero_point.requires_grad_()

    def forward(self):
        return torch._fake_quantize_learnable_per_channel_affine(
            self.input, self.scale, self.zero_point, self.axis, self.quant_min, self.quant_max
        )

op_bench.generate_pt_test(
    fake_quantize_learnable_configs,
    FakeQuantizeLearnablePerChannelBenchmark
)
op_bench.generate_pt_gradient_test(
    fake_quantize_learnable_configs,
    FakeQuantizeLearnablePerChannelBenchmark
)

if __name__ == "__main__":
    op_bench.benchmark_runner.main()
