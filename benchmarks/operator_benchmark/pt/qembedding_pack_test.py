from __future__ import absolute_import, division, print_function, unicode_literals

import operator_benchmark as op_bench
import torch
import numpy as np

embeddingbag_conversion_configs = op_bench.cross_product_configs(
    num_embeddings=(100, 80, 120, 1000),
    embedding_dim=(16, 4, 16, 64, 128),
    tags=('short',)
)

conversion_ops = op_bench.op_list(
    attrs=(
        ('qembeddingbag_byte_prepack', torch.ops.quantized.embedding_bag_byte_prepack),
        ('qembeddingbag_4bit_prepack', torch.ops.quantized.embedding_bag_4bit_prepack),
    ),
    attr_names=('op_name', 'op_func'),
)

unpack_ops = op_bench.op_list(
    attrs=(
        ('qembeddingbag_byte_unpack', torch.ops.quantized.embedding_bag_byte_unpack),
        ('qembeddingbag_4bit_unpack', torch.ops.quantized.embedding_bag_4bit_unpack),
    ),
    attr_names=('op_name', 'op_func'),
)

class EmbeddingBagFloatToFusedBase(op_bench.TorchBenchmarkBase):
    def init(self, num_embeddings, embedding_dim, op_func):
        self.weight = torch.from_numpy((np.random.random_sample((
            num_embeddings, embedding_dim)) + 1).astype(np.float32))
        self.op_func = op_func

    def forward(self):
        return self.op_func(self.weight)

class EmbeddingBagFusedToFloatBase(op_bench.TorchBenchmarkBase):
    def init(self, num_embeddings, embedding_dim, op_func):
        weight = torch.randn(num_embeddings, embedding_dim + 8, dtype=torch.float)
        self.packed_weight = weight.to(torch.uint8)
        self.op_func = op_func

    def forward(self):
        return self.op_func(self.packed_weight)


op_bench.generate_pt_tests_from_op_list(conversion_ops, embeddingbag_conversion_configs, EmbeddingBagFloatToFusedBase)
op_bench.generate_pt_tests_from_op_list(unpack_ops, embeddingbag_conversion_configs, EmbeddingBagFusedToFloatBase)

if __name__ == "__main__":
    op_bench.benchmark_runner.main()
