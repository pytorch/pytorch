
import operator_benchmark as op_bench
import torch
import numpy as np

embeddingbag_conversion_short_configs = op_bench.cross_product_configs(
    num_embeddings=(80,),
    embedding_dim=(128, 256, 512),
    tags=('short',)
)

embeddingbag_conversion_long_configs = op_bench.cross_product_configs(
    num_embeddings=(100, 120, 1000),
    embedding_dim=(16, 64, 128, 256, 512, 1024, 2048),
    tags=('long',)
)

conversion_ops = op_bench.op_list(
    attrs=(
        ('qembeddingbag_byte_prepack', torch.ops.quantized.embedding_bag_byte_prepack),
        ('qembeddingbag_4bit_prepack', torch.ops.quantized.embedding_bag_4bit_prepack),
        ('qembeddingbag_2bit_prepack', torch.ops.quantized.embedding_bag_2bit_prepack),
    ),
    attr_names=('op_name', 'op_func'),
)

unpack_ops = op_bench.op_list(
    attrs=(
        ('qembeddingbag_byte_unpack', torch.ops.quantized.embedding_bag_byte_unpack),
        ('qembeddingbag_4bit_unpack', torch.ops.quantized.embedding_bag_4bit_unpack),
        ('qembeddingbag_2bit_unpack', torch.ops.quantized.embedding_bag_2bit_unpack),
    ),
    attr_names=('op_name', 'op_func'),
)

class EmbeddingBagFloatToFusedBase(op_bench.TorchBenchmarkBase):
    def init(self, num_embeddings, embedding_dim, op_func):
        self.inputs = {
            "weight": torch.from_numpy((np.random.random_sample((
                num_embeddings, embedding_dim)) + 1).astype(np.float32))
        }
        self.op_func = op_func

    def forward(self, weight):
        return self.op_func(weight)

class EmbeddingBagFusedToFloatBase(op_bench.TorchBenchmarkBase):
    def init(self, num_embeddings, embedding_dim, op_func):
        weight = torch.randn(num_embeddings, embedding_dim + 8, dtype=torch.float)
        self.inputs = {
            "packed_weight": weight.to(torch.uint8)
        }
        self.op_func = op_func

    def forward(self, packed_weight):
        return self.op_func(packed_weight)


op_bench.generate_pt_tests_from_op_list(conversion_ops,
                                        embeddingbag_conversion_short_configs + embeddingbag_conversion_long_configs,
                                        EmbeddingBagFloatToFusedBase)
op_bench.generate_pt_tests_from_op_list(unpack_ops,
                                        embeddingbag_conversion_short_configs + embeddingbag_conversion_long_configs,
                                        EmbeddingBagFusedToFloatBase)

if __name__ == "__main__":
    op_bench.benchmark_runner.main()
