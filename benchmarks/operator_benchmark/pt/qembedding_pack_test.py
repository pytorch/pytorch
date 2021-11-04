
import operator_benchmark as op_bench
import torch

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

embeddingbag_conversion_three_dim_configs = op_bench.cross_product_configs(
    num_embeddings=(80,),
    embedding_dim=(128, 256, 512),
    batch_size=(10,),
    tags=('short',)
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
            "weight": torch.rand(num_embeddings, embedding_dim, dtype=torch.float) + 1
        }
        self.op_func = op_func

    def forward(self, weight):
        return self.op_func(weight)

class EmbeddingBagThreeDimFloatToFusedBase(op_bench.TorchBenchmarkBase):
    def init(self, num_embeddings, embedding_dim, batch_size, op_func):
        self.inputs = {
            "weight": torch.rand(batch_size, num_embeddings, embedding_dim, dtype=torch.float) + 1
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

class EmbeddingBagThreeDimFusedToFloatBase(op_bench.TorchBenchmarkBase):
    def init(self, num_embeddings, embedding_dim, batch_size, op_func):
        weight = torch.randn(batch_size, num_embeddings, embedding_dim + 8, dtype=torch.float)
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
op_bench.generate_pt_tests_from_op_list(conversion_ops,
                                        embeddingbag_conversion_three_dim_configs,
                                        EmbeddingBagThreeDimFloatToFusedBase)
op_bench.generate_pt_tests_from_op_list(unpack_ops,
                                        embeddingbag_conversion_three_dim_configs,
                                        EmbeddingBagThreeDimFusedToFloatBase)

if __name__ == "__main__":
    op_bench.benchmark_runner.main()
