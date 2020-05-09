import operator_benchmark as op_bench
import torch
import numpy

"""EmbeddingBag Operator Benchmark"""

embeddingbag_short_configs = op_bench.cross_product_configs(
    embeddingbags=[80, 120, 1000, 2300],
    dim=[64],
    mode=['sum'],
    input_size=[8, 16, 64],
    offset=[0],
    sparse=[True],
    device=['cpu'],
    tags=['short']
)


class EmbeddingBagBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, embeddingbags, dim, mode, input_size, offset, sparse, device):
        self.embegging = torch.nn.EmbeddingBag(
            num_embeddings=embeddingbags,
            embedding_dim=dim,
            mode=mode,
            sparse=sparse).to(device=device)
        numpy.random.seed((1 << 32) - 1)
        self.input = torch.tensor(numpy.random.randint(0, embeddingbags, input_size), device=device).long()
        self.offset = torch.LongTensor([offset], device=device)

        self.set_module_name('embeddingbag')

    def forward(self):
        return self.embegging(self.input, self.offset)


op_bench.generate_pt_test(embeddingbag_short_configs, EmbeddingBagBenchmark)
op_bench.generate_pt_gradient_test(embeddingbag_short_configs, EmbeddingBagBenchmark)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
