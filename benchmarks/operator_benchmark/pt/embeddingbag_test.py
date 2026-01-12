import numpy
from pt import configs

import operator_benchmark as op_bench
import torch


"""Embedding and EmbeddingBag Operator Benchmark"""


class EmbeddingBagBenchmark(op_bench.TorchBenchmarkBase):
    def init(
        self,
        embeddingbags,
        dim,
        mode,
        input_size,
        offset,
        sparse,
        include_last_offset,
        device,
    ):
        self.embedding = torch.nn.EmbeddingBag(
            num_embeddings=embeddingbags,
            embedding_dim=dim,
            mode=mode,
            include_last_offset=include_last_offset,
            sparse=sparse,
        ).to(device=device)
        numpy.random.seed((1 << 32) - 1)
        offsets = torch.LongTensor([offset], device=device)
        input = torch.tensor(
            numpy.random.randint(0, embeddingbags, input_size), device=device
        ).long()
        self.inputs = {
            "input": input,
            "offset": torch.cat(
                (offsets, torch.tensor([input.size(0)], dtype=torch.long)), 0
            ),
        }
        self.set_module_name("embeddingbag")

    def forward(self, input, offset):
        return self.embedding(input, offset)


op_bench.generate_pt_test(configs.embeddingbag_short_configs, EmbeddingBagBenchmark)
op_bench.generate_pt_gradient_test(
    configs.embeddingbag_short_configs, EmbeddingBagBenchmark
)


class EmbeddingBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, num_embeddings, embedding_dim, input_size, device):
        self.embedding = torch.nn.Embedding(
            num_embeddings=num_embeddings, embedding_dim=embedding_dim
        ).to(device=device)
        numpy.random.seed((1 << 32) - 1)
        input = torch.tensor(
            numpy.random.randint(0, num_embeddings, input_size), device=device
        ).long()
        self.inputs = {"input": input}
        self.set_module_name("embedding")

    def forward(self, input):
        return self.embedding(input)


op_bench.generate_pt_test(configs.embedding_short_configs, EmbeddingBenchmark)
op_bench.generate_pt_gradient_test(configs.embedding_short_configs, EmbeddingBenchmark)

if __name__ == "__main__":
    op_bench.benchmark_runner.main()
