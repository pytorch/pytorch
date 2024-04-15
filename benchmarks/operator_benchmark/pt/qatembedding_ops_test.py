import numpy
import torch
import torch.ao.nn.qat as nnqat
from pt import configs
from torch.ao.quantization import default_embedding_qat_qconfig

import operator_benchmark as op_bench

"""
Microbenchmarks for QAT Embedding + EmbeddingBag operators.
"""


class QATEmbeddingBagBenchmark(op_bench.TorchBenchmarkBase):
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
        qconfig = default_embedding_qat_qconfig
        self.embedding = nnqat.EmbeddingBag(
            num_embeddings=embeddingbags,
            embedding_dim=dim,
            mode=mode,
            include_last_offset=include_last_offset,
            sparse=sparse,
            device=device,
            qconfig=qconfig,
        )
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
        self.set_module_name("qatEmbeddingBag")

    def forward(self, input, offset):
        return self.embedding(input, offset)


# Currently, EmbeddingBag QAT does not support sparse embeddings.
embeddingbag_short_dense_configs = [
    config
    for config in configs.embeddingbag_short_configs
    if {"sparse": True} not in config
]

op_bench.generate_pt_test(embeddingbag_short_dense_configs, QATEmbeddingBagBenchmark)
op_bench.generate_pt_gradient_test(
    embeddingbag_short_dense_configs, QATEmbeddingBagBenchmark
)


class QATEmbeddingBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, num_embeddings, embedding_dim, input_size, device):
        qconfig = default_embedding_qat_qconfig
        self.embedding = nnqat.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            qconfig=qconfig,
            device=device,
        )
        self.embedding.qconfig = default_embedding_qat_qconfig
        numpy.random.seed((1 << 32) - 1)
        self.input = torch.tensor(
            numpy.random.randint(0, num_embeddings, input_size), device=device
        ).long()
        self.inputs = {"input": self.input}
        self.set_module_name("qatEmbedding")

    def forward(self, input):
        return self.embedding(input)


op_bench.generate_pt_test(configs.embedding_short_configs, QATEmbeddingBenchmark)
op_bench.generate_pt_gradient_test(
    configs.embedding_short_configs, QATEmbeddingBenchmark
)

if __name__ == "__main__":
    op_bench.benchmark_runner.main()
