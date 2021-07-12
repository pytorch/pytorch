
import operator_benchmark as op_bench
import torch
import torch.nn.quantized as nnq
import numpy
from pt import configs

"""
Microbenchmarks for qEmbeddingBag operators.
"""

class QEmbeddingBagBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, embeddingbags, dim, mode, input_size, offset, sparse, include_last_offset, device):
        self.embedding = nnq.EmbeddingBag(
            num_embeddings=embeddingbags,
            embedding_dim=dim,
            mode=mode,
            include_last_offset=include_last_offset).to(device=device)
        numpy.random.seed((1 << 32) - 1)
        self.input = torch.tensor(numpy.random.randint(0, embeddingbags, input_size), device=device).long()
        offset = torch.LongTensor([offset], device=device)
        self.offset = torch.cat((offset, torch.tensor([self.input.size(0)], dtype=torch.long)), 0)
        self.inputs = {
            "input": self.input,
            "offset": self.offset
        }
        self.set_module_name('qEmbeddingBag')

    def forward(self, input, offset):
        return self.embedding(input, offset)


op_bench.generate_pt_test(configs.embeddingbag_short_configs, QEmbeddingBagBenchmark)

if __name__ == "__main__":
    op_bench.benchmark_runner.main()
