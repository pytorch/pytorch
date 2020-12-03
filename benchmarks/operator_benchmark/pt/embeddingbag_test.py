import operator_benchmark as op_bench
import torch
import numpy
from pt import configs

"""EmbeddingBag Operator Benchmark"""

class EmbeddingBagBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, embeddingbags, dim, mode, input_size, offset, sparse, include_last_offset, device):
        self.embedding = torch.nn.EmbeddingBag(
            num_embeddings=embeddingbags,
            embedding_dim=dim,
            mode=mode,
            include_last_offset=include_last_offset,
            sparse=sparse).to(device=device)
        numpy.random.seed((1 << 32) - 1)
        offsets = torch.LongTensor([offset], device=device)
        input = torch.tensor(numpy.random.randint(0, embeddingbags, input_size), device=device).long()
        self.inputs = {
            "input": input,
            "offset": torch.cat((offsets, torch.tensor([input.size(0)], dtype=torch.long)), 0)
        }
        self.set_module_name('embeddingbag')

    def forward(self, input, offset):
        return self.embedding(input, offset)

op_bench.generate_pt_test(configs.embeddingbag_short_configs, EmbeddingBagBenchmark)
op_bench.generate_pt_gradient_test(configs.embeddingbag_short_configs, EmbeddingBagBenchmark)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
