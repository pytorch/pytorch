from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import operator_benchmark as op_bench
import torch


"""Microbenchmarks for Chunk operator"""


# Configs for PT Chunk operator
chunks_short_configs = op_bench.cross_product_configs(
    M=[64, 128],
    N=[64],
    chunks=[3],
    tags=['short']
)


class ChunkBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, M, N, chunks): 
        self.input_one = torch.rand(M, N) 
        self.chunks = chunks
        self.set_module_name('chunks')

    def forward(self):
        return torch.chunk(self.input_one, self.chunks)


op_bench.generate_pt_test(chunks_short_configs, ChunkBenchmark)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
