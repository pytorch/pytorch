from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import operator_benchmark as op_bench
import torch


"""Microbenchmarks for Chunk operator"""


# Configs for PT Chunk operator
chunk_short_configs = op_bench.config_list(
    attr_names=["M", "N", "chunks"],
    attrs=[
        [8, 8, 2],
        [256, 512, 2],
        [512, 512, 2],
    ],
    cross_product_configs={
        'device': ['cpu', 'cuda'],
    },
    tags=["short"],
)

chunks_long_configs = op_bench.cross_product_configs(
    M=[128, 1024],
    N=[128, 1024],
    chunks=[2, 4],
    device=['cpu', 'cuda'],
    tags=['long']
)


class ChunkBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, M, N, chunks, device):
        self.input_one = torch.rand(M, N, device=device)
        self.chunks = chunks
        self.set_module_name('chunk')

    def forward(self):
        return torch.chunk(self.input_one, self.chunks)


op_bench.generate_pt_test(chunk_short_configs + chunks_long_configs,
                          ChunkBenchmark)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
