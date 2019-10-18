from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import operator_benchmark as op_bench
import torch
import numpy


"""Microbenchmarks for gather operator."""

# An example input from this configuration is M=4, N=4, dim=0.
configs = op_bench.config_list(
    attrs=[
        [256, 512, 0],
        [512, 512, 1],
    ],
    attr_names=["M", "N", "dim"],
    tags=["short"]
)


class GatherBenchmark(op_bench.TorchBenchmarkBase):
    # TODO (mingzhe0908): should we have a global seed for all ops?
    def init(self, M, N, dim):
        self.input_one = torch.rand(M, N)
        self.dim = dim
        min_val = M if dim == 0 else N
        numpy.random.seed((1 << 32) - 1)
        self.index = torch.tensor(numpy.random.randint(0, min_val, (M, N)))
        self.set_module_name("gather")

    def forward(self):
        return torch.gather(self.input_one, self.dim, self.index)


op_bench.generate_pt_test(configs, GatherBenchmark)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
