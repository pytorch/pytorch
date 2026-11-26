import numpy

import operator_benchmark as op_bench

import torch


"""Microbenchmarks for index_select operator."""

# An example input from this configuration is M=4, N=4, dim=0.
index_select_configs_short = op_bench.config_list(
    attr_names=["M", "N", "K", "dim"],
    attrs=[
        [8, 8, 1, 1],
        [256, 512, 1, 1],
        [512, 512, 1, 1],
        [8, 8, 2, 1],
        [256, 512, 2, 1],
        [512, 512, 2, 1],
    ],
    cross_product_configs={
        "device": ["cpu", "cuda"],
    },
    tags=["short"],
)


index_select_configs_long = op_bench.cross_product_configs(
    M=[128, 1024],
    N=[128, 1024],
    K=[1, 2],
    dim=[1],
    device=["cpu", "cuda"],
    tags=["long"],
)


class IndexSelectBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, M, N, K, dim, device):
        max_val = N
        numpy.random.seed((1 << 32) - 1)
        index_dim = numpy.random.randint(0, N)
        self.inputs = {
            "input_one": torch.rand(M, N, K, device=device),
            "dim": dim,
            "index": torch.tensor(
                numpy.random.randint(0, max_val, index_dim), device=device
            ),
        }
        self.set_module_name("index_select")

    def forward(self, input_one, dim, index):
        return torch.index_select(input_one, dim, index)


op_bench.generate_pt_test(
    index_select_configs_short + index_select_configs_long, IndexSelectBenchmark
)


# aten::index with one indexed dim, `x[idx]`, on CUDA at a recommendation
# model's shapes; it copies whole rows like a row gather. There is no aten::index
# benchmark, so it sits next to index_select.
index_configs_rows = op_bench.config_list(
    attr_names=["M", "N", "num_rows"],
    attrs=[
        [24582, 1536, 17750],
        [24582, 1536, 7500],
    ],
    cross_product_configs={"device": ["cuda"], "dtype": [torch.bfloat16]},
    tags=["row_gather"],
)


class IndexBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, M, N, num_rows, device, dtype):
        self.inputs = {
            "input_one": torch.randn(M, N, device=device, dtype=dtype),
            "index": torch.randint(M, (num_rows,), device=device),
        }
        self.set_module_name("index")

    def forward(self, input_one, index):
        return input_one[index]


op_bench.generate_pt_test(index_configs_rows, IndexBenchmark)


# aten::index with one indexed dim whose rows the vectorized row gather does not
# take: one-element rows, 1-D inputs and an indexed inner dim, at fleet shapes.
# The (300000000, 1) tables take 2.4 GB (int64) and 1.2 GB (int32) of device
# memory, and benchmark_core builds the inputs of every config before filtering
# by tag, so running this file on CUDA needs that memory free.
index_configs_elements = op_bench.config_list(
    attr_names=["input_shape", "num_indices", "dim", "dtype"],
    attrs=[
        [(300000000, 1), 5362, 0, torch.long],
        [(300000000, 1), 5362, 0, torch.int32],
        [(300000000, 1), 1040, 0, torch.int32],
        [(36103000,), 7000, 0, torch.long],
        [(864000,), 57600, 0, torch.long],
        [(3456001,), 3456000, 0, torch.long],
        [(864000, 256), 150, 1, torch.int8],
    ],
    cross_product_configs={"device": ["cuda"]},
    tags=["element_gather"],
)


class IndexElementsBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, input_shape, num_indices, dim, dtype, device):
        index = torch.randint(input_shape[dim], (num_indices,), device=device)
        input_one = torch.randint(-100, 100, input_shape, device=device, dtype=dtype)
        self.inputs = {
            "input_one": input_one,
            "indices": [None] * dim + [index],
        }
        self.set_module_name("index")

    def forward(self, input_one, indices):
        return torch.ops.aten.index.Tensor(input_one, indices)


op_bench.generate_pt_test(index_configs_elements, IndexElementsBenchmark)


# 1-D index_select on CUDA at a fleet shape; it runs as a gather.
index_select_configs_1d = op_bench.config_list(
    attr_names=["numel", "num_indices"],
    attrs=[
        [3456001, 3456000],
    ],
    cross_product_configs={"device": ["cuda"], "dtype": [torch.long]},
    tags=["element_gather"],
)


class IndexSelect1DBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, numel, num_indices, device, dtype):
        self.inputs = {
            "input_one": torch.randint(1000, (numel,), device=device, dtype=dtype),
            "dim": 0,
            "index": torch.randint(numel, (num_indices,), device=device),
        }
        self.set_module_name("index_select")

    def forward(self, input_one, dim: int, index):
        return torch.index_select(input_one, dim, index)


op_bench.generate_pt_test(index_select_configs_1d, IndexSelect1DBenchmark)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
