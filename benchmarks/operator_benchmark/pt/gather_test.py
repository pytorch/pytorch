import numpy

import operator_benchmark as op_bench

import torch


"""Microbenchmarks for gather operator."""

# An example input from this configuration is M=4, N=4, dim=0.
gather_configs_short = op_bench.config_list(
    attr_names=["M", "N", "dim"],
    attrs=[
        [256, 512, 0],
        [512, 512, 1],
    ],
    cross_product_configs={
        "device": ["cpu", "cuda"],
    },
    tags=["short"],
)


gather_configs_long = op_bench.cross_product_configs(
    M=[128, 1024], N=[128, 1024], dim=[0, 1], device=["cpu", "cuda"], tags=["long"]
)


class GatherBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, M, N, dim, device):
        min_val = M if dim == 0 else N
        numpy.random.seed((1 << 32) - 1)
        self.inputs = {
            "input_one": torch.rand(M, N, device=device),
            "dim": dim,
            "index": torch.tensor(
                numpy.random.randint(0, min_val, (M, N)), device=device
            ),
        }
        self.set_module_name("gather")

    def forward(self, input_one, dim: int, index):
        return torch.gather(input_one, dim, index)


op_bench.generate_pt_test(gather_configs_short + gather_configs_long, GatherBenchmark)


# Row gathers on CUDA: the index is expanded along the row, so each output row
# is a whole input row. Shapes from a recommendation model, plus a few-rows
# control and a short-row control.
gather_configs_rows = op_bench.config_list(
    attr_names=["src_shape", "num_rows", "slices"],
    attrs=[
        [(17750, 2, 1536), 17750, 2],
        [(142, 1, 1536), 17750, 1],
        [(7500, 2, 1536), 7500, 2],
        [(60, 1, 1536), 7500, 1],
        [(17750, 8192), 17750, 1],
        [(1000, 1536), 1000, 1],
        [(36000, 256), 36000, 1],
    ],
    cross_product_configs={"device": ["cuda"], "dtype": [torch.bfloat16]},
    tags=["row_gather"],
)


class GatherRowsBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, src_shape, num_rows, slices, device, dtype, index_step=1):
        src = torch.randn(src_shape, device=device, dtype=dtype)
        if len(src_shape) == 3:
            index_shape = (num_rows, 1, 1)
            expanded_shape = (num_rows, slices, src_shape[2])
        else:
            index_shape = (num_rows, 1)
            expanded_shape = (num_rows, src_shape[1])
        index_full_shape = (num_rows * index_step, *index_shape[1:])
        index_full = torch.randint(src_shape[0], index_full_shape, device=device)
        index = index_full[::index_step].expand(expanded_shape)
        self.inputs = {"input_one": src, "dim": 0, "index": index}
        self.set_module_name("gather")

    def forward(self, input_one, dim: int, index):
        return torch.gather(input_one, dim, index)


# Row gathers whose index steps by index_step entries along the gathered dim,
# gather(src, 0, idx[::index_step, None].expand(-1, N)), at a recommendation
# model's shapes, plus a contiguous-index, a few-rows and a short-row control.
gather_configs_strided_index = op_bench.config_list(
    attr_names=["src_shape", "num_rows", "slices", "index_step"],
    attrs=[
        [(24582, 1536), 17750, 1, 2],
        [(24582, 1536), 17750, 1, 3],
        [(24582, 8192), 7500, 1, 2],
        [(24582, 1536), 17750, 1, 1],
        [(24582, 1536), 64, 1, 2],
        [(24582, 128), 17750, 1, 2],
    ],
    cross_product_configs={"device": ["cuda"], "dtype": [torch.bfloat16]},
    tags=["strided_index"],
)


op_bench.generate_pt_test(
    gather_configs_rows + gather_configs_strided_index, GatherRowsBenchmark
)


# Gathers along dim 1 that no row gather takes, at fleet shapes. The input rows
# along the last dim start row_stride elements apart (0: one row expanded); a
# 3-D index is constant along the last dim.
gather_configs_elements = op_bench.config_list(
    attr_names=["src_shape", "row_stride", "index_shape", "dtype"],
    attrs=[
        [(362, 2500, 128), 259, (362, 400, 128), torch.bfloat16],
        [(362, 2500, 128), 128, (362, 400, 128), torch.bfloat16],
        [(18, 8684), 0, (18, 8684), torch.long],
        [(71, 36103000), 0, (71, 100), torch.int32],
        [(68, 6008), 0, (68, 32), torch.long],
        [(24, 2400), 2400, (24, 1200), torch.long],
    ],
    cross_product_configs={"device": ["cuda"]},
    tags=["element_gather"],
)


class GatherElementsBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, src_shape, row_stride, index_shape, dtype, device):
        *outer, inner = src_shape
        if row_stride == 0:
            row = torch.randint(1000, (1, inner), device=device, dtype=dtype)
            src = row.expand(src_shape)
        else:
            rows = torch.randint(1000, (*outer, row_stride), device=device, dtype=dtype)
            src = rows[..., :inner]
        if len(index_shape) == 3:
            index = torch.randint(src_shape[1], (*index_shape[:2], 1), device=device)
            index = index.expand(index_shape)
        else:
            index = torch.randint(src_shape[1], index_shape, device=device)
        self.inputs = {"input_one": src, "dim": 1, "index": index}
        self.set_module_name("gather")

    def forward(self, input_one, dim: int, index):
        return torch.gather(input_one, dim, index)


op_bench.generate_pt_test(gather_configs_elements, GatherElementsBenchmark)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
