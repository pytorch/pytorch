import operator_benchmark as op_bench

import torch


"""Microbenchmarks for reductions along the contiguous dim.

Not imported by benchmark_all_other_test.py: the inputs (up to 1.7 GB) are
allocated when the module is loaded, before tag filtering.
"""

# 16-bit rows of 128 to 8192 elements at recommendation-model shapes, a row
# width where the lane count changes (2047), a full reduction, uint8 rows, and
# strided-dim and fp32 controls.
reduce_contiguous_rows_configs = op_bench.config_list(
    attr_names=["reduce_op", "sizes", "dim", "dtype"],
    attrs=[
        ["amax", (17, 2, 5000, 5000), -1, torch.bfloat16],
        ["vector_norm", (1897500, 128), -1, torch.bfloat16],
        ["amax", (262144, 256), -1, torch.bfloat16],
        ["amax", (65536, 1024), -1, torch.bfloat16],
        ["amax", (32768, 2048), -1, torch.bfloat16],
        ["amax", (32768, 2047), -1, torch.bfloat16],
        ["amax", (8000,), -1, torch.bfloat16],
        ["argmax", (8192, 8192), -1, torch.half],
        ["amax", (17, 2, 5000, 5000), -2, torch.bfloat16],
        ["mean", (6733, 1024), -1, torch.float],
        ["amax", (65536, 1024), -1, torch.uint8],
    ],
    cross_product_configs={"device": ["cuda"]},
    tags=["contiguous_rows"],
)

REDUCE_OPS = {
    "amax": torch.amax,
    "argmax": torch.argmax,
    "mean": torch.mean,
    "vector_norm": torch.linalg.vector_norm,
}


class ReduceBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, reduce_op, sizes, dim, dtype, device):
        if dtype.is_floating_point:
            input_tensor = torch.randn(sizes, device=device, dtype=dtype)
        else:
            input_tensor = torch.randint(0, 256, sizes, device=device, dtype=dtype)
        self.inputs = {"input_tensor": input_tensor, "dim": dim}
        self.op_func = REDUCE_OPS[reduce_op]
        self.set_module_name(reduce_op)

    def forward(self, input_tensor, dim: int):
        return self.op_func(input_tensor, dim=dim)


op_bench.generate_pt_test(reduce_contiguous_rows_configs, ReduceBenchmark)

if __name__ == "__main__":
    op_bench.benchmark_runner.main()
