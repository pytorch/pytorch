import operator_benchmark as op_bench

import torch


"""Microbenchmarks for topk operator"""


topk_configs_short = op_bench.config_list(
    attr_names=["shape", "k", "dim"],
    attrs=[
        [(16, 4), 4, 1],
        [(1024 * 1024,), 16, 0],
    ],
    cross_product_configs={"device": ["cpu"], "dtype": [torch.float]},
    tags=["short"],
)

topk_configs_long = op_bench.cross_product_configs(
    shape=[(64, 2), (1024 * 1024,), (128,)],
    k=[1, 2, 4, 16, 32],
    dim=[0],
    device=["cpu", "cuda"],
    dtype=[torch.float, torch.bfloat16],
    tags=["long"],
)


# Many slices of up to 512 elements on CUDA, at a recommendation model's shapes,
# plus slices that take the single-block, warp sort or multi-block paths on
# every GPU, and longer ones for gfx1250's wave select and lane-private counts.
topk_configs_warp_sort = op_bench.config_list(
    attr_names=["shape", "k", "dim", "dtype", "sorted"],
    attrs=[
        [(17750, 512), 2, 1, torch.float, False],
        [(12500, 512), 2, 1, torch.float, False],
        [(7500, 512), 2, 1, torch.float, False],
        [(17750, 416), 2, 1, torch.float, False],
        [(17750, 512), 16, 1, torch.float, True],
        [(17750, 512), 2, 1, torch.bfloat16, False],
        [(17750, 384), 2, 1, torch.float, False],
        [(17750, 256), 2, 1, torch.float, False],
        [(17750, 1024), 2, 1, torch.float, False],
        [(1000, 512), 2, 1, torch.float, False],
        [(8000, 4096), 16, 1, torch.float, False],
        [(17750, 1024), 50, 1, torch.float, False],
    ],
    cross_product_configs={"device": ["cuda"]},
    tags=["warp_sort"],
)

# A few long slices on CUDA, at a recommendation model's beam search shapes,
# plus shorter slices whose kernels do not change.
topk_configs_long_slices = op_bench.config_list(
    attr_names=["shape", "k", "dim", "dtype", "sorted"],
    attrs=[
        [(64, 307200), 240, 1, torch.bfloat16, True],
        [(64, 307200), 240, 1, torch.bfloat16, False],
        [(51, 307200), 240, 1, torch.bfloat16, True],
        [(26, 307200), 240, 1, torch.bfloat16, True],
        [(1280, 81920), 20, 1, torch.bfloat16, False],
        [(64, 307200), 240, 1, torch.float, True],
        [(64, 1280), 240, 1, torch.bfloat16, True],
        [(1280, 1280), 10, 1, torch.bfloat16, True],
        [(17750, 512), 50, 1, torch.bfloat16, True],
    ],
    cross_product_configs={"device": ["cuda"]},
    tags=["long_slices"],
)

# Long slices on CUDA whose multi-block topk gathers many items per block, at
# recommendation models' shapes.
topk_configs_multiblock_gather = op_bench.config_list(
    attr_names=["shape", "k", "dim", "dtype", "sorted"],
    attrs=[
        [(81, 1216039), 100, 1, torch.half, True],
        [(730, 150780), 2500, 1, torch.half, False],
        [(81, 6008), 100, 1, torch.bfloat16, True],
        [(86, 698454), 100, 1, torch.half, True],
        [(759, 54568), 2500, 1, torch.bfloat16, False],
        [(720, 6842), 100, 1, torch.half, True],
    ],
    cross_product_configs={"device": ["cuda"]},
    tags=["multiblock_gather"],
)


class TopkBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, shape, k, dim, dtype, device, sorted=True):
        self.inputs = {
            "input": torch.randn(shape, device=device, dtype=dtype),
            "k": k,
            "dim": dim,
            "sorted": sorted,
        }

        self.set_module_name("topk")

    def forward(self, input, k, dim, sorted):
        return torch.topk(input, k=k, dim=dim, sorted=sorted)


op_bench.generate_pt_test(
    topk_configs_short
    + topk_configs_long
    + topk_configs_warp_sort
    + topk_configs_long_slices
    + topk_configs_multiblock_gather,
    TopkBenchmark,
)

if __name__ == "__main__":
    op_bench.benchmark_runner.main()
