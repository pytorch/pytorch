import random

import operator_benchmark as op_bench

import torch


"""Microbenchmarks for Cat operator"""

cross_product_configs = {
    "device": ["cpu", "cuda"],
}

# Configs for PT Cat operator
cat_configs_short = op_bench.config_list(
    attr_names=["sizes", "N", "dim"],
    attrs=[
        [(1, 1, 1), 2, 0],  # noqa: E241
        [(512, 512, 2), 2, 1],  # noqa: E241
        [(128, 1024, 2), 2, 1],  # noqa: E241
    ],
    cross_product_configs=cross_product_configs,
    tags=["short"],
)

# Configs specific to static runtime feature - a fast path runtime for pared down models
cat_configs_static_runtime = op_bench.config_list(
    attr_names=["sizes", "N", "dim"],
    attrs=[
        [[(1, 160), (1, 14)], -1, 1],
        [[(1, 20, 40), (1, 4, 40), (1, 5, 40)], -1, 1],
        [[(1, 580), (1, 174)], -1, 1],
        [[(20, 160), (20, 14)], -1, 1],
        [[(20, 20, 40), (20, 4, 40), (20, 5, 40)], -1, 1],
        [[(20, 580), (20, 174)], -1, 1],
    ],
    cross_product_configs=cross_product_configs,
    tags=["static_runtime"],
)

cat_configs_long = op_bench.config_list(
    attr_names=["sizes", "N", "dim"],
    attrs=[
        [(2**10, 2**10, 2), 2, 0],  # noqa: E241
        [(2**10 + 1, 2**10 - 1, 2), 2, 1],  # noqa: E226,E241
        [(2**10, 2**10, 2), 2, 2],  # noqa: E241
        [
            [
                lambda: random.randint(2**6, 2**7),
                2**7 - 17,
                2**6 + 1,
            ],  # noqa: E201,E226,E241
            5,
            0,
        ],
        [
            [
                2**6 + 2**5,
                lambda: random.randint(2**6, 2**7),
                2**6,
            ],  # noqa: E201,E226,E241,E272
            5,
            1,
        ],
        [
            [
                2**7,
                2**6,
                lambda: random.randint(2**6, 2**7),
            ],  # noqa: E201,E241,E272
            5,
            2,
        ],
        [[lambda: random.randint(2**5, 2**6), 2**5, 2**6], 50, 0],  # noqa: E241
        [
            [2**5, lambda: random.randint(2**5, 2**6), 2**6],  # noqa: E241,E272
            50,
            1,
        ],
        [
            [
                2**5 + 1,
                2**6 + 1,
                lambda: random.randint(2**5, 2**6),
            ],  # noqa: E226,E241,E272
            50,
            2,
        ],
    ],
    cross_product_configs=cross_product_configs,
    tags=["long"],
)

# There is a different codepath on CUDA for >4 dimensions
cat_configs_multidim = op_bench.config_list(
    attr_names=["sizes", "N", "dim"],
    attrs=[
        [(2**6, 2**5, 2**2, 2**4, 2**5), 2, 2],  # noqa: E241
        [(2**4, 2**5, 2**2, 2**4, 2**5), 8, 2],  # noqa: E241
        [
            (2**3 + 1, 2**5 - 1, 2**2 + 1, 2**4 - 1, 2**5 + 1),
            17,
            4,
        ],  # noqa: E226,E241
    ],
    cross_product_configs=cross_product_configs,
    tags=["multidim"],
)

cat_configs_manyinputs = op_bench.config_list(
    attr_names=["sizes", "N", "dim"],
    attrs=[
        [[lambda: random.randint(1, 10000)], 100, 0],
        [[lambda: random.randint(1, 1000)], 1000, 0],
        [[lambda: random.randint(1, 500)], 2000, 0],
        [[lambda: random.randint(1, 300)], 3000, 0],
    ],
    cross_product_configs=cross_product_configs,
    tags=["manyinputs"],
)

# Contiguous cats with a slice that is not a multiple of 16 bytes (1 or 2
# elements, 6431 elements, or the 260 elements of [N, 5, 52]), at a
# recommendation model's shapes, plus an aligned control ([N, 64, 52] twice).
# In fp32 the [N, 5, 52] slice is 1040 bytes, a multiple of 16, so that case
# takes the aligned vectorized kernel there.
cat_configs_unaligned = op_bench.config_list(
    attr_names=["sizes", "N", "dim"],
    attrs=[
        [[(6719, 6432), (6719, 1)], -1, 1],
        [[(6561, 4096), (6561, 1)], -1, 1],
        [[(11081, 80, 52), (11081, 5, 52)], -1, 1],
        [[(6719, 6431), (6719, 2)], -1, 1],
        [[(11033, 64, 52), (11033, 64, 52)], -1, 1],
    ],
    cross_product_configs={
        "device": ["cuda"],
        "dtype": [torch.bfloat16, torch.float],
    },
    tags=["unaligned_cat"],
)


# Contiguous bf16 cats on CUDA at a recommendation model's shapes: a KV-cache
# style append along dim 1 and a concatenation along dim 0, plus a cat whose
# slices are not multiples of 16 bytes.
cat_configs_vectorized = op_bench.config_list(
    attr_names=["sizes", "N", "dim"],
    attrs=[
        [[(17750, 2, 1536), (17750, 1, 1536)], -1, 1],
        [[(17750, 1, 1536), (17750, 1, 1536)], -1, 1],
        [[(12500, 2, 1536), (12500, 1, 1536)], -1, 1],
        [[(12500, 1, 1536), (12500, 1, 1536)], -1, 1],
        [[(7500, 2, 1536), (7500, 1, 1536)], -1, 1],
        [[(7500, 1, 1536), (7500, 1, 1536)], -1, 1],
        [[(8192, 1536), (4096, 1536)], -1, 0],
        [[(36257, 256), (36257, 216), (36257, 216), (36257, 15)], -1, 1],
    ],
    cross_product_configs={"device": ["cuda"], "dtype": [torch.bfloat16]},
    tags=["vectorized"],
)

# Non-contiguous cats at a recommendation model's strides, on CUDA. Each input
# is "contig", "narrowP" (the first sizes[dim] of P along dim), "every_other"
# (every other element along dim) or "transposed" (last two dims swapped).
cat_configs_strided = op_bench.config_list(
    attr_names=["sizes", "layouts", "dim"],
    attrs=[
        [[(71, 16, 256), (71, 640, 256)], ["every_other", "contig"], 1],
        [[(96, 16, 256), (96, 640, 256)], ["transposed", "contig"], 1],
        [[(95, 640, 256), (95, 512, 256)], ["narrow656", "contig"], 1],
        [[(16384, 256), (16384, 1024)], ["narrow512", "contig"], 1],
        [[(11033, 32, 52), (11033, 32, 52)], ["narrow64", "narrow64"], 1],
        [[(64, 8), (64, 120)], ["narrow16", "contig"], 1],
    ],
    cross_product_configs={
        "device": ["cuda"],
        "dtype": [torch.bfloat16, torch.float],
    },
    tags=["strided_cat"],
)


class CatBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, sizes, N, dim, device, dtype=torch.float):
        random.seed(42)
        inputs = []
        gen_sizes = []
        if type(sizes) is list and N == -1:
            gen_sizes = sizes
        else:
            for i in range(N):
                gen_sizes.append(
                    [
                        old_size() if callable(old_size) else old_size
                        for old_size in sizes
                    ]
                )

        for s in gen_sizes:
            inputs.append(torch.rand(s, device=device, dtype=dtype))
        result = torch.empty(0, device=device, dtype=dtype)
        self.inputs = {"result": result, "inputs": inputs, "dim": dim}
        self.set_module_name("cat")

    def forward(self, result: torch.Tensor, inputs: list[torch.Tensor], dim: int):
        return torch.cat(inputs, dim=dim, out=result)


class CatStridedBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, sizes, layouts, dim, device, dtype):
        inputs = []
        for size, layout in zip(sizes, layouts):
            if layout == "transposed":
                *batch, rows, cols = size
                t = torch.rand(*batch, cols, rows, device=device, dtype=dtype)
                inputs.append(t.transpose(-1, -2))
                continue
            parent = list(size)
            if layout == "every_other":
                parent[dim] *= 2
            elif layout.startswith("narrow"):
                parent[dim] = int(layout[len("narrow") :])
            t = torch.rand(parent, device=device, dtype=dtype)
            if layout == "every_other":
                t = t[(slice(None),) * dim + (slice(None, None, 2),)]
            inputs.append(t.narrow(dim, 0, size[dim]))
        result = torch.empty(0, device=device, dtype=dtype)
        self.inputs = {"result": result, "inputs": inputs, "dim": dim}
        self.set_module_name("cat")

    def forward(self, result: torch.Tensor, inputs: list[torch.Tensor], dim: int):
        return torch.cat(inputs, dim=dim, out=result)


op_bench.generate_pt_test(
    cat_configs_short
    + cat_configs_long
    + cat_configs_multidim
    + cat_configs_manyinputs
    + cat_configs_static_runtime
    + cat_configs_vectorized
    + cat_configs_unaligned,
    CatBenchmark,
)
op_bench.generate_pt_test(cat_configs_strided, CatStridedBenchmark)

if __name__ == "__main__":
    op_bench.benchmark_runner.main()
