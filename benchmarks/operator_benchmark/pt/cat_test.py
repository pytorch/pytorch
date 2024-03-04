import random
from typing import List

import torch

import operator_benchmark as op_bench


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


class CatBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, sizes, N, dim, device):
        random.seed(42)
        inputs = []
        gen_sizes = []
        if type(sizes) == list and N == -1:
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
            inputs.append(torch.rand(s, device=device))
        result = torch.empty(0, device=device)
        self.inputs = {"result": result, "inputs": inputs, "dim": dim}
        self.set_module_name("cat")

    def forward(self, result: torch.Tensor, inputs: List[torch.Tensor], dim: int):
        return torch.cat(inputs, dim=dim, out=result)


op_bench.generate_pt_test(
    cat_configs_short
    + cat_configs_long
    + cat_configs_multidim
    + cat_configs_manyinputs
    + cat_configs_static_runtime,
    CatBenchmark,
)

if __name__ == "__main__":
    op_bench.benchmark_runner.main()
