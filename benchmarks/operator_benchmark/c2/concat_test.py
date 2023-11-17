import random

import benchmark_caffe2 as op_bench_c2
from benchmark_caffe2 import Caffe2BenchmarkBase  # noqa: F401
from caffe2.python import core

import operator_benchmark as op_bench


"""Microbenchmarks for Concat operator. Supports both Caffe2/PyTorch."""

cross_product_configs = {
    "device": ["cpu", "cuda"],
    "dtype": ["float"],
    "add_axis": [0],
}

# Configs for C2 concat operator
cat_configs_short = op_bench.config_list(
    attr_names=["sizes", "N", "axis"],
    attrs=[
        [(1, 1, 1), 2, 0],  # noqa: E241
        [(512, 512, 2), 2, 1],  # noqa: E241
        [(128, 1024, 2), 2, 1],  # noqa: E241
    ],
    cross_product_configs=cross_product_configs,
    tags=["short"],
)

# Configs specific to static runtime feature - a fast runtime for pared down models
cat_configs_static_runtime = op_bench.config_list(
    attr_names=["sizes", "N", "axis", "add_axis"],
    attrs=[
        [(1, 40), 5, 1, 1],
        [[(1, 160), (1, 14)], -1, 1, 0],
        [[(1, 20, 40), (1, 4, 40), (1, 5, 40)], -1, 1, 0],
        [[(1, 580), (1, 174)], -1, 1, 0],
        [(20, 40), 5, 1, 1],
        [[(20, 160), (20, 14)], -1, 1, 0],
        [[(20, 20, 40), (20, 4, 40), (20, 5, 40)], -1, 1, 0],
        [[(20, 580), (20, 174)], -1, 1, 0],
    ],
    cross_product_configs=cross_product_configs,
    tags=["static_runtime"],
)

cat_configs_long = op_bench.config_list(
    attr_names=["sizes", "N", "axis"],
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
    attr_names=["sizes", "N", "axis", "dtype"],
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
    attr_names=["sizes", "N", "axis"],
    attrs=[
        [[lambda: random.randint(1, 10000)], 100, 0],
        [[lambda: random.randint(1, 1000)], 1000, 0],
        [[lambda: random.randint(1, 500)], 2000, 0],
        [[lambda: random.randint(1, 300)], 3000, 0],
    ],
    cross_product_configs=cross_product_configs,
    tags=["manyinputs"],
)


class ConcatBenchmark(op_bench_c2.Caffe2BenchmarkBase):
    def init(self, sizes, N, axis, add_axis, dtype, device):
        random.seed(42)
        self.inputs = []
        self.args = {"axis": axis, "add_axis": add_axis}
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
            self.inputs.append(self.tensor(s, dtype, device=device))

        self.output = self.tensor(gen_sizes[0], dtype, device=device)
        self.split_info = self.tensor(gen_sizes[0], "int")
        self.set_module_name("concat")

    def forward(self):
        op = core.CreateOperator(
            "Concat", self.inputs, [self.output, self.split_info], **self.args
        )
        return op


op_bench_c2.generate_c2_test(
    cat_configs_short
    + cat_configs_long
    + cat_configs_multidim
    + cat_configs_manyinputs
    + cat_configs_static_runtime,
    ConcatBenchmark,
)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
