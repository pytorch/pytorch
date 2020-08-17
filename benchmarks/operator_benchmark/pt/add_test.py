from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import operator_benchmark as op_bench
import torch

from torch.utils._benchmark.op_fuzzers import binary

"""Microbenchmarks for add_ operator. Supports both Caffe2/PyTorch."""


def make_fuzzed_config(
    scale=binary.SMALL,
    n: int = 10,
    seed: int = 0,
    cross_product_configs=None,
    tags=None,
    checksum=None
):
    fuzzer = binary.BinaryOpFuzzer(seed=seed, scale=scale)
    attr_names = [binary.X_SIZE, binary.Y_SIZE]
    attrs = []
    for i in range(n):
        params = fuzzer.structure_params(fuzzer.params[i])
        attrs.append([params[a] for a in attr_names])

    # Because the generated tests depend on Fuzzer for random numbers,
    # it is advisable to use a checksum to ensure that the configurations
    # being benchmarked do not silently change.
    if checksum is not None:
        total = 0
        for a in attrs:
            total += sum(sum(i) for i in a)
        if total != checksum:
            raise ValueError(f"Checksum failed: Total {total} != {checksum}")

    return op_bench.config_list(
        attr_names=[a.upper() for a in attr_names],
        attrs=attrs,
        cross_product_configs=cross_product_configs or {},
        tags=tags or [],
    )


add_short_fuzzed_configs = make_fuzzed_config(
    binary.SMALL, n=10, seed=0,
    cross_product_configs={
        'device': ['cpu', 'cuda'],
    },
    tags=["short"],
    checksum=1381,
)

add_long_fuzzed_configs = make_fuzzed_config(
    binary.MEDIUM, n=10, seed=0,
    cross_product_configs={
        'device': ['cpu', 'cuda'],
    },
    tags=["long"],
    checksum=9198,
)


class AddBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, X_SIZE, Y_SIZE, device):
        self.input_one = torch.rand(*X_SIZE, device=device, requires_grad=self.auto_set())
        self.input_two = torch.rand(*Y_SIZE, device=device, requires_grad=self.auto_set())
        self.set_module_name("add")

    def forward(self):
        return torch.add(self.input_one, self.input_two)

# The generated test names based on add_short_fuzzed_configs will be in the following pattern:
# add_X_SIZE(128,)_Y_SIZE(1,)_cpu
# add_X_SIZE(128,)_Y_SIZE(1,)_cpu_bwdall
# add_X_SIZE(128,)_Y_SIZE(1,)_cpu_bwd1
# add_X_SIZE(128,)_Y_SIZE(1,)_cpu_bwd2
# ...
# Those names can be used to filter tests.

op_bench.generate_pt_test(add_short_fuzzed_configs + add_long_fuzzed_configs, AddBenchmark)
op_bench.generate_pt_gradient_test(add_short_fuzzed_configs + add_long_fuzzed_configs, AddBenchmark)


# Configs for PT add operator
add_long_configs = op_bench.cross_product_configs(
    M=[8, 128],
    N=[32, 64],
    K=[256, 512],
    device=['cpu', 'cuda'],
    tags=["long"]
)


add_short_configs = op_bench.config_list(
    attr_names=["M", "N", "K"],
    attrs=[
        [1, 1, 1],
        [64, 64, 64],
        [64, 64, 128],
    ],
    cross_product_configs={
        'device': ['cpu', 'cuda'],
    },
    tags=["short"],
)

"""Mircobenchmark for addmm operator."""


class AddmmBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, M, N, K, device):
        self.input_one = torch.rand(M, K, device=device, requires_grad=self.auto_set())
        self.mat1 = torch.rand(M, N, device=device, requires_grad=self.auto_set())
        self.mat2 = torch.rand(N, K, device=device, requires_grad=self.auto_set())
        self.set_module_name("addmm")

    def forward(self):
        return torch.addmm(self.input_one, self.mat1, self.mat2)

op_bench.generate_pt_test(add_long_configs + add_short_configs, AddmmBenchmark)
op_bench.generate_pt_gradient_test(add_long_configs + add_short_configs, AddmmBenchmark)


"""Mircobenchmark for addr operator."""


class AddrBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, M, N, device, dtype):
        self.input_one = torch.rand((M, N), device=device, requires_grad=self.auto_set(), dtype=dtype)
        self.vec1 = torch.rand((M,), device=device, requires_grad=self.auto_set(), dtype=dtype)
        self.vec2 = torch.rand((N,), device=device, requires_grad=self.auto_set(), dtype=dtype)
        self.set_module_name("addr")

    def forward(self):
        return torch.addr(self.input_one, self.vec1, self.vec2)

addr_configs = op_bench.cross_product_configs(
    M=[8, 256],
    N=[256, 16],
    device=['cpu', 'cuda'],
    dtype=[torch.double, torch.half],
    tags=["addr"],
)

op_bench.generate_pt_test(addr_configs, AddrBenchmark)
op_bench.generate_pt_gradient_test(addr_configs, AddrBenchmark)


"""Mircobenchmark for addbmm operator."""


class AddbmmBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, B, M, N, K, device):
        self.input_one = torch.rand((M, N), device=device, requires_grad=self.auto_set())
        self.batch1 = torch.rand((B, M, K), device=device, requires_grad=self.auto_set())
        self.batch2 = torch.rand((B, K, N,), device=device, requires_grad=self.auto_set())
        self.set_module_name("addbmm")

    def forward(self):
        return torch.addbmm(self.input_one, self.batch1, self.batch2)

addbmm_configs = op_bench.cross_product_configs(
    B=[2, 100],
    M=[8, 256],
    N=[256, 16],
    K=[15, 16],
    device=['cpu', 'cuda'],
    tags=["addbmm"],
)

op_bench.generate_pt_test(addbmm_configs, AddbmmBenchmark)
op_bench.generate_pt_gradient_test(addbmm_configs, AddbmmBenchmark)

if __name__ == "__main__":
    op_bench.benchmark_runner.main()
