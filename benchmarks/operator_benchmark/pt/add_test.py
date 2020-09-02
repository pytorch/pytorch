from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import benchmark_fuzz_utils as fuzz_utils
import operator_benchmark as op_bench
import torch


"""Microbenchmarks for add_ operator. Supports both Caffe2/PyTorch."""
# There are two sets of add benchmarks. The first (this file) follows the
# README and demonstrates explicit configurations. The second is in
# binary_test.py and tests a broader range of behavior. (broadcasting,
# type promotion, etc.)

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


class AddBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, M, N, K, device):
        self.input_one = torch.rand(M, N, K, device=device, requires_grad=self.auto_set())
        self.input_two = torch.rand(M, N, K, device=device, requires_grad=self.auto_set())
        self.set_module_name("add")

    def forward(self):
        return torch.add(self.input_one, self.input_two)

# The generated test names based on add_short_configs will be in the following pattern:
# add_M8_N16_K32_devicecpu
# add_M8_N16_K32_devicecpu_bwdall
# add_M8_N16_K32_devicecpu_bwd1
# add_M8_N16_K32_devicecpu_bwd2
# ...
# Those names can be used to filter tests.

op_bench.generate_pt_test(add_long_configs + add_short_configs, AddBenchmark)
op_bench.generate_pt_gradient_test(add_long_configs + add_short_configs, AddBenchmark)


"""Mircobenchmark for addmm operator."""
addmm_short_fuzzed_configs = fuzz_utils.make_fuzzed_config(
    fuzz_utils.Fuzzers.MATMUL,
    fuzz_utils.Scale.SMALL,
    n=10,
    seed="AddMM",
    cross_product_configs={
        'device': ['cpu', 'cuda'],
    },
    tags=["short"],
    checksum=5944,
)

addmm_long_fuzzed_configs = fuzz_utils.make_fuzzed_config(
    fuzz_utils.Fuzzers.MATMUL,
    fuzz_utils.Scale.MEDIUM,
    n=10,
    seed="AddMM",
    cross_product_configs={
        'device': ['cpu', 'cuda'],
    },
    tags=["long"],
    checksum=11136,
)

class AddmmBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, X_SIZE, Y_SIZE, device):
        self.input_one = torch.rand(X_SIZE[0], Y_SIZE[1], device=device, requires_grad=self.auto_set())
        self.mat1 = torch.rand(*X_SIZE, device=device, requires_grad=self.auto_set())
        self.mat2 = torch.rand(*Y_SIZE, device=device, requires_grad=self.auto_set())
        self.set_module_name("addmm")

    def forward(self):
        return torch.addmm(self.input_one, self.mat1, self.mat2)

op_bench.generate_pt_test(addmm_short_fuzzed_configs + addmm_long_fuzzed_configs, AddmmBenchmark)
op_bench.generate_pt_gradient_test(addmm_short_fuzzed_configs + addmm_long_fuzzed_configs, AddmmBenchmark)


"""Mircobenchmark for addr operator."""
def make_addr_configs(device, dtypes):
    return fuzz_utils.make_fuzzed_config(
        fuzz_utils.Fuzzers.UNARY,
        fuzz_utils.Scale.SMALL,
        n=10,
        fuzzer_kwargs={"dim": 2},
        seed="AddR",
        cross_product_configs={
            'device': [device],
            'dtype': dtypes,
        },
        tags=["short"],
        checksum=636,
    )

addr_fuzzed_configs = (
    # CPU does not support fp16.
    make_addr_configs('cpu', [torch.double]) +
    make_addr_configs('cuda', [torch.double, torch.half])
)


class AddrBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, X_SIZE, device, dtype):
        self.input_one = torch.rand(X_SIZE, device=device, requires_grad=self.auto_set(), dtype=dtype)
        self.vec1 = torch.rand((X_SIZE[0],), device=device, requires_grad=self.auto_set(), dtype=dtype)
        self.vec2 = torch.rand((X_SIZE[1],), device=device, requires_grad=self.auto_set(), dtype=dtype)
        self.set_module_name("addr")

    def forward(self):
        return torch.addr(self.input_one, self.vec1, self.vec2)

op_bench.generate_pt_test(addr_fuzzed_configs, AddrBenchmark)
op_bench.generate_pt_gradient_test(addr_fuzzed_configs, AddrBenchmark)


"""Mircobenchmark for addbmm operator."""
baddmm_short_fuzzed_configs = fuzz_utils.make_fuzzed_config(
    fuzz_utils.Fuzzers.BATCH_MATMUL,
    fuzz_utils.Scale.SMALL,
    n=10,
    seed="BAddMM",
    cross_product_configs={
        'device': ['cpu', 'cuda'],
    },
    tags=["short"],
    checksum=1685,
)

baddmm_long_fuzzed_configs = fuzz_utils.make_fuzzed_config(
    fuzz_utils.Fuzzers.BATCH_MATMUL,
    fuzz_utils.Scale.MEDIUM,
    n=10,
    seed="BAddMM",
    cross_product_configs={
        'device': ['cpu', 'cuda'],
    },
    tags=["long"],
    checksum=4466,
)

class AddbmmBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, X_SIZE, Y_SIZE, device):
        self.input_one = torch.rand((X_SIZE[1], Y_SIZE[2]), device=device, requires_grad=self.auto_set())
        self.batch1 = torch.rand(X_SIZE, device=device, requires_grad=self.auto_set())
        self.batch2 = torch.rand(Y_SIZE, device=device, requires_grad=self.auto_set())
        self.set_module_name("addbmm")

    def forward(self):
        return torch.addbmm(self.input_one, self.batch1, self.batch2)

op_bench.generate_pt_test(baddmm_short_fuzzed_configs + baddmm_long_fuzzed_configs, AddbmmBenchmark)
op_bench.generate_pt_gradient_test(baddmm_short_fuzzed_configs + baddmm_long_fuzzed_configs, AddbmmBenchmark)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
