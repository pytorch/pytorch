import operator_benchmark as op_bench

import torch
from torch._C import dtype


"""Microbenchmarks for add_ operator. Supports both Caffe2/PyTorch."""

# Configs for PT add operator
add_long_configs = op_bench.cross_product_configs(
    M=[256, 1024, 3000],
    N=[512, 4096],
    K=[512, 4096],
    device=["cuda"],
    tags=["long"],
    dtype=[torch.float16, torch.bfloat16, torch.float32],
)


add_short_configs = op_bench.config_list(
    attr_names=["M", "N", "K"],
    attrs=[
        [1, 1, 1],
        [64, 64, 64],
        [64, 64, 128],
    ],
    cross_product_configs={
        "device": ["cpu", "cuda"],
        "dtype": [torch.float],
    },
    tags=["short"],
)


class AddBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, M, N, K, device, dtype):
        self.inputs = {
            "input_one": torch.rand(
                M, N, K, device=device, requires_grad=self.auto_set(), dtype=dtype
            ),
            "input_two": torch.rand(
                M, N, K, device=device, requires_grad=self.auto_set(), dtype=dtype
            ),
        }
        self.set_module_name("add")

    def forward(self, input_one, input_two):
        return torch.add(input_one, input_two)


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


class AddmmBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, M, N, K, device, dtype):
        self.inputs = {
            "input_one": torch.rand(M, K, device=device, requires_grad=self.auto_set(), dtype=dtype),
            "mat1": torch.rand(M, N, device=device, requires_grad=self.auto_set(), dtype=dtype),
            "mat2": torch.rand(N, K, device=device, requires_grad=self.auto_set(), dtype=dtype),
        }
        self.set_module_name("addmm")

    def forward(self, input_one, mat1, mat2):
        return torch.addmm(input_one, mat1, mat2)


op_bench.generate_pt_test(add_long_configs + add_short_configs, AddmmBenchmark)
op_bench.generate_pt_gradient_test(add_long_configs + add_short_configs, AddmmBenchmark)


"""Mircobenchmark for addr operator."""


class AddrBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, M, N, device, dtype):
        self.inputs = {
            "input_one": torch.rand(
                (M, N), device=device, requires_grad=self.auto_set(), dtype=dtype
            ),
            "vec1": torch.rand(
                (M,), device=device, requires_grad=self.auto_set(), dtype=dtype
            ),
            "vec2": torch.rand(
                (N,), device=device, requires_grad=self.auto_set(), dtype=dtype
            ),
        }
        self.set_module_name("addr")

    def forward(self, input_one, vec1, vec2):
        return torch.addr(input_one, vec1, vec2)


addr_configs = op_bench.cross_product_configs(
    M=[8, 256],
    N=[256, 16],
    device=["cpu", "cuda"],
    dtype=[torch.double, torch.half],
    tags=["addr"],
)

op_bench.generate_pt_test(addr_configs, AddrBenchmark)
op_bench.generate_pt_gradient_test(addr_configs, AddrBenchmark)


"""Mircobenchmark for addbmm operator."""


class AddbmmBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, B, M, N, K, device, dtype):
        self.inputs = {
            "input_one": torch.rand(
                (M, N), device=device, requires_grad=self.auto_set(), dtype=dtype
            ),
            "batch1": torch.rand(
                (B, M, K), device=device, requires_grad=self.auto_set(), dtype=dtype
            ),
            "batch2": torch.rand(
                (
                    B,
                    K,
                    N,
                ),
                device=device,
                requires_grad=self.auto_set(),
                dtype=dtype,
            ),
        }
        self.set_module_name("addbmm")

    def forward(self, input_one, batch1, batch2):
        return torch.addbmm(input_one, batch1, batch2)


addbmm_configs = op_bench.cross_product_configs(
    B=[8, 32],
    M=[256, 1024],
    N=[256, 1024],
    K=[64, 128],
    device=["cpu", "cuda"],
    dtype=[torch.float16, torch.bfloat16, torch.float32],
    tags=["addbmm"],
)

op_bench.generate_pt_test(addbmm_configs, AddbmmBenchmark)
op_bench.generate_pt_gradient_test(addbmm_configs, AddbmmBenchmark)

if __name__ == "__main__":
    op_bench.benchmark_runner.main()
