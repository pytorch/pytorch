import operator_benchmark as op_bench

import torch


"""Microbenchmarks for add_ operator. Supports both Caffe2/PyTorch."""

# Configs for PT add operator
add_long_configs = op_bench.cross_product_configs(
    M=[8, 128], N=[32, 64], K=[256, 512], device=["cpu", "cuda"], tags=["long"]
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
    },
    tags=["short"],
)


class AddBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, M, N, K, device):
        self.inputs = {
            "input_one": torch.rand(
                M, N, K, device=device, requires_grad=self.auto_set()
            ),
            "input_two": torch.rand(
                M, N, K, device=device, requires_grad=self.auto_set()
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

if __name__ == "__main__":
    op_bench.benchmark_runner.main()
