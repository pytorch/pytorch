import operator_benchmark as op_bench

import torch


"""Microbenchmarks for boolean operators. Supports both Caffe2/PyTorch."""

# Configs for PT all operator
all_long_configs = op_bench.cross_product_configs(
    M=[8, 128], N=[32, 64], K=[256, 512], device=["cpu", "cuda"], tags=["long"]
)


all_short_configs = op_bench.config_list(
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


class AllBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, M, N, K, device):
        self.inputs = {
            "input_one": torch.randint(0, 2, (M, N, K), device=device, dtype=torch.bool)
        }
        self.set_module_name("all")

    def forward(self, input_one):
        return torch.all(input_one)


# The generated test names based on all_short_configs will be in the following pattern:
# all_M8_N16_K32_devicecpu
# all_M8_N16_K32_devicecpu_bwdall
# all_M8_N16_K32_devicecpu_bwd1
# all_M8_N16_K32_devicecpu_bwd2
# ...
# Those names can be used to filter tests.

op_bench.generate_pt_test(all_long_configs + all_short_configs, AllBenchmark)

"""Mircobenchmark for any operator."""


class AnyBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, M, N, device):
        self.inputs = {
            "input_one": torch.randint(0, 2, (M, N), device=device, dtype=torch.bool)
        }
        self.set_module_name("any")

    def forward(self, input_one):
        return torch.any(input_one)


any_configs = op_bench.cross_product_configs(
    M=[8, 256],
    N=[256, 16],
    device=["cpu", "cuda"],
    tags=["any"],
)

op_bench.generate_pt_test(any_configs, AnyBenchmark)

"""Microbenchmark for any and all along one dim."""

any_all_dim_ops_list = op_bench.op_list(
    attr_names=["op_name", "op_func"],
    attrs=[["any", torch.any], ["all", torch.all]],
)

# Reductions along the contiguous dim on CUDA at a recommendation model's mask
# shapes and at 128- and 200-element rows, in bool and bf16, plus a reduction
# along the strided dim.
any_all_dim_configs = op_bench.config_list(
    attr_names=["shape", "dim", "dtype"],
    attrs=[
        [(28160, 8192), -1, torch.bool],
        [(22440, 8192), -1, torch.bool],
        [(11440, 8192), -1, torch.bool],
        [(16640, 1280), -1, torch.bool],
        [(64, 128, 256), -1, torch.bool],
        [(262144, 200), -1, torch.bool],
        [(262144, 128), -1, torch.bool],
        [(28160, 8192), 0, torch.bool],
        [(16640, 1280), -1, torch.bfloat16],
        [(28160, 8192), -1, torch.bfloat16],
        [(262144, 128), -1, torch.bfloat16],
    ],
    cross_product_configs={"device": ["cuda"]},
    tags=["contiguous_dim"],
)


class AnyAllDimBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, shape, dim, dtype, device, op_func):
        # The input is created on first use: every config is built before the
        # tag and device filters run, and these inputs take up to 460 MB.
        self.inputs = {}
        self.input_args = (shape, device, dtype)
        self.dim = dim
        self.op_func = op_func

    def get_inputs(self):
        if not self.inputs:
            shape, device, dtype = self.input_args
            self.inputs = {
                "input_one": torch.randint(0, 2, shape, device=device, dtype=dtype)
            }
            self.extract_inputs_tuple()
        return self.inputs_tuple

    def forward(self, input_one):
        return self.op_func(input_one, self.dim)


op_bench.generate_pt_tests_from_op_list(
    any_all_dim_ops_list, any_all_dim_configs, AnyAllDimBenchmark
)

if __name__ == "__main__":
    op_bench.benchmark_runner.main()
