from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import benchmark_fuzz_utils as fuzz_utils
import operator_benchmark as op_bench
import torch

"""Microbenchmarks for addmm operator"""
mm_short_fuzzed_configs = fuzz_utils.make_fuzzed_config(
    fuzz_utils.Fuzzers.MATMUL,
    fuzz_utils.Scale.SMALL,
    n=10,
    seed="AddMM",
    cross_product_configs={
        "device": ["cpu", "cuda"],
        "trans_x": [False],
        "trans_y": [False],
    },
    tags=["short"],
    checksum=1650,
)

mm_long_fuzzed_configs = fuzz_utils.make_fuzzed_config(
    fuzz_utils.Fuzzers.MATMUL,
    fuzz_utils.CPU_MEDIUM_CUDA_LARGER,
    n=10,
    seed="AddMM",
    cross_product_configs={
        "trans_x": [True, False],
        "trans_y": [True, False],
    },
    tags=["long"],
    checksum=(11136, 30246571),
)

def make_tensor(shape, device, transposed_layout=None):
    if transposed_layout is None:
        return torch.rand(*shape, device=device)

    shape = list(shape)
    d0, d1 = transposed_layout
    shape[d0], shape[d1] = shape[d1], shape[d0]
    return torch.rand(*shape, device=device).transpose(d0, d1)

class AddmmBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, X_SIZE, Y_SIZE, device, trans_x, trans_y):
        self.input_one = torch.rand(X_SIZE[0], Y_SIZE[1], device=device, requires_grad=self.auto_set())
        self.mat1 = make_tensor(X_SIZE, device, (0, 1) if trans_x else None)
        self.mat2 = make_tensor(Y_SIZE, device, (0, 1) if trans_y else None)

        self.mat1.requires_grad_(requires_grad=self.auto_set())
        self.mat2.requires_grad_(requires_grad=self.auto_set())
        self.set_module_name("addmm")

    def forward(self):
        return torch.addmm(self.input_one, self.mat1, self.mat2)

op_bench.generate_pt_test(mm_short_fuzzed_configs + mm_long_fuzzed_configs, AddmmBenchmark)
op_bench.generate_pt_gradient_test(mm_short_fuzzed_configs + mm_long_fuzzed_configs, AddmmBenchmark)


"""Microbenchmarks for MatMul operator"""
class MatMulBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, X_SIZE, Y_SIZE, device, trans_x, trans_y):
        self.mat1 = make_tensor(X_SIZE, device, (0, 1) if trans_x else None)
        self.mat2 = make_tensor(Y_SIZE, device, (0, 1) if trans_y else None)

        self.mat1.requires_grad_(requires_grad=self.auto_set())
        self.mat2.requires_grad_(requires_grad=self.auto_set())
        self.set_module_name("matmul")

    def forward(self):
        return torch.matmul(self.mat1, self.mat2)

# matmul will generally follow the same path as addmm, so a restricted set is
# tested as a sanity check.
op_bench.generate_pt_test(mm_short_fuzzed_configs[:2], MatMulBenchmark)
op_bench.generate_pt_gradient_test(mm_short_fuzzed_configs[:2], MatMulBenchmark)


"""Mircobenchmark for addbmm operator."""
baddmm_short_fuzzed_configs = fuzz_utils.make_fuzzed_config(
    fuzz_utils.Fuzzers.BATCH_MATMUL,
    fuzz_utils.Scale.SMALL,
    n=10,
    seed="BAddMM",
    cross_product_configs={
        "device": ["cpu", "cuda"],
        "trans_x": [False],
        "trans_y": [False],
    },
    tags=["short"],
    checksum=1213,
)

baddmm_long_fuzzed_configs = fuzz_utils.make_fuzzed_config(
    fuzz_utils.Fuzzers.BATCH_MATMUL,
    fuzz_utils.CPU_MEDIUM_CUDA_LARGER,
    n=10,
    seed="BAddMM",
    cross_product_configs={
        "device": ["cpu", "cuda"],
        "trans_x": [True, False],
        "trans_y": [True, False],
    },
    tags=["long"],
    checksum=(4197, 16841683),
)

class AddbmmBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, X_SIZE, Y_SIZE, device, trans_x, trans_y):
        self.input_one = torch.rand((X_SIZE[1], Y_SIZE[2]), device=device, requires_grad=self.auto_set())
        self.batch1 = make_tensor(X_SIZE, device, (1, 2) if trans_x else None)
        self.batch2 = make_tensor(Y_SIZE, device, (1, 2) if trans_y else None)

        self.batch1.requires_grad_(requires_grad=self.auto_set())
        self.batch2.requires_grad_(requires_grad=self.auto_set())
        self.set_module_name("addbmm")

    def forward(self):
        return torch.addbmm(self.input_one, self.batch1, self.batch2)

op_bench.generate_pt_test(baddmm_short_fuzzed_configs + baddmm_long_fuzzed_configs, AddbmmBenchmark)
op_bench.generate_pt_gradient_test(baddmm_short_fuzzed_configs + baddmm_long_fuzzed_configs, AddbmmBenchmark)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
