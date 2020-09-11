from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import benchmark_fuzz_utils as fuzz_utils
import operator_benchmark as op_bench
import torch


"""Microbenchmarks for point-wise unary operator."""

unary_ops_configs_short = op_bench.config_list(
    attr_names=['X_SIZE'],
    attrs=[[(512, 512)]],
    cross_product_configs={'device': ['cpu', 'cuda']},
    tags=['short']
)

unary_ops_configs_long = op_bench.config_list(
    attr_names=['X_SIZE'],
    attrs=[
        [(256, 256)], [(256, 1024)], [(1024, 256)], [(1024, 1024)]
    ],
    cross_product_configs={'device': ['cpu', 'cuda']},
    tags=['long']
)

unary_ops_fuzzed_configs_short = fuzz_utils.make_fuzzed_config(
    fuzz_utils.Fuzzers.UNARY,
    fuzz_utils.Scale.SMALL,
    n=10,
    seed="UnaryOps",
    cross_product_configs={"device": ["cpu", "cuda"]},
    tags=["short"],
    checksum=548,
)

unary_ops_fuzzed_configs_long = fuzz_utils.make_fuzzed_config(
    fuzz_utils.Fuzzers.UNARY,
    fuzz_utils.CPU_MEDIUM_CUDA_LARGER,
    n=10,
    seed="UnaryOps",
    cross_product_configs={"device": ["cpu", "cuda"]},
    tags=["long"],
    checksum=(3596, 81118319),
)

no_grad_ops = ('argsort', 'clone', 'unique', 'uniform_')

scrutinized_unary_ops_list = op_bench.op_list(
    attr_names=['op_name', 'op_func'],
    attrs=[
        ['argsort', torch.argsort],
        ['clone', torch.clone],
        ['exp', torch.exp],
        ['log', torch.log],
        # ['logit', torch.logit],
        ['reciprocal', torch.reciprocal],
        ['relu', torch.relu],
        ['sqrt', torch.sqrt],
        # ['square', torch.square],
        ['unique', torch.unique],
        ['uniform_', lambda t: t.uniform_()],
    ],
)

other_unary_ops_list = op_bench.op_list(
    attr_names=['op_name', 'op_func'],
    attrs=[
        ['abs', torch.abs],
        ['abs_', torch.abs_],
        ['acos', torch.acos],
        ['acos_', torch.acos_],
        ['asin', torch.asin],
        ['asin_', torch.asin_],
        ['atan', torch.atan],
        ['atan_', torch.atan_],
        ['ceil', torch.ceil],
        ['ceil_', torch.ceil_],
        ['cos', torch.cos],
        ['cos_', torch.cos_],
        ['cosh', torch.cosh],
        ['digamma', torch.digamma],
        ['erf', torch.erf],
        ['erf_', torch.erf_],
        ['erfc', torch.erfc],
        ['erfc_', torch.erfc_],
        ['erfinv', torch.erfinv],
        ['exp_', torch.exp_],
        ['expm1', torch.expm1],
        ['expm1_', torch.expm1_],
        ['floor', torch.floor],
        ['floor_', torch.floor_],
        ['frac', torch.frac],
        ['frac_', torch.frac_],
        ['hardshrink', torch.hardshrink],
        ['lgamma', torch.lgamma],
        ['log10', torch.log10],
        ['log10_', torch.log10_],
        ['log1p', torch.log1p],
        ['log1p_', torch.log1p_],
        ['log2', torch.log2],
        ['log2_', torch.log2_],
        ['log_', torch.log_],
        # ['logit_', torch.logit_],
        ['neg', torch.neg],
        ['neg_', torch.neg_],
        ['reciprocal_', torch.reciprocal_],
        ['relu_', torch.relu_],
        ['round', torch.round],
        ['round_', torch.round_],
        ['rsqrt', torch.rsqrt],
        ['rsqrt_', torch.rsqrt_],
        ['sigmoid', torch.sigmoid],
        ['sigmoid_', torch.sigmoid_],
        ['sign', torch.sign],
        ['sin', torch.sin],
        ['sin_', torch.sin_],
        ['sinh', torch.sinh],
        ['sqrt_', torch.sqrt_],
        # ['square_', torch.square_],
        ['tan', torch.tan],
        ['tan_', torch.tan_],
        ['tanh', torch.tanh],
        ['tanh_', torch.tanh_],
        ['trunc', torch.trunc],
        ['trunc_', torch.trunc_],
        ['zero_', torch.zero_],
        ['bernoulli_', lambda t: t.bernoulli_()],
        ['cauchy_', lambda t: t.cauchy_()],
        ['digamma_', lambda t: t.digamma_()],
        ['exponential_', lambda t: t.exponential_()],
        ['normal_', lambda t: t.normal_()],
        ['random_', lambda t: t.random_()],
        ['sign_', lambda t: t.sign_()],
        ['half', lambda t: t.half()],
        ['long', lambda t: t.long()],
    ],
)

class UnaryOpBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, X_SIZE, device, op_func):
        self.input_one = torch.rand(X_SIZE, device=device, requires_grad=self.auto_set())
        self.op_func = op_func

    def forward(self):
        return self.op_func(self.input_one)

unary_configs = unary_ops_configs_short + unary_ops_configs_long
scrutinized_unary_configs = (
    unary_configs + unary_ops_fuzzed_configs_short +
    unary_ops_fuzzed_configs_long)

op_bench.generate_pt_tests_from_op_list(
    scrutinized_unary_ops_list, scrutinized_unary_configs, UnaryOpBenchmark)
op_bench.generate_pt_gradient_tests_from_op_list(
    [i for i in scrutinized_unary_ops_list if i['op_name'] not in no_grad_ops],
    scrutinized_unary_configs, UnaryOpBenchmark)
op_bench.generate_pt_tests_from_op_list(
    other_unary_ops_list, unary_configs, UnaryOpBenchmark)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
