from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import operator_benchmark as op_bench
import torch


"""Microbenchmarks for point-wise unary operator."""


# Configs for pointwise unary ops
unary_ops_configs_short = op_bench.config_list(
    attr_names=['M', 'N'],
    attrs=[
        [512, 512],
    ],
    cross_product_configs={
        'device': ['cpu', 'cuda'],
    },
    tags=['short']
)

unary_ops_configs_long = op_bench.cross_product_configs(
    M=[256, 1024],
    N=[256, 1024],
    device=['cpu', 'cuda'],
    tags=['long']
)

class UnaryOpBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, M, N, device, op_func):
        self.input_one = torch.rand(M, N, device=device)
        self.op_func = op_func

    def forward(self):
        return self.op_func(self.input_one)


unary_ops_list = op_bench.op_list(
    attr_names=['op_name', 'op_func'],
    attrs=[
        ['abs', torch.abs],
        ['abs_', torch.abs_],
        ['acos', torch.acos],
        ['acos_', torch.acos_],
        ['argsort', torch.argsort],
        ['asin', torch.asin],
        ['asin_', torch.asin_],
        ['atan', torch.atan],
        ['atan_', torch.atan_],
        ['ceil', torch.ceil],
        ['ceil_', torch.ceil_],
        ['clone', torch.clone],
        ['cos', torch.cos],
        ['cos_', torch.cos_],
        ['cosh', torch.cosh],
        ['digamma', torch.digamma],
        ['erf', torch.erf],
        ['erf_', torch.erf_],
        ['erfc', torch.erfc],
        ['erfc_', torch.erfc_],
        ['erfinv', torch.erfinv],
        ['exp', torch.exp],
        ['exp_', torch.exp_],
        ['expm1', torch.expm1],
        ['expm1_', torch.expm1_],
        ['floor', torch.floor],
        ['floor_', torch.floor_],
        ['frac', torch.frac],
        ['frac_', torch.frac_],
        ['hardshrink', torch.hardshrink],
        ['lgamma', torch.lgamma],
        ['log', torch.log],
        ['log10', torch.log10],
        ['log10_', torch.log10_],
        ['log1p', torch.log1p],
        ['log1p_', torch.log1p_],
        ['log2', torch.log2],
        ['log2_', torch.log2_],
        ['log_', torch.log_],
        ['neg', torch.neg],
        ['neg_', torch.neg_],
        ['reciprocal', torch.reciprocal],
        ['reciprocal_', torch.reciprocal_],
        ['relu', torch.relu],
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
        ['sqrt', torch.sqrt],
        ['sqrt_', torch.sqrt_],
        ['square', torch.square],
        ['square_', torch.square_],
        ['tan', torch.tan],
        ['tan_', torch.tan_],
        ['tanh', torch.tanh],
        ['tanh_', torch.tanh_],
        ['trunc', torch.trunc],
        ['trunc_', torch.trunc_],
        ['unique', torch.unique],
        ['zero_', torch.zero_],
        ['bernoulli_', lambda t: t.bernoulli_()],
        ['cauchy_', lambda t: t.cauchy_()],
        ['digamma_', lambda t: t.digamma_()],
        ['exponential_', lambda t: t.exponential_()],
        ['normal_', lambda t: t.normal_()],
        ['random_', lambda t: t.random_()],
        ['sign_', lambda t: t.sign_()],
        ['uniform_', lambda t: t.uniform_()],
        ['half', lambda t: t.half()],
        ['long', lambda t: t.long()],
    ],
)


op_bench.generate_pt_tests_from_op_list(unary_ops_list,
                                        unary_ops_configs_short + unary_ops_configs_long,
                                        UnaryOpBenchmark)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
