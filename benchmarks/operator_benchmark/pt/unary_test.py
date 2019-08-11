from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import operator_benchmark as op_bench
import torch


"""Microbenchmarks for point-wise unary operator."""


# Configs for pointwise unary ops
unary_ops_configs = op_bench.config_list(
    attrs=[
        [128, 128],
        [256, 256],
        [1024, 1024],
    ],
    attr_names=["M", "N"],
    tags=["short"]
)


class UnaryOpBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, M, N, op_func): 
        self.input_one = torch.rand(M, N)
        self.op_func = op_func

    def forward(self):
        return self.op_func(self.input_one)


unary_ops_list = op_bench.op_list(
    attr_names=["op_name", "op_func"],
    attrs=[
        ["abs", torch.abs],
        ["abs_", torch.abs_],
        ["acos", torch.acos],
        ["acos_", torch.acos_],
        ["argsort", torch.argsort],
        ["asin", torch.asin],
        ["asin_", torch.asin_],
        ["atan", torch.atan],
        ["atan_", torch.atan_],
        ["ceil", torch.ceil],
        ["ceil_", torch.ceil_],
        ["clone", torch.clone],
        ["cos", torch.cos],
        ["cos_", torch.cos_],
        ["cosh", torch.cosh],
        ["cosh_", torch.cosh_],
        ["digamma", torch.digamma],
        ["erf", torch.erf],
        ["erf_", torch.erf_],
        ["erfc", torch.erfc],
        ["erfc_", torch.erfc_],
        ["erfinv", torch.erfinv],
        ["exp", torch.exp],
        ["exp_", torch.exp_],
        ["expm1", torch.expm1],
        ["expm1_", torch.expm1_],
        ["floor", torch.floor],
        ["floor_", torch.floor_],
        ["frac", torch.frac],
        ["frac_", torch.frac_],
        ["hardshrink", torch.hardshrink],
        ["lgamma", torch.lgamma],
        ["log", torch.log],
        ["log10", torch.log10],
        ["log10_", torch.log10_],
        ["log1p", torch.log1p],
        ["log1p_", torch.log1p_],
        ["log2", torch.log2],
        ["log2_", torch.log2_],
        ["log_", torch.log_],
        ["neg", torch.neg],
        ["neg_", torch.neg_],
        ["reciprocal", torch.reciprocal],
        ["reciprocal_", torch.reciprocal_],
        ["relu", torch.relu],
        ["relu_", torch.relu_],
        ["round", torch.round],
        ["round_", torch.round_],
        ["rsqrt", torch.rsqrt],
        ["rsqrt_", torch.rsqrt_],
        ["sigmoid", torch.sigmoid],
        ["sigmoid_", torch.sigmoid_],
        ["sign", torch.sign],
        ["sin", torch.sin],
        ["sin_", torch.sin_],
        ["sinh", torch.sinh],
        ["sinh_", torch.sinh_],
        ["sqrt", torch.sqrt],
        ["sqrt_", torch.sqrt_],
        ["tan", torch.tan],
        ["tan_", torch.tan_],
        ["tanh", torch.tanh],
        ["tanh_", torch.tanh_],
        ["trunc", torch.trunc],
        ["trunc_", torch.trunc_],
        ["unique", torch.unique],
        ["zero_", torch.zero_],
        ["bernoulli_", lambda t: t.bernoulli_()],
        ["cauchy_", lambda t: t.cauchy_()],
        ["contiguous", lambda t: t.contiguous()],
        ["digamma_", lambda t: t.digamma_()],
        ["erfinv_", lambda t: t.erfinv_()],
        ["exponential_", lambda t: t.exponential_()],
        ["lgamma_", lambda t: t.lgamma_()],
        ["normal_", lambda t: t.normal_()],
        ["random_", lambda t: t.random_()],
        ["sign_", lambda t: t.sign_()],
        ["uniform_", lambda t: t.uniform_()],
        ["half", lambda t: t.half()],
        ["long", lambda t: t.long()],
    ],
)


op_bench.generate_pt_tests_from_op_list(unary_ops_list, unary_ops_configs, UnaryOpBenchmark)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
