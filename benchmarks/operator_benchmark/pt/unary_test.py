import torch

import operator_benchmark as op_bench


"""Microbenchmarks for point-wise unary operator."""


# Configs for pointwise unary ops
unary_ops_configs_short = op_bench.config_list(
    attr_names=["M", "N"],
    attrs=[
        [512, 512],
    ],
    cross_product_configs={
        "device": ["cpu", "cuda"],
        "dtype": [torch.float32, torch.bfloat16],
    },
    tags=["short"],
)

unary_ops_configs_long = op_bench.cross_product_configs(
    M=[256, 1024],
    N=[256, 1024],
    device=["cpu", "cuda"],
    tags=["long"],
    dtype=[torch.float32, torch.bfloat16],
)


class UnaryOpBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, M, N, device, dtype, op_func):
        self.op_func = op_func
        if self.op_func == exponential_:
            self.inputs = {"input": torch.rand(M, N, device=device, dtype=dtype).abs()}
        else:
            self.inputs = {"input": torch.rand(M, N, device=device, dtype=dtype)}

    def forward(self, input):
        return self.op_func(input)


def bernoulli_(input):
    return input.bernoulli_()


def cauchy_(input):
    return input.cauchy_()


def digamma_(input):
    return input.digamma_()


def exponential_(input):
    return input.exponential_()


def normal_(input):
    return input.normal_()


def log_normal_(input):
    return input.log_normal_()


def random_(input):
    return input.random_()


def sign_(input):
    return input.sign_()


def uniform_(input):
    return input.uniform_()


def half_(input):
    return input.half()


def long_(input):
    return input.long()


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
        ["logit", torch.logit],
        ["logit_", torch.logit_],
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
        ["sgn", torch.sgn],
        ["sin", torch.sin],
        ["sin_", torch.sin_],
        ["sinh", torch.sinh],
        ["sqrt", torch.sqrt],
        ["sqrt_", torch.sqrt_],
        ["square", torch.square],
        ["square_", torch.square_],
        ["tan", torch.tan],
        ["tan_", torch.tan_],
        ["tanh", torch.tanh],
        ["tanh_", torch.tanh_],
        ["trunc", torch.trunc],
        ["trunc_", torch.trunc_],
        ["unique", torch.functional._return_output],
        ["zero_", torch.zero_],
        ["bernoulli_", bernoulli_],
        ["cauchy_", cauchy_],
        ["digamma_", digamma_],
        ["exponential_", exponential_],
        ["log_normal_", log_normal_],
        ["normal_", normal_],
        ["random_", random_],
        ["sign_", sign_],
        ["uniform_", uniform_],
        ["half", half_],
        ["long", long_],
    ],
)


op_bench.generate_pt_tests_from_op_list(
    unary_ops_list, unary_ops_configs_short + unary_ops_configs_long, UnaryOpBenchmark
)


# Ops modeled after SciPy's special module
special_ops_list = op_bench.op_list(
    attr_names=["op_name", "op_func"],
    attrs=[
        ["digamma", torch.special.digamma],
        ["special.entr", torch.special.entr],
        ["special.expit", torch.special.expit],
        ["special.erfcx", torch.special.erfcx],
        ["special.bessel_j0", torch.special.bessel_j0],
        ["special.bessel_j1", torch.special.bessel_j1],
        ["special.i0", torch.special.i0],
        ["special.i0e", torch.special.i0e],
        ["special.i1", torch.special.i1],
        ["special.i1e", torch.special.i1e],
        ["special.log_ndtr", torch.special.log_ndtr],
        ["special.ndtr", torch.special.ndtr],
        ["special.ndtri", torch.special.ndtri],
        ["special.scaled_modified_bessel_k0", torch.special.scaled_modified_bessel_k0],
        ["special.scaled_modified_bessel_k1", torch.special.scaled_modified_bessel_k1],
    ],
)

op_bench.generate_pt_tests_from_op_list(
    special_ops_list, unary_ops_configs_short, UnaryOpBenchmark
)

if __name__ == "__main__":
    op_bench.benchmark_runner.main()
