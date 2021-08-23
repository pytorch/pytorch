// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <functorch/csrc/BatchRulesHelper.h>
#include <functorch/csrc/PlumbingHelper.h>
#include <functorch/csrc/InPlacePlumbing.h>

namespace at { namespace functorch {

template <typename F, F Func>
static Tensor& unary_inplace_func_batch_rule(Tensor& self, optional<int64_t>) {
  Func(self);
  return self;
}

TORCH_LIBRARY_IMPL(aten, FT_BATCHED_KEY, m) {
#define SINGLE_ARG(...) __VA_ARGS__

  using UnaryInplaceBRType = Tensor& (*)(Tensor&, optional<int64_t>);
#define UNARY_POINTWISE_(op) \
  m.impl(#op, inplacePlumbing1<UnaryInplaceBRType, &unary_inplace_batch_rule<decltype(&Tensor::op), &Tensor::op>>);
#define UNARY_POINTWISE_FUNC_(op) \
  m.impl(#op, inplacePlumbing1<UnaryInplaceBRType, &unary_inplace_func_batch_rule<decltype(&at::op), &at::op>>);

#define UNARY_POINTWISE(op) \
  VMAP_SUPPORT(#op, BASIC_UNARY_BATCH_RULE(ATEN_FN(op)));

#define UNARY_POINTWISE_ALL(op) \
  UNARY_POINTWISE_(op ## _); \
  VMAP_SUPPORT(#op, BASIC_UNARY_BATCH_RULE(ATEN_FN(op)));

  UNARY_POINTWISE(_to_copy);
  UNARY_POINTWISE(alias);
  UNARY_POINTWISE_ALL(abs);
  UNARY_POINTWISE_ALL(acos);
  UNARY_POINTWISE_ALL(acosh);
  UNARY_POINTWISE(angle);
  UNARY_POINTWISE_ALL(asin);
  UNARY_POINTWISE_ALL(asinh);
  UNARY_POINTWISE_ALL(atan);
  UNARY_POINTWISE_ALL(atanh);
  UNARY_POINTWISE_ALL(bitwise_not);
  UNARY_POINTWISE_ALL(ceil);
  UNARY_POINTWISE_ALL(cos);
  UNARY_POINTWISE_ALL(cosh);
  UNARY_POINTWISE(_conj);
  UNARY_POINTWISE_ALL(deg2rad);
  UNARY_POINTWISE_ALL(digamma);
  UNARY_POINTWISE_ALL(erf);
  UNARY_POINTWISE_ALL(exp);
  UNARY_POINTWISE_ALL(expm1);
  UNARY_POINTWISE_ALL(floor);
  UNARY_POINTWISE_ALL(frac);
  UNARY_POINTWISE(glu);
  UNARY_POINTWISE(inverse);
  UNARY_POINTWISE(isfinite);
  UNARY_POINTWISE(isnan);
  UNARY_POINTWISE_ALL(lgamma);
  UNARY_POINTWISE_ALL(log);
  UNARY_POINTWISE_ALL(log10);
  UNARY_POINTWISE_ALL(log1p);
  UNARY_POINTWISE_ALL(log2);
  UNARY_POINTWISE(logit);
  UNARY_POINTWISE(mish);
  UNARY_POINTWISE_ALL(neg);
  UNARY_POINTWISE_ALL(rad2deg);
  UNARY_POINTWISE_ALL(reciprocal);
  UNARY_POINTWISE_ALL(round);
  UNARY_POINTWISE_ALL(rsqrt);
  UNARY_POINTWISE_ALL(sgn);
  UNARY_POINTWISE_ALL(sign);
  UNARY_POINTWISE_ALL(sin);
  UNARY_POINTWISE_ALL(sinc);
  UNARY_POINTWISE_ALL(sinh);
  UNARY_POINTWISE_ALL(sqrt);
  UNARY_POINTWISE_ALL(tan);
  UNARY_POINTWISE(threshold);
  UNARY_POINTWISE_ALL(trunc);

  // special-related
  UNARY_POINTWISE_ALL(i0);
  UNARY_POINTWISE_ALL(erfc);
  UNARY_POINTWISE_ALL(erfinv);
  UNARY_POINTWISE_ALL(exp2);

  // torch.special.* functions
  UNARY_POINTWISE(special_entr);
  UNARY_POINTWISE(special_erf);
  UNARY_POINTWISE(special_erfc);
  UNARY_POINTWISE(special_erfcx);
  UNARY_POINTWISE(special_erfinv);
  UNARY_POINTWISE(special_expit);
  UNARY_POINTWISE(special_expm1);
  UNARY_POINTWISE(special_digamma);
  UNARY_POINTWISE(special_psi);
  UNARY_POINTWISE(special_exp2);
  UNARY_POINTWISE(special_gammaln);
  UNARY_POINTWISE(special_i0);
  UNARY_POINTWISE(special_i0e);
  UNARY_POINTWISE(special_i1);
  UNARY_POINTWISE(special_i1e);
  UNARY_POINTWISE(special_log1p);
  UNARY_POINTWISE(special_ndtr);
  UNARY_POINTWISE(special_ndtri);
  UNARY_POINTWISE(special_round);
  UNARY_POINTWISE(special_sinc);

  // Activation functions (from https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity)
  UNARY_POINTWISE(elu);
  UNARY_POINTWISE(hardshrink);
  UNARY_POINTWISE(hardsigmoid);
  UNARY_POINTWISE(hardtanh);
  UNARY_POINTWISE(hardswish);
  UNARY_POINTWISE(leaky_relu);
  UNARY_POINTWISE(log_sigmoid);
  UNARY_POINTWISE_ALL(relu);
  UNARY_POINTWISE(relu6);
  UNARY_POINTWISE(selu);
  UNARY_POINTWISE(celu);
  UNARY_POINTWISE(gelu);
  UNARY_POINTWISE_ALL(sigmoid);
  UNARY_POINTWISE(silu);
  UNARY_POINTWISE(softplus);
  UNARY_POINTWISE(softshrink);
  UNARY_POINTWISE_ALL(tanh);


  UNARY_POINTWISE_(zero_);

#undef UNARY_POINTWISE
#undef UNARY_POINTWISE_
#undef UNARY_POINTWISE_ALL

}

#undef INVOKE
}}
