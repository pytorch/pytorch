// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <functorch/csrc/BatchRulesHelper.h>
#include <functorch/csrc/PlumbingHelper.h>
#include <functorch/csrc/InPlacePlumbing.h>

namespace at { namespace functorch {

TORCH_LIBRARY_IMPL(aten, FT_BATCHED_KEY, m) {

#define UNARY_POINTWISE_ALL(op) \
  POINTWISE_BOXED(op ## _); \
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
  UNARY_POINTWISE(isfinite);
  UNARY_POINTWISE(isnan);
  UNARY_POINTWISE(isposinf);
  UNARY_POINTWISE(isneginf);
  UNARY_POINTWISE_ALL(lgamma);
  UNARY_POINTWISE_ALL(log);
  UNARY_POINTWISE_ALL(log10);
  UNARY_POINTWISE_ALL(log1p);
  UNARY_POINTWISE_ALL(log2);
  UNARY_POINTWISE_ALL(logical_not);
  UNARY_POINTWISE_ALL(logit);
  UNARY_POINTWISE_ALL(mish);
  UNARY_POINTWISE_ALL(mvlgamma);
  UNARY_POINTWISE_ALL(nan_to_num);
  UNARY_POINTWISE_ALL(neg);
  UNARY_POINTWISE_ALL(positive);
  UNARY_POINTWISE_ALL(rad2deg);
  UNARY_POINTWISE_ALL(reciprocal);
  UNARY_POINTWISE_ALL(round);
  UNARY_POINTWISE_ALL(rsqrt);
  UNARY_POINTWISE_ALL(sgn);
  UNARY_POINTWISE_ALL(sign);
  UNARY_POINTWISE(signbit);
  UNARY_POINTWISE_ALL(sin);
  UNARY_POINTWISE_ALL(sinc);
  UNARY_POINTWISE_ALL(sinh);
  UNARY_POINTWISE_ALL(sqrt);
  UNARY_POINTWISE_ALL(tan);
  UNARY_POINTWISE_ALL(threshold);
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
  UNARY_POINTWISE_ALL(elu);
  UNARY_POINTWISE(hardshrink);
  UNARY_POINTWISE_ALL(hardsigmoid);
  UNARY_POINTWISE_ALL(hardtanh);
  UNARY_POINTWISE_ALL(hardswish);
  UNARY_POINTWISE_ALL(leaky_relu);
  UNARY_POINTWISE(log_sigmoid);
  UNARY_POINTWISE_ALL(relu);
  UNARY_POINTWISE_ALL(relu6);
  UNARY_POINTWISE_ALL(selu);
  UNARY_POINTWISE_ALL(celu);
  UNARY_POINTWISE(gelu);
  UNARY_POINTWISE_ALL(sigmoid);
  UNARY_POINTWISE_ALL(silu);
  UNARY_POINTWISE(softplus);
  UNARY_POINTWISE(softshrink);
  UNARY_POINTWISE_ALL(tanh);

  POINTWISE_BOXED(fill_.Scalar);
  POINTWISE_BOXED(zero_);

#undef UNARY_POINTWISE
#undef UNARY_POINTWISE_ALL

}

#undef INVOKE
}}
