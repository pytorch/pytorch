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
  VMAP_SUPPORT(#op, SINGLE_ARG(basic_unary_batch_rule<decltype(&at::op), &at::op>));

#define UNARY_POINTWISE_SCALAR(op) \
  VMAP_SUPPORT(#op, SINGLE_ARG(basic_unary_batch_rule<decltype(&at::op), &at::op, const Scalar&>));

#define UNARY_POINTWISE_SCALAR_SCALAR(op) \
  VMAP_SUPPORT(#op, SINGLE_ARG(basic_unary_batch_rule<decltype(&at::op), &at::op, const Scalar&, const Scalar&>));

#define UNARY_POINTWISE_SCALAR_SCALAR_SCALAR(op) \
  VMAP_SUPPORT(#op, SINGLE_ARG(basic_unary_batch_rule<decltype(&at::op), &at::op, const Scalar&, const Scalar&, const Scalar&>));

#define UNARY_POINTWISE_ALL(op) \
  UNARY_POINTWISE_(op ## _); \
  VMAP_SUPPORT(#op, SINGLE_ARG(basic_unary_batch_rule<decltype(&at::op), &at::op>));

  UNARY_POINTWISE_ALL(abs);
  UNARY_POINTWISE_ALL(acos);
  UNARY_POINTWISE_ALL(acosh);
  UNARY_POINTWISE(angle);
  UNARY_POINTWISE_ALL(asin);
  UNARY_POINTWISE_ALL(asinh);
  UNARY_POINTWISE_ALL(atan);
  UNARY_POINTWISE_ALL(atanh);
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
  UNARY_POINTWISE(inverse);
  OP_DECOMPOSE(isinf);
  UNARY_POINTWISE(isfinite);
  UNARY_POINTWISE(isnan);
  UNARY_POINTWISE_ALL(lgamma);
  UNARY_POINTWISE_ALL(log);
  UNARY_POINTWISE_ALL(log10);
  UNARY_POINTWISE_ALL(log1p);
  UNARY_POINTWISE_ALL(log2);
  UNARY_POINTWISE_ALL(neg);
  UNARY_POINTWISE_ALL(rad2deg);
  UNARY_POINTWISE_ALL(reciprocal);
  UNARY_POINTWISE_ALL(round);
  UNARY_POINTWISE_ALL(rsqrt);
  UNARY_POINTWISE_ALL(sign);
  UNARY_POINTWISE_ALL(sin);
  UNARY_POINTWISE_ALL(sinh);
  UNARY_POINTWISE_ALL(sqrt);
  UNARY_POINTWISE_ALL(tan);
  UNARY_POINTWISE_ALL(trunc);

  // Activation functions (from https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity)
  UNARY_POINTWISE_SCALAR_SCALAR_SCALAR(elu);
  UNARY_POINTWISE_SCALAR(hardshrink);
  UNARY_POINTWISE(hardsigmoid);
  UNARY_POINTWISE_SCALAR_SCALAR(hardtanh);
  UNARY_POINTWISE(hardswish);
  UNARY_POINTWISE_SCALAR(leaky_relu);
  UNARY_POINTWISE(log_sigmoid);
  UNARY_POINTWISE_ALL(relu);
  UNARY_POINTWISE(relu6);
  UNARY_POINTWISE(selu);
  UNARY_POINTWISE_SCALAR(celu);
  UNARY_POINTWISE(gelu);
  UNARY_POINTWISE_ALL(sigmoid);
  UNARY_POINTWISE(silu);
  UNARY_POINTWISE_SCALAR_SCALAR(softplus);
  UNARY_POINTWISE_SCALAR(softshrink);
  UNARY_POINTWISE_ALL(tanh);


  UNARY_POINTWISE_(zero_);

#undef UNARY_POINTWISE_SCALAR_SCALAR_SCALAR
#undef UNARY_POINTWISE_SCALAR_SCALAR
#undef UNARY_POINTWISE_SCALAR
#undef UNARY_POINTWISE
#undef UNARY_POINTWISE_
#undef UNARY_POINTWISE_ALL

}

#undef INVOKE
}}
