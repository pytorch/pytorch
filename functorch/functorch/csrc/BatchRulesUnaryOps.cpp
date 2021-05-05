#include <functorch/csrc/BatchRulesHelper.h>

namespace at { namespace functorch {

TORCH_LIBRARY_IMPL(aten, FT_BATCHED_KEY, m) {
#define SINGLE_ARG(...) __VA_ARGS__
#define UNARY_POINTWISE(op) \
  VMAP_SUPPORT(#op, SINGLE_ARG(basic_unary_batch_rule<decltype(&at::op), &at::op>));

  UNARY_POINTWISE(abs);
  UNARY_POINTWISE(acos);
  UNARY_POINTWISE(asin);
  UNARY_POINTWISE(atan);
  UNARY_POINTWISE(ceil);
  UNARY_POINTWISE(cos);
  UNARY_POINTWISE(cosh);
  UNARY_POINTWISE(_conj);
  UNARY_POINTWISE(digamma);
  UNARY_POINTWISE(exp);
  UNARY_POINTWISE(expm1);
  UNARY_POINTWISE(floor);
  UNARY_POINTWISE(frac);
  UNARY_POINTWISE(lgamma);
  UNARY_POINTWISE(log);
  UNARY_POINTWISE(log10);
  UNARY_POINTWISE(log1p);
  UNARY_POINTWISE(log2);
  UNARY_POINTWISE(neg);
  UNARY_POINTWISE(reciprocal);
  UNARY_POINTWISE(relu);
  UNARY_POINTWISE(round);
  UNARY_POINTWISE(rsqrt);
  UNARY_POINTWISE(sigmoid);
  UNARY_POINTWISE(sign);
  UNARY_POINTWISE(sin);
  UNARY_POINTWISE(sinh);
  UNARY_POINTWISE(sqrt);
  UNARY_POINTWISE(tan);
  UNARY_POINTWISE(tanh);
  UNARY_POINTWISE(trunc);

#undef UNARY_POINTWISE
}

}}
