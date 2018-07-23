#include "ATen/ATen.h"

namespace at { namespace native {

Tensor& _clamp__cuda(Tensor& self, Scalar min, Scalar max) {
  return _th_clamp_(self, min, max);
}

Tensor& _clamp_out_cuda(
    Tensor& result,
    const Tensor& self,
    Scalar min,
    Scalar max) {
  result.resize_(self.sizes());
  result.copy_(self);
  return _th_clamp_(result, min, max);
}

Tensor& _clamp_max__cuda(Tensor& self, Scalar max) {
  return _th_clamp_max_(self, max);
}

Tensor& _clamp_max_out_cuda(Tensor& result, const Tensor& self, Scalar max) {
  result.resize_(self.sizes());
  result.copy_(self);
  return _th_clamp_max_(result, max);
}

Tensor& _clamp_min__cuda(Tensor& self, Scalar min) {
  return _th_clamp_min_(self, min);
}

Tensor& _clamp_min_out_cuda(Tensor& result, const Tensor& self, Scalar min) {
  result.resize_(self.sizes());
  result.copy_(self);
  return _th_clamp_min_(result, min);
}

// These are just forwarding stubs

#define IMPLEMENT_UNARY_OP_PREQUEL(op)                           \
  Tensor& _##op##__cuda(Tensor& self) {                          \
    return at::_##op##_out(self, self);                          \
  }                                                              \
  Tensor& _##op##_out_cuda(Tensor& result, const Tensor& self) { \
    return at::_##op##_out(result, self);                        \
  }


IMPLEMENT_UNARY_OP_PREQUEL(abs)
IMPLEMENT_UNARY_OP_PREQUEL(acos)
IMPLEMENT_UNARY_OP_PREQUEL(asin)
IMPLEMENT_UNARY_OP_PREQUEL(atan)
IMPLEMENT_UNARY_OP_PREQUEL(ceil)
IMPLEMENT_UNARY_OP_PREQUEL(cos)
IMPLEMENT_UNARY_OP_PREQUEL(cosh)
IMPLEMENT_UNARY_OP_PREQUEL(erf)
IMPLEMENT_UNARY_OP_PREQUEL(erfc)
IMPLEMENT_UNARY_OP_PREQUEL(exp)
IMPLEMENT_UNARY_OP_PREQUEL(expm1)
IMPLEMENT_UNARY_OP_PREQUEL(floor)
IMPLEMENT_UNARY_OP_PREQUEL(log)
IMPLEMENT_UNARY_OP_PREQUEL(log10)
IMPLEMENT_UNARY_OP_PREQUEL(log1p)
IMPLEMENT_UNARY_OP_PREQUEL(log2)
IMPLEMENT_UNARY_OP_PREQUEL(round)
IMPLEMENT_UNARY_OP_PREQUEL(rsqrt)
IMPLEMENT_UNARY_OP_PREQUEL(sigmoid)
IMPLEMENT_UNARY_OP_PREQUEL(sin)
IMPLEMENT_UNARY_OP_PREQUEL(sinh)
IMPLEMENT_UNARY_OP_PREQUEL(sqrt)
IMPLEMENT_UNARY_OP_PREQUEL(tan)
IMPLEMENT_UNARY_OP_PREQUEL(tanh)
IMPLEMENT_UNARY_OP_PREQUEL(trunc)

}}
