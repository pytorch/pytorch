#include "ATen/ATen.h"

namespace at { namespace native {

// These are just forwarding stubs

Tensor _contiguous_cuda(const Tensor& self) {
  return self.type()._th_contiguous(self);
}

Tensor& _frac_out_cuda(Tensor& result, const Tensor& self) {
  result.resize_(self.sizes());
  return at::_th_frac_out(result, self);
}

Tensor& _sigmoid_out_cuda(Tensor& result, const Tensor& self) {
  result.resize_(self.sizes());
  return at::_th_sigmoid_out(result, self);
}

Tensor& _neg_out_cuda(Tensor& result, const Tensor& self) {
  result.resize_(self.sizes());
  return at::_th_neg_out(result, self);
}

Tensor& _reciprocal_out_cuda(Tensor& result, const Tensor& self) {
  result.resize_(self.sizes());
  return at::_th_reciprocal_out(result, self);
}

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
IMPLEMENT_UNARY_OP_PREQUEL(erfinv)
IMPLEMENT_UNARY_OP_PREQUEL(exp)
IMPLEMENT_UNARY_OP_PREQUEL(expm1)
IMPLEMENT_UNARY_OP_PREQUEL(floor)
IMPLEMENT_UNARY_OP_PREQUEL(log)
IMPLEMENT_UNARY_OP_PREQUEL(log10)
IMPLEMENT_UNARY_OP_PREQUEL(log1p)
IMPLEMENT_UNARY_OP_PREQUEL(log2)
IMPLEMENT_UNARY_OP_PREQUEL(round)
IMPLEMENT_UNARY_OP_PREQUEL(sin)
IMPLEMENT_UNARY_OP_PREQUEL(sinh)
IMPLEMENT_UNARY_OP_PREQUEL(sqrt)
IMPLEMENT_UNARY_OP_PREQUEL(rsqrt)
IMPLEMENT_UNARY_OP_PREQUEL(tan)
IMPLEMENT_UNARY_OP_PREQUEL(trunc)

Tensor& _tanh__cuda(Tensor& self) {
  return at::_th_tanh_out(self, self);
}
Tensor& _tanh_out_cuda(Tensor& result, const Tensor& self) {
  return at::_th_tanh_out(result, self);
}

}}
