#include "ATen/ATen.h"
#include "ATen/Dispatch.h"
#include "ATen/ExpandUtils.h"
#include "ATen/NativeFunctions.h"
#include "ATen/WrapDimUtils.h"

#include "ATen/CPUApplyUtils.h"
#include "ATen/Parallel.h"
#include "ATen/native/cpu/UnaryOpsKernel.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric>
#include <vector>

#include <map>

// NOTE:
// YOU ARE NOT OBLIGED TO USE THESE MACROS
// If you're writing something more specialized, please don't try to make them
// work for your case, but just write something new instead.

namespace at { namespace native {

Tensor& fill_(Tensor& self, Scalar value) {
  return self._fill_(value);
}

Tensor& fill_(Tensor& self, const Tensor& value) {
  return self._fill_(value);
}

Tensor& zero_(Tensor& self) {
  self._th_zero_();
  return self;
}

Tensor sign(const Tensor& self) {
  Tensor result = self.type().tensor();
  return native::sign_out(result, self);
}

Tensor& sign_(Tensor& self) {
  return native::sign_out(self, self);
}

Tensor& sign_out(Tensor& result, const Tensor& self) {
  result.resize_(self.sizes());
  result.copy_(_sign(self));
  return result;
}

Tensor clamp(const Tensor& self, Scalar value1, Scalar value2) {
  Tensor result = self.type().tensor();
  return native::clamp_out(result, self, value1, value2);
}

Tensor& clamp_(Tensor& self, Scalar value1, Scalar value2) {
  return self._clamp_(value1, value2);
}

Tensor& clamp_out(Tensor& result, const Tensor& self, Scalar value1, Scalar value2) {
  result.resize_(self.sizes());
  result.copy_(self);
  return result._clamp_(value1, value2);
}

Tensor clamp_max(const Tensor& self, Scalar value) {
  Tensor result = self.type().tensor();
  return native::clamp_max_out(result, self, value);
}

Tensor& clamp_max_(Tensor& self, Scalar value) {
  return self._clamp_max_(value);
}

Tensor& clamp_max_out(Tensor& result, const Tensor& self, Scalar value1) {
  result.resize_(self.sizes());
  result.copy_(self);
  return result._clamp_max_(value1);
}

Tensor clamp_min(const Tensor& self, Scalar value) {
  Tensor result = self.type().tensor();
  return native::clamp_min_out(result, self, value);
}

Tensor& clamp_min_(Tensor& self, Scalar value) {
  return self._clamp_min_(value);
}

Tensor& clamp_min_out(Tensor& result, const Tensor& self, Scalar value1) {
  result.resize_(self.sizes());
  result.copy_(self);
  return result._clamp_min_(value1);
}

Tensor frac(const Tensor& self) {
  Tensor result = self.type().tensor();
  return frac_out(result, self);
}

Tensor& frac_(Tensor& self) {
  return frac_out(self, self);
}

Tensor& _frac_out_cpu(Tensor& result, const Tensor& self) {
  result.resize_(self.sizes());
  return at::_th_frac_out(result, self);
}

Tensor erfinv(const Tensor& self) {
  Tensor result = self.type().tensor();
  return erfinv_out(result, self);
}

Tensor& _erfinv__cpu(Tensor& self) {
  return _erfinv_out_cpu(self, self);
}

Tensor& _erfinv_out_cpu(Tensor& result, const Tensor& self) {
  result.resize_(self.sizes());
  return at::_erfinv_out(result, self);
}

Tensor sigmoid(const Tensor& self) {
  Tensor result = self.type().tensor();
  return sigmoid_out(result, self);
}

Tensor& sigmoid_(Tensor& self) {
  return sigmoid_out(self, self);
}

Tensor& _sigmoid_out_cpu(Tensor& result, const Tensor& self) {
  result.resize_(self.sizes());
  return at::_th_sigmoid_out(result, self);
}

Tensor clone(const Tensor& self) {
  return self.type()._th_clone(self);
}

Tensor _contiguous_cpu(const Tensor& self) {
  return self.type()._th_contiguous(self);
}

Tensor neg(const Tensor& self) {
  Tensor result = self.type().tensor();
  return neg_out(result, self);
}

Tensor& neg_(Tensor& self) {
  return neg_out(self, self);
}

Tensor& _neg_out_cpu(Tensor& result, const Tensor& self) {
  result.resize_(self.sizes());
  return at::_th_neg_out(result, self);
}

Tensor reciprocal(const Tensor& self) {
  Tensor result = self.type().tensor();
  return reciprocal_out(result, self);
}

Tensor& reciprocal_(Tensor& self) {
  return reciprocal_out(self, self);
}

Tensor& _reciprocal_out_cpu(Tensor& result, const Tensor& self) {
  result.resize_(self.sizes());
  return at::_th_reciprocal_out(result, self);
}

// NB: If you use this macro, you may also need to add a CUDA forwarding
// stub in CUDAUnaryOps
#define IMPLEMENT_UNARY_OP(op)                                  \
  Tensor op(const Tensor& self) {                               \
    Tensor result = self.type().tensor();                       \
    return at::op##_out(result, self);                          \
  }                                                             \
                                                                \
  Tensor& _##op##__cpu(Tensor& self_) {                         \
    Tensor self = sort_strides(self_);                          \
    _##op##_out_cpu(self, self);                                \
    return self_;                                               \
  }                                                             \
                                                                \
  Tensor& _##op##_out_cpu(Tensor& result, const Tensor& self) { \
    result.resize_(self.sizes());                               \
    op##Impl(result, self);                                     \
    return result;                                              \
  }

IMPLEMENT_UNARY_OP(abs)
IMPLEMENT_UNARY_OP(acos)
IMPLEMENT_UNARY_OP(asin)
IMPLEMENT_UNARY_OP(atan)
IMPLEMENT_UNARY_OP(ceil)
IMPLEMENT_UNARY_OP(cos)
IMPLEMENT_UNARY_OP(cosh)
IMPLEMENT_UNARY_OP(erf)
IMPLEMENT_UNARY_OP(exp)
IMPLEMENT_UNARY_OP(expm1)
IMPLEMENT_UNARY_OP(floor)
IMPLEMENT_UNARY_OP(log)
IMPLEMENT_UNARY_OP(log10)
IMPLEMENT_UNARY_OP(log1p)
IMPLEMENT_UNARY_OP(log2)
IMPLEMENT_UNARY_OP(round)
IMPLEMENT_UNARY_OP(sin)
IMPLEMENT_UNARY_OP(sinh)
IMPLEMENT_UNARY_OP(sqrt)
IMPLEMENT_UNARY_OP(rsqrt)
IMPLEMENT_UNARY_OP(tan)
IMPLEMENT_UNARY_OP(tanh)
IMPLEMENT_UNARY_OP(trunc)

}} // namespace at::native
