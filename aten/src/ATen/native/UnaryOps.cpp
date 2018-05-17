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

IMPLEMENT_UNARY_OP(cos)
IMPLEMENT_UNARY_OP(cosh)
IMPLEMENT_UNARY_OP(sin)
IMPLEMENT_UNARY_OP(sinh)
IMPLEMENT_UNARY_OP(tan)

IMPLEMENT_UNARY_OP(abs)
IMPLEMENT_UNARY_OP(acos)
IMPLEMENT_UNARY_OP(asin)
IMPLEMENT_UNARY_OP(atan)
IMPLEMENT_UNARY_OP(ceil)
IMPLEMENT_UNARY_OP(erf)
IMPLEMENT_UNARY_OP(exp)
IMPLEMENT_UNARY_OP(expm1)
IMPLEMENT_UNARY_OP(floor)
IMPLEMENT_UNARY_OP(log)
IMPLEMENT_UNARY_OP(log10)
IMPLEMENT_UNARY_OP(log1p)
IMPLEMENT_UNARY_OP(log2)
IMPLEMENT_UNARY_OP(round)
IMPLEMENT_UNARY_OP(sqrt)
IMPLEMENT_UNARY_OP(rsqrt)
IMPLEMENT_UNARY_OP(tanh)
IMPLEMENT_UNARY_OP(trunc)

}} // namespace at::native
