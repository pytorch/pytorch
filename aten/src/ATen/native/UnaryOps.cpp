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

Tensor clamp(const Tensor& self, Scalar min, Scalar max) {
  Tensor result = self.type().tensor();
  return clamp_out(result, self, min, max);
}

Tensor& _clamp__cpu(Tensor& self_, Scalar min, Scalar max) {
  Tensor self = sort_strides(self_);                          
  clamp_Impl(self, min, max);
  return self_;
}

Tensor& _clamp_out_cpu(Tensor& result, const Tensor& self, Scalar min, Scalar max) {
  result.resize_(self.sizes());
  clampImpl(result, self, min, max);
  return result;
}

Tensor clamp_max(const Tensor& self, Scalar max) {
  Tensor result = self.type().tensor();
  return clamp_max_out(result, self, max);
}

Tensor& _clamp_max__cpu(Tensor& self_, Scalar max) {
   Tensor self = sort_strides(self_);                          
   clampMax_Impl(self, max);
   return self_;
}

Tensor& _clamp_max_out_cpu(Tensor& result, const Tensor& self, Scalar max) {
  result.resize_(self.sizes());
  clampMaxImpl(result, self, max);
  return result;
}

Tensor clamp_min(const Tensor& self, Scalar min) {
  Tensor result = self.type().tensor();
  return clamp_min_out(result, self, min);
}

Tensor& _clamp_min__cpu(Tensor& self_, Scalar min) {
  Tensor self = sort_strides(self_);                          
  clampMin_Impl(self, min);
  return self_;
}

Tensor& _clamp_min_out_cpu(Tensor& result, const Tensor& self, Scalar min) {
  result.resize_(self.sizes());
  clampMinImpl(result, self, min);
  return result;
}

#define IMPLEMENT_KERNEL_LOOP(types, op, opfn)                          \
  static void op##_Impl(Tensor& self) {                                 \
    AT_DISPATCH_##types##_TYPES(self.type(), #op, [&] {                 \
      CPU_tensor_parallel_apply1<scalar_t>(                             \
          self, [](scalar_t& x) { x = opfn(x); });                      \
    });                                                                 \
  }                                                                     \
  static void op##Impl(Tensor& result, const Tensor& self) {            \
    AT_DISPATCH_##types##_TYPES(self.type(), #op, [&] {                 \
      CPU_tensor_parallel_apply2<scalar_t, scalar_t>(                   \
          result, self, [](scalar_t& x, scalar_t& y) { x = opfn(y); }); \
    });                                                                 \
  }

IMPLEMENT_KERNEL_LOOP(FLOATING, cos, std::cos)
IMPLEMENT_KERNEL_LOOP(FLOATING, cosh, std::cosh)
IMPLEMENT_KERNEL_LOOP(FLOATING, sin, std::sin)
IMPLEMENT_KERNEL_LOOP(FLOATING, sinh, std::sinh)
IMPLEMENT_KERNEL_LOOP(FLOATING, tan, std::tan)

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
    op##_Impl(self);                                            \
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
IMPLEMENT_UNARY_OP(frac)
IMPLEMENT_UNARY_OP(floor)
IMPLEMENT_UNARY_OP(log)
IMPLEMENT_UNARY_OP(log10)
IMPLEMENT_UNARY_OP(log1p)
IMPLEMENT_UNARY_OP(log2)
IMPLEMENT_UNARY_OP(neg)
IMPLEMENT_UNARY_OP(reciprocal)
IMPLEMENT_UNARY_OP(round)
IMPLEMENT_UNARY_OP(sigmoid)
IMPLEMENT_UNARY_OP(sin)
IMPLEMENT_UNARY_OP(sinh)
IMPLEMENT_UNARY_OP(sqrt)
IMPLEMENT_UNARY_OP(rsqrt)
IMPLEMENT_UNARY_OP(tan)
IMPLEMENT_UNARY_OP(tanh)
IMPLEMENT_UNARY_OP(trunc)

}} // namespace at::native
