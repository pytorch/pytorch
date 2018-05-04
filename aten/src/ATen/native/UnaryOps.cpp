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

namespace at {
namespace native {

Tensor& fill_(Tensor& self, Scalar value) {
  return self._fill_(value);
}

Tensor& fill_(Tensor& self, const Tensor& value) {
  return self._fill_(value);
}

#define IMPLEMENT_UNARY_OP_PREQUEL(op)                           \
  Tensor op(const Tensor& self) {                                \
    Tensor result = self.type().tensor();                        \
    return at::op##_out(result, self);                           \
  }                                                              \
  Tensor& _##op##__cuda(Tensor& self) {                          \
    return at::_##op##_out(self, self);                          \
  }                                                              \
  Tensor& _##op##_out_cuda(Tensor& result, const Tensor& self) { \
    return at::_##op##_out(result, self);                        \
  }

#define IMPLEMENT_UNARY_OP_FLOAT_CMATH(op, opfn)                          \
  Tensor& _##op##__cpu(Tensor& self_) {                                   \
    if (self_.numel() > 0) {                                              \
      Tensor self = sort_strides(self_);                                  \
      AT_DISPATCH_FLOATING_TYPES(self.type(), op, [&] {                   \
        CPU_tensor_parallel_apply1<scalar_t>(                             \
            self, [](scalar_t& y) { y = opfn(y); });                      \
      });                                                                 \
    }                                                                     \
    return self_;                                                         \
  }                                                                       \
  Tensor& _##op##_out_cpu(Tensor& result, const Tensor& self) {           \
    result.resize_(self.sizes());                                         \
    if (result.numel() > 0) {                                             \
      AT_DISPATCH_FLOATING_TYPES(self.type(), op, [&] {                   \
        CPU_tensor_parallel_apply2<scalar_t, scalar_t>(                   \
            result, self, [](scalar_t& y, scalar_t& x) { y = opfn(x); }); \
      });                                                                 \
    }                                                                     \
    return result;                                                        \
  }

#define IMPLEMENT_UNARY_OP_VEC(op, opfn)                                    \
  Tensor& _##op##__cpu(Tensor& self_) {                                     \
    if (self_.numel() > 0) {                                                \
      Tensor self = sort_strides(self_);                                    \
      if (self.is_contiguous()) {                                           \
        op##Impl(self, self);                                               \
      } else {                                                              \
        AT_DISPATCH_FLOATING_TYPES(self.type(), op, [&] {                   \
          CPU_tensor_parallel_apply1<scalar_t>(                             \
              self, [](scalar_t& y) { y = opfn(y); });                      \
        });                                                                 \
      }                                                                     \
    }                                                                       \
    return self_;                                                           \
  }                                                                         \
  Tensor& _##op##_out_cpu(Tensor& result, const Tensor& self) {             \
    result.resize_(self.sizes());                                           \
    if (result.numel() > 0) {                                               \
      if (result.is_contiguous() && self.is_contiguous()) {                 \
        op##Impl(result, self);                                             \
      } else {                                                              \
        AT_DISPATCH_FLOATING_TYPES(self.type(), op, [&] {                   \
          CPU_tensor_parallel_apply2<scalar_t, scalar_t>(                   \
              result, self, [](scalar_t& y, scalar_t& x) { y = opfn(x); }); \
        });                                                                 \
      }                                                                     \
    }                                                                       \
    return result;                                                          \
  }

IMPLEMENT_UNARY_OP_PREQUEL(abs)
IMPLEMENT_UNARY_OP_PREQUEL(acos)
IMPLEMENT_UNARY_OP_PREQUEL(asin)
IMPLEMENT_UNARY_OP_PREQUEL(atan)
IMPLEMENT_UNARY_OP_PREQUEL(ceil)
IMPLEMENT_UNARY_OP_PREQUEL(cos)
IMPLEMENT_UNARY_OP_PREQUEL(cosh)
IMPLEMENT_UNARY_OP_PREQUEL(erf)
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

Tensor tanh(const Tensor& self) {
  Tensor result = self.type().tensor();
  return at::tanh_out(result, self);
}
Tensor& _tanh__cuda(Tensor& self) {
  return at::_th_tanh_out(self, self);
}
Tensor& _tanh_out_cuda(Tensor& result, const Tensor& self) {
  return at::_th_tanh_out(result, self);
}

IMPLEMENT_UNARY_OP_VEC(abs, std::abs)
IMPLEMENT_UNARY_OP_VEC(acos, std::acos)
IMPLEMENT_UNARY_OP_VEC(asin, std::asin)
IMPLEMENT_UNARY_OP_VEC(atan, std::atan)
IMPLEMENT_UNARY_OP_VEC(ceil, std::ceil)
IMPLEMENT_UNARY_OP_FLOAT_CMATH(cos, std::cos)
IMPLEMENT_UNARY_OP_FLOAT_CMATH(cosh, std::cosh)
IMPLEMENT_UNARY_OP_VEC(erf, std::erf)
IMPLEMENT_UNARY_OP_VEC(exp, std::exp)
IMPLEMENT_UNARY_OP_VEC(expm1, std::expm1)
IMPLEMENT_UNARY_OP_VEC(floor, std::floor)
IMPLEMENT_UNARY_OP_VEC(log, std::log)
IMPLEMENT_UNARY_OP_VEC(log10, std::log10)
IMPLEMENT_UNARY_OP_VEC(log1p, std::log1p)
IMPLEMENT_UNARY_OP_VEC(log2, std::log2)
IMPLEMENT_UNARY_OP_VEC(round, std::round)
IMPLEMENT_UNARY_OP_VEC(rsqrt, 1 / std::sqrt)
IMPLEMENT_UNARY_OP_FLOAT_CMATH(sin, std::sin)
IMPLEMENT_UNARY_OP_FLOAT_CMATH(sinh, std::sinh)
IMPLEMENT_UNARY_OP_VEC(sqrt, std::sqrt)
IMPLEMENT_UNARY_OP_FLOAT_CMATH(tan, std::tan)
IMPLEMENT_UNARY_OP_FLOAT_CMATH(tanh, std::tanh)
IMPLEMENT_UNARY_OP_VEC(trunc, std::trunc)
}
} // namespace at
