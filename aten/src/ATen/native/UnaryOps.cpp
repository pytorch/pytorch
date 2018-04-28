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

namespace at {
namespace native {

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

#define IMPLEMENT_UNARY_OP_FLOAT_CMATH(op)                                   \
  Tensor& _##op##__cpu(Tensor& self_) {                                      \
    if (self_.numel() > 0) {                                                 \
      Tensor self = sort_strides(self_);                                     \
      AT_DISPATCH_FLOATING_TYPES(self.type(), op, [&] {                      \
        CPU_tensor_parallel_apply1<scalar_t>(                                \
            self, [](scalar_t& y) { y = std::op(y); });                      \
      });                                                                    \
    }                                                                        \
    return self_;                                                            \
  }                                                                          \
  Tensor& _##op##_out_cpu(Tensor& result, const Tensor& self) {              \
    result.resize_(self.sizes());                                            \
    if (result.numel() > 0) {                                                \
      AT_DISPATCH_FLOATING_TYPES(self.type(), op, [&] {                      \
        CPU_tensor_parallel_apply2<scalar_t, scalar_t>(                      \
            result, self, [](scalar_t& y, scalar_t& x) { y = std::op(x); }); \
      });                                                                    \
    }                                                                        \
    return result;                                                           \
  }

#define IMPLEMENT_UNARY_OP_VEC(op)                                             \
  Tensor& _##op##__cpu(Tensor& self_) {                                        \
    if (self_.numel() > 0) {                                                   \
      Tensor self = sort_strides(self_);                                       \
      if (self.is_contiguous()) {                                              \
        op##Impl(self, self);                                                  \
      } else {                                                                 \
        AT_DISPATCH_FLOATING_TYPES(self.type(), op, [&] {                      \
          CPU_tensor_parallel_apply1<scalar_t>(                                \
              self, [](scalar_t& y) { y = std::op(y); });                      \
        });                                                                    \
      }                                                                        \
    }                                                                          \
    return self_;                                                              \
  }                                                                            \
  Tensor& _##op##_out_cpu(Tensor& result, const Tensor& self) {                \
    result.resize_(self.sizes());                                              \
    if (result.numel() > 0) {                                                  \
      if (result.is_contiguous() && self.is_contiguous()) {                    \
        op##Impl(result, self);                                                \
      } else {                                                                 \
        AT_DISPATCH_FLOATING_TYPES(self.type(), op, [&] {                      \
          CPU_tensor_parallel_apply2<scalar_t, scalar_t>(                      \
              result, self, [](scalar_t& y, scalar_t& x) { y = std::op(x); }); \
        });                                                                    \
      }                                                                        \
    }                                                                          \
    return result;                                                             \
  }

IMPLEMENT_UNARY_OP_PREQUEL(abs)
IMPLEMENT_UNARY_OP_PREQUEL(ceil)
IMPLEMENT_UNARY_OP_PREQUEL(cos)
IMPLEMENT_UNARY_OP_PREQUEL(exp)
IMPLEMENT_UNARY_OP_PREQUEL(floor)
IMPLEMENT_UNARY_OP_PREQUEL(log)
IMPLEMENT_UNARY_OP_PREQUEL(round)
IMPLEMENT_UNARY_OP_PREQUEL(sin)
IMPLEMENT_UNARY_OP_PREQUEL(sqrt)
IMPLEMENT_UNARY_OP_PREQUEL(trunc)

IMPLEMENT_UNARY_OP_VEC(abs)
IMPLEMENT_UNARY_OP_VEC(ceil)
IMPLEMENT_UNARY_OP_FLOAT_CMATH(cos)
IMPLEMENT_UNARY_OP_FLOAT_CMATH(exp)
IMPLEMENT_UNARY_OP_VEC(floor)
IMPLEMENT_UNARY_OP_FLOAT_CMATH(log)
IMPLEMENT_UNARY_OP_VEC(round)
IMPLEMENT_UNARY_OP_FLOAT_CMATH(sin)
IMPLEMENT_UNARY_OP_VEC(sqrt)
IMPLEMENT_UNARY_OP_VEC(trunc)
}
} // namespace at
