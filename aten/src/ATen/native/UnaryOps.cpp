#include "ATen/ATen.h"
#include "ATen/Dispatch.h"
#include "ATen/ExpandUtils.h"
#include "ATen/NativeFunctions.h"
#include "ATen/WrapDimUtils.h"
#include "cpu/UnaryOpsKernel.h"

#include <algorithm>
#include <functional>
#include <numeric>
#include <vector>

#include <map>

namespace at { namespace native {

#define IMPLEMENT_UNARY_OP(op)                                                \
Tensor op(const Tensor& self) {                                               \
  Tensor result = self.type().tensor();                                       \
  return at::op ## _out(result, self);                                        \
}                                                                             \
Tensor& op##_(Tensor& self) {                                                 \
  return at::op ## _out(self, self);                                          \
}                                                                             \
Tensor& _ ## op ## _out_cuda(Tensor& result, const Tensor& self) {            \
  return at::_ ## op ## _out(result, self);                                   \
}                                                                             \
Tensor& _ ## op ## _out_cpu(Tensor& result, const Tensor& self) {             \
  if (result.is_contiguous() && self.is_contiguous()) {                       \
    result.resize_(self.sizes());                                             \
    if (result.numel() > 0) {                                                 \
      op ## Impl(result, self);                                               \
    }                                                                         \
    return result;                                                            \
  }                                                                           \
  return at::_ ## op ## _out(result, self);                                   \
}

IMPLEMENT_UNARY_OP(abs)
IMPLEMENT_UNARY_OP(ceil)
IMPLEMENT_UNARY_OP(cos)
IMPLEMENT_UNARY_OP(exp)
IMPLEMENT_UNARY_OP(floor)
IMPLEMENT_UNARY_OP(log)
IMPLEMENT_UNARY_OP(round)
IMPLEMENT_UNARY_OP(sin)
IMPLEMENT_UNARY_OP(sqrt)
IMPLEMENT_UNARY_OP(trunc)

}} // namespace at::native
