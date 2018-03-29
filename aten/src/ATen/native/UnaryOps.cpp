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

using unary_type = void(Tensor&, const Tensor&);

#define DISPATCH(NAME) \
unary_type* NAME ## Impl = DispatchStub<unary_type>::init<NAME ## ImplC, &NAME ## Impl>;\

#define BASIC(NAME) \
Tensor NAME(const Tensor& self) { \
  Tensor result = self.type().tensor(); \
  return at::NAME ## _out(result, self); \
} \

#define SELF(NAME) \
Tensor& NAME##_(Tensor& self) { \
  return at::NAME ## _out(self, self); \
} \

#define OUTCPU(NAME) \
Tensor& _ ## NAME ## _out_cpu(Tensor& result, const Tensor& self) { \
  return _unops_out_cpu(NAME ## Impl, result, self) ? result \
                                                : at::_ ## NAME ## _out(result, self); \
} \

#define OUTCUDA(NAME) \
Tensor& _ ## NAME ## _out_cuda(Tensor& result, const Tensor& self) { \
  return at::_ ## NAME ## _out(result, self); \
} \

bool _unops_out_cpu(unary_type* f, Tensor& result, const Tensor& self) {
  if (result.is_contiguous() && self.is_contiguous()) {
    result.resize_(self.sizes());
    f(result, self);
    return true;
  }
  return false;
}

UNARY_OPS_MACRO(DISPATCH)
UNARY_OPS_MACRO(BASIC)
UNARY_OPS_MACRO(SELF)
UNARY_OPS_MACRO(OUTCPU)
UNARY_OPS_MACRO(OUTCUDA)

}} // namespace at::native
