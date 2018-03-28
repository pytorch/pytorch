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
unary_type* ceilImpl = DispatchStub<unary_type>::init<ceilImplC, &ceilImpl>;
unary_type* floorImpl = DispatchStub<unary_type>::init<floorImplC, &floorImpl>;
unary_type* roundImpl = DispatchStub<unary_type>::init<roundImplC, &roundImpl>;
unary_type* truncImpl = DispatchStub<unary_type>::init<truncImplC, &truncImpl>;
unary_type* sqrtImpl = DispatchStub<unary_type>::init<sqrtImplC, &sqrtImpl>;

// WRAP OPS #################################################################

Tensor ceil(const Tensor& self) {
  Tensor result = self.type().tensor();
  return at::ceil_out(result, self);
}
Tensor floor(const Tensor& self) {
  Tensor result = self.type().tensor();
  return at::floor_out(result, self);
}
Tensor round(const Tensor& self) {
  Tensor result = self.type().tensor();
  return at::round_out(result, self);
}
Tensor trunc(const Tensor& self) {
  Tensor result = self.type().tensor();
  return at::trunc_out(result, self);
}
Tensor sqrt(const Tensor& self) {
  Tensor result = self.type().tensor();
  return at::sqrt_out(result, self);
}

Tensor& ceil_(Tensor& self) {
  return at::ceil_out(self, self);
}
Tensor& floor_(Tensor& self) {
  return at::floor_out(self, self);
}
Tensor& round_(Tensor& self) {
  return at::round_out(self, self);
}
Tensor& trunc_(Tensor& self) {
  return at::trunc_out(self, self);
}
Tensor& sqrt_(Tensor& self) {
  return at::sqrt_out(self, self);
}

// \WRAP OPS #################################################################

bool _unops_out_cpu(unary_type* f, Tensor& result, const Tensor& self) {
  if (result.is_contiguous() && self.is_contiguous()) {
    result.resize_(self.sizes());
    f(result, self);
    return true;
  }
  return false;
}

// CPU OPS ###################################################################

Tensor& _ceil_out_cpu(Tensor& result, const Tensor& self) {
  return _unops_out_cpu(ceilImpl, result, self) ? result
                                                : at::_ceil_out(result, self);
}
Tensor& _floor_out_cpu(Tensor& result, const Tensor& self) {
  return _unops_out_cpu(floorImpl, result, self) ? result
                                                 : at::_floor_out(result, self);
}
Tensor& _round_out_cpu(Tensor& result, const Tensor& self) {
  return _unops_out_cpu(roundImpl, result, self) ? result
                                                 : at::_round_out(result, self);
}
Tensor& _trunc_out_cpu(Tensor& result, const Tensor& self) {
  return _unops_out_cpu(truncImpl, result, self) ? result
                                                 : at::_trunc_out(result, self);
}
Tensor& _sqrt_out_cpu(Tensor& result, const Tensor& self) {
  return _unops_out_cpu(sqrtImpl, result, self) ? result
                                                : at::_sqrt_out(result, self);
}

// \CPU OPS #################################################################

// CUDA OPS #################################################################

Tensor& _ceil_out_cuda(Tensor& result, const Tensor& self) {
  return at::_ceil_out(result, self);
}
Tensor& _floor_out_cuda(Tensor& result, const Tensor& self) {
  return at::_floor_out(result, self);
}
Tensor& _round_out_cuda(Tensor& result, const Tensor& self) {
  return at::_round_out(result, self);
}
Tensor& _trunc_out_cuda(Tensor& result, const Tensor& self) {
  return at::_trunc_out(result, self);
}
Tensor& _sqrt_out_cuda(Tensor& result, const Tensor& self) {
  return at::_sqrt_out(result, self);
}

// \CUDA OPS ################################################################
}} // namespace at::native
