#include "ATen/ATen.h"
#include "ATen/CPUApplyUtils.h"
#include "ATen/Dispatch.h"
#include "ATen/NativeFunctions.h"
#include "ATen/core/Half.h"

namespace at { namespace native {

static const double SELU_ALPHA = 1.6732632423543772848170429916717;
static const double SELU_SCALE = 1.0507009873554804934193349852946;

Tensor relu(const Tensor & self) {
  return self.clamp_min(0.0);
}

Tensor & relu_(Tensor & self) {
  return self.clamp_min_(0.0);
}

Tensor selu(const Tensor & self) {
  return at::elu(self, SELU_ALPHA, SELU_SCALE);
}

Tensor & selu_(Tensor & self) {
  return at::elu_(self, SELU_ALPHA, SELU_SCALE);
}

Tensor celu(const Tensor & self, Scalar alpha) {
  double inv_alpha = 1. / alpha.to<double>();
  return at::elu(self, 1.0, alpha, Scalar(inv_alpha));
}

Tensor & celu_(Tensor & self, Scalar alpha) {
  double inv_alpha = 1. / alpha.to<double>();
  return at::elu_(self, 1.0, alpha, Scalar(inv_alpha));
}

Tensor rrelu(const Tensor & self, Scalar lower, Scalar upper, bool training, Generator* generator) {
  return at::rrelu_with_noise(self, self.type().tensor(), lower, upper, training, generator);
}

Tensor & rrelu_(Tensor & self, Scalar lower, Scalar upper, bool training, Generator* generator) {
  return at::rrelu_with_noise_(self, self.type().tensor(), lower, upper, training, generator);
}

Tensor hardshrink_cpu(const Tensor & self, Scalar lambd) {
  auto out_tensor = at::empty_like(self);
  AT_DISPATCH_FLOATING_TYPES(self.type(), "hardshrink_cpu", [&] {
    auto lambd_val = lambd.to<scalar_t>();
    at::CPU_tensor_apply2<scalar_t, scalar_t>(
      self,
      out_tensor,
      [&](
        scalar_t& self_val,
        scalar_t& out_tensor_val) {
          out_tensor_val = (self_val >= -lambd_val && self_val <= lambd_val) ? scalar_t(0) : self_val;
    });
  });
  return out_tensor;
}

Tensor hardshrink_backward_cpu(const Tensor & grad, const Tensor & self, Scalar lambd) {
  auto out_tensor = at::empty_like(self);
  AT_DISPATCH_FLOATING_TYPES(self.type(), "hardshrink_backward_cpu", [&] {
    auto lambd_val = lambd.to<scalar_t>();
    at::CPU_tensor_apply3<scalar_t, scalar_t, scalar_t>(
      self,
      grad,
      out_tensor,
      [&](
        scalar_t& self_val,
        scalar_t& grad_val,
        scalar_t& out_tensor_val) {
          out_tensor_val = (self_val >= -lambd_val && self_val <= lambd_val) ? scalar_t(0) : grad_val;
    });
  });
  return out_tensor;
}

}}  // namespace at::native
