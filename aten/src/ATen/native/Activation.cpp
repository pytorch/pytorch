#include "ATen/ATen.h"
#include "ATen/NativeFunctions.h"
#include "ATen/Dispatch.h"
#include "ATen/CPUApplyUtils.h"
#include "ATen/Half.h"

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

Tensor rrelu(const Tensor & self, Scalar lower, Scalar upper, bool training, Generator* generator) {
  return at::rrelu_with_noise(self, self.type().tensor(), lower, upper, training, generator);
}

Tensor & rrelu_(Tensor & self, Scalar lower, Scalar upper, bool training, Generator* generator) {
  return at::rrelu_with_noise_(self, self.type().tensor(), lower, upper, training, generator);
}

Tensor hardshrink_cpu(const Tensor & self, Scalar lambd) {
  auto lambd_tensor = at::zeros_like(self).fill_(lambd.toTensor());
  auto out_tensor = self.clone();
  AT_DISPATCH_FLOATING_TYPES(self.type(), "hardshrink_cpu", [&] {
    at::CPU_tensor_apply2<scalar_t, scalar_t>(
        out_tensor,
        lambd_tensor,
        [](scalar_t& out_tensor_val,
           scalar_t& lambd_tensor_val) {
             if (out_tensor_val >= -lambd_tensor_val && out_tensor_val <= lambd_tensor_val) {
               out_tensor_val = convert<scalar_t, double>(0.0);
             }
    });
  });
  return out_tensor;
}

Tensor hardshrink_backward_cpu(const Tensor & grad, const Tensor & self, Scalar lambd) {
  auto lambd_tensor = at::zeros_like(self).fill_(lambd);
  auto out_tensor = grad.clone();
  AT_DISPATCH_FLOATING_TYPES(self.type(), "hardshrink_backward_cpu", [&] {
    at::CPU_tensor_apply3<scalar_t, scalar_t, scalar_t>(
        out_tensor,
        lambd_tensor,
        self,
        [](scalar_t& out_tensor_val,
           scalar_t& lambd_tensor_val,
           scalar_t& self_val) {
             if (self_val >= -lambd_tensor_val && self_val <= lambd_tensor_val) {
               out_tensor_val = convert<scalar_t, double>(0.0);
             }
    });
  });
  return out_tensor;
}

}}  // namespace at::native
