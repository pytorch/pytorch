#include "ATen/ATen.h"
#include "ATen/NativeFunctions.h"
#include "ATen/Dispatch.h"
#include "ATen/CPUApplyUtils.h"

// #ifdef _OPENMP
// #include <omp.h>
// #endif

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

Tensor hard_shrink_cpu(const Tensor & self, const double lambda) {
  auto scalarType = self.type().scalarType();
  if (scalarType != kDouble
      && scalarType != kFloat) {
        std::stringstream ss;
        ss << "hardshrink only accepts types "
          << "(Double, Float), "
          << "tensor has invalid type = "
          << scalarType;
        throw std::runtime_error(ss.str());
  }

  auto lambda_t = at::zeros_like(self).fill_(lambda);
  auto zero_t = at::zeros_like(self);
  auto out_t = self.clone();
  AT_DISPATCH_FLOATING_TYPES(self.type(), "hard_shrink_cpu", [&] {
    at::CPU_tensor_apply3<scalar_t, scalar_t, scalar_t>(
        out_t,
        lambda_t,
        zero_t,
        [](scalar_t& out_t_val,
           const scalar_t& lambda_t_val,
           const scalar_t& zero_t_val) {
             if (out_t_val >= -lambda_t_val && out_t_val <= lambda_t_val) {
               out_t_val = zero_t_val;
             }
    });
  });
  return out_t;
}

Tensor hard_shrink_backward_cpu(const Tensor & grad, const Tensor & self, const double lambda) {
  auto lambda_t = at::zeros_like(self).fill_(lambda);
  auto zero_t = at::zeros_like(self);
  auto out_t = grad.clone();
  AT_DISPATCH_FLOATING_TYPES(self.type(), "hard_shrink_backward_cpu", [&] {
    at::CPU_tensor_apply4<scalar_t, scalar_t, scalar_t, scalar_t>(
        out_t,
        lambda_t,
        zero_t,
        self,
        [](scalar_t& out_t_val,
           const scalar_t& lambda_t_val,
           const scalar_t& zero_t_val,
           const scalar_t& self_val) {
             if (self_val >= -lambda_t_val && self_val <= lambda_t_val) {
               out_t_val = zero_t_val;
             }
    });
  });
  return out_t;
}

}}  // namespace at::native
