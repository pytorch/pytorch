#include <ATen/native/Lerp.h>

#include <ATen/ATen.h>
#include <ATen/CPUApplyUtils.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Dispatch.h>
#include <ATen/ExpandUtils.h>

namespace at {
namespace native {

Tensor& lerp_cpu_tensor_out(Tensor& result, const Tensor& self,
                            const Tensor& end, const Tensor& weight) {
  Tensor b_self, b_end, b_weight;
  TORCH_CHECK(weight.dim() <= std::max(self.dim(), end.dim()),
           "weight should be of dimension max(self.dim(), end.dim()) or lesser");
  std::tie(b_self, b_end, b_weight) = expand_outplace(self, end, weight, "lerp_out_cpu");
  result.resize_as_(b_self);
  lerp_kernel_tensor_weight(kCPU, result, b_self, b_end, b_weight);
  return result;
}

Tensor& lerp_cpu_scalar_out(Tensor& result, const Tensor& self,
                            const Tensor& end, Scalar weight) {
  Tensor b_self, b_end;
  std::tie(b_self, b_end) = expand_outplace(self, end, "lerp_out_cpu");
  result.resize_as_(b_self);
  lerp_kernel_scalar_weight(kCPU, result, b_self, b_end, weight);
  return result;
}

Tensor& lerp_cpu_tensor_(Tensor& self, const Tensor& end, const Tensor& weight) {
  Tensor b_self, b_end, b_weight;
  std::tie(b_self, b_end, b_weight) = expand_outplace(self, end, weight, "lerp__cpu");
  TORCH_CHECK(b_self.sizes() == self.sizes(),
           "output with shape ", self.sizes(),
           " doesn't match the broadcast shape ", b_self.sizes());
  TORCH_CHECK(weight.dim() <= std::max(self.dim(), end.dim()),
           "weight should be of dimension max(self.dim(), end.dim()) or lesser");
  lerp_kernel_tensor_weight(kCPU, self, b_self, b_end, b_weight);
  return self;
}

Tensor& lerp_cpu_scalar_(Tensor& self, const Tensor& end, Scalar weight) {
  Tensor b_self, b_end;
  std::tie(b_self, b_end) = expand_outplace(self, end, "lerp__cpu");
  TORCH_CHECK(b_self.sizes() == self.sizes(),
           "output with shape ", self.sizes(),
           " doesn't match the broadcast shape ", b_self.sizes());
  lerp_kernel_scalar_weight(kCPU, self, b_self, b_end, weight);
  return self;
}

Tensor lerp_cpu_tensor(const Tensor& self, const Tensor& end, const Tensor& weight) {
  Tensor b_self, b_end, b_weight;
  TORCH_CHECK(weight.dim() <= std::max(self.dim(), end.dim()),
           "weight should be of dimension max(self.dim(), end.dim()) or lesser");
  std::tie(b_self, b_end, b_weight) = expand_outplace(self, end, weight, "lerp_cpu");
  Tensor result = at::empty_like(b_self);
  lerp_kernel_tensor_weight(kCPU, result, b_self, b_end, b_weight);
  return result;
}

Tensor lerp_cpu_scalar(const Tensor& self, const Tensor& end, Scalar weight) {
  Tensor b_self, b_end;
  std::tie(b_self, b_end) = expand_outplace(self, end, "lerp_cpu");
  Tensor result = at::empty_like(b_self);
  lerp_kernel_scalar_weight(kCPU, result, b_self, b_end, weight);
  return result;
}

DEFINE_DISPATCH(lerp_kernel_scalar_weight);
DEFINE_DISPATCH(lerp_kernel_tensor_weight);

} // namespace native
} // namespace at
