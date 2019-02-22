#include <ATen/ATen.h>
#include <ATen/CPUApplyUtils.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Dispatch.h>
#include <ATen/ExpandUtils.h>

namespace {
template <typename scalar_t>
void lerp_cpu(
    at::Tensor& ret,
    const at::Tensor& self,
    const at::Tensor& end,
    const at::Tensor& weight) {
  at::CPU_tensor_apply4<scalar_t, scalar_t, scalar_t, scalar_t>(
      ret,
      self,
      end,
      weight,
      [=](scalar_t& ret_val,
         const scalar_t& self_val,
         const scalar_t& end_val,
         const scalar_t& weight_val) {
        ret_val = self_val + weight_val * (end_val - self_val);
      });
}

template <typename scalar_t>
void lerp_cpu(
    at::Tensor& ret,
    const at::Tensor& self,
    const at::Tensor& end,
    scalar_t weight_val) {
  at::CPU_tensor_apply3<scalar_t, scalar_t, scalar_t>(
      ret,
      self,
      end,
      [=](scalar_t& ret_val,
         const scalar_t& self_val,
         const scalar_t& end_val) {
        ret_val = self_val + weight_val * (end_val - self_val);
      });
}
} // namespace

namespace at {
namespace native {

Tensor lerp_cpu_tensor(const Tensor& self, const Tensor& end, const Tensor& weight) {
  Tensor b_self, b_end, b_weight;
  AT_CHECK(weight.dim() <= std::max(self.dim(), end.dim()),
           "weight should be of dimension max(self.dim(), end.dim()) or lesser");
  std::tie(b_self, b_end, b_weight) = expand_outplace(self, end, weight, "lerp");
  Tensor ret = at::empty_like(b_self);
  AT_DISPATCH_FLOATING_TYPES(ret.type(), "lerp", [&]{
    lerp_cpu<scalar_t>(ret, b_self, b_end, b_weight);
  });
  return ret;
}

Tensor lerp_cpu_scalar(const Tensor& self, const Tensor& end, Scalar weight) {
  Tensor b_self, b_end;
  std::tie(b_self, b_end) = expand_outplace(self, end, "lerp");
  Tensor ret = at::empty_like(b_self);
  AT_DISPATCH_FLOATING_TYPES(ret.type(), "lerp", [&]{
    lerp_cpu<scalar_t>(ret, b_self, b_end, weight.to<scalar_t>());
  });
  return ret;
}

Tensor& lerp_out(Tensor& result, const Tensor& self, const Tensor& end, const Tensor& weight) {
  Tensor result_tmp;
  result_tmp = at::lerp(self, end, weight);
  result.resize_as_(result_tmp).copy_(result_tmp);
  return result;
}

Tensor& lerp_out(Tensor& result, const Tensor& self, const Tensor& end, Scalar weight) {
  Tensor result_tmp;
  result_tmp = at::lerp(self, end, weight);
  result.resize_as_(result_tmp).copy_(result_tmp);
  return result;
}

Tensor& lerp_(Tensor& self, const Tensor& end, const Tensor& weight) {
  Tensor result_tmp, b_self, b_end, b_weight;
  std::tie(b_self, b_end, b_weight) = expand_outplace(self, end, weight, "lerp");
  AT_CHECK(b_self.sizes() == self.sizes(),
           "output with shape ", self.sizes(),
           " doesn't match the broadcast shape ", b_self.sizes());
  AT_CHECK(weight.dim() <= std::max(self.dim(), end.dim()),
           "weight should be of dimension max(self.dim(), end.dim()) or lesser");
  result_tmp = at::lerp(b_self, b_end, b_weight);
  self.copy_(result_tmp);
  return self;
}

Tensor& lerp_(Tensor& self, const Tensor& end, Scalar weight) {
  Tensor result_tmp, b_self, b_end;
  std::tie(b_self, b_end) = expand_outplace(self, end, "lerp");
  AT_CHECK(b_self.sizes() == self.sizes(),
           "output with shape ", self.sizes(),
           " doesn't match the broadcast shape ", b_self.sizes());
  result_tmp = at::lerp(b_self, b_end, weight);
  self.copy_(result_tmp);
  return self;
}

} // namespace native
} // namespace at
