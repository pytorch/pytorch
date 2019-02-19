#include <ATen/NativeFunctions.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/Dispatch.h>
#include <ATen/ExpandUtils.h>

namespace {
template <typename scalar_t>
void lerp_cuda(
    at::Tensor& ret,
    const at::Tensor& self,
    const at::Tensor& end,
    const at::Tensor& weight) {
  at::cuda::CUDA_tensor_apply4<scalar_t, scalar_t, scalar_t, scalar_t>(
      ret,
      self,
      end,
      weight,
      [=] __device__(
         scalar_t& ret_val,
         const scalar_t& self_val,
         const scalar_t& end_val,
         const scalar_t& weight_val) {
        ret_val = self_val + weight_val * (end_val - self_val);
      });
}

template <typename scalar_t>
void lerp_cuda(
    at::Tensor& ret,
    const at::Tensor& self,
    const at::Tensor& end,
    scalar_t weight_val) {
  at::cuda::CUDA_tensor_apply3<scalar_t, scalar_t, scalar_t>(
      ret,
      self,
      end,
      [=] __device__(
         scalar_t& ret_val,
         const scalar_t& self_val,
         const scalar_t& end_val) {
        ret_val = self_val + weight_val * (end_val - self_val);
      });
}
} // namespace

namespace at {
namespace native {

Tensor lerp_cuda_tensor(const Tensor& self, const Tensor& end, const Tensor& weight) {
  Tensor b_self, b_end, b_weight;
  AT_CHECK(weight.dim() <= std::max(self.dim(), end.dim()),
           "weight should be of dimension max(self.dim(), end.dim()) or lesser");
  std::tie(b_self, b_end, b_weight) = expand_outplace(self, end, weight, "lerp");
  Tensor ret = at::empty_like(b_self);
  AT_DISPATCH_FLOATING_TYPES(ret.type(), "lerp", [&]{
    lerp_cuda<scalar_t>(ret, b_self, b_end, b_weight);
  });
  return ret;
}

Tensor lerp_cuda_scalar(const Tensor& self, const Tensor& end, Scalar weight) {
  Tensor b_self, b_end;
  std::tie(b_self, b_end) = expand_outplace(self, end, "lerp");
  Tensor ret = at::empty_like(b_self);
  AT_DISPATCH_FLOATING_TYPES(ret.type(), "lerp", [&]{
    lerp_cuda<scalar_t>(ret, b_self, b_end, weight.to<scalar_t>());
  });
  return ret;
}

} // namespace native
} // namespace at
