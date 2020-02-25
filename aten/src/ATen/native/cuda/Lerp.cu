#include <ATen/NativeFunctions.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/Dispatch.h>
#include <ATen/ExpandUtils.h>

namespace {
template <typename scalar_t>
void lerp_cuda(at::Tensor& ret, const at::Tensor& self, const at::Tensor& end, const at::Tensor& weight) {
  at::cuda::CUDA_tensor_apply4<scalar_t, scalar_t, scalar_t, scalar_t>(
      ret, self, end, weight,
      [] __device__(
         scalar_t& ret_val,
         const scalar_t& self_val,
         const scalar_t& end_val,
         const scalar_t& weight_val) {
        ret_val = (weight_val < 0.5) ?
            self_val + weight_val * (end_val - self_val) : end_val - (end_val - self_val) * (1 - weight_val);
      });
}

template <typename scalar_t>
void lerp_cuda(at::Tensor& ret, const at::Tensor& self, const at::Tensor& end, scalar_t weight_val) {
  at::cuda::CUDA_tensor_apply3<scalar_t, scalar_t, scalar_t>(
      ret, self, end,
      [=] __device__(
         scalar_t& ret_val,
         const scalar_t& self_val,
         const scalar_t& end_val) {
        ret_val = (weight_val < 0.5) ?
            self_val + weight_val * (end_val - self_val) : end_val - (end_val - self_val) * (1 - weight_val);
      });
}
} // namespace

namespace at {
namespace native {

Tensor& lerp_cuda_tensor_out(Tensor& result, const Tensor& self,
                            const Tensor& end, const Tensor& weight) {
  Tensor b_self, b_end, b_weight;
  TORCH_CHECK(weight.dim() <= std::max(self.dim(), end.dim()),
           "weight should be of dimension max(self.dim(), end.dim()) or lesser");
  std::tie(b_self, b_end, b_weight) = expand_outplace(self, end, weight, "lerp_out_cuda");
  result.resize_as_(b_self);
  AT_DISPATCH_FLOATING_TYPES(self.scalar_type(), "lerp_out_cuda", [&]{
    lerp_cuda<scalar_t>(result, b_self, b_end, b_weight);
  });
  return result;
}

Tensor& lerp_cuda_scalar_out(Tensor& result, const Tensor& self,
                            const Tensor& end, Scalar weight) {
  Tensor b_self, b_end;
  std::tie(b_self, b_end) = expand_outplace(self, end, "lerp_out_cuda");
  result.resize_as_(b_self);
  AT_DISPATCH_FLOATING_TYPES(self.scalar_type(), "lerp_out_cuda", [&]{
    lerp_cuda<scalar_t>(result, b_self, b_end, weight.to<scalar_t>());
  });
  return result;
}

Tensor& lerp_cuda_tensor_(Tensor& self, const Tensor& end, const Tensor& weight) {
  Tensor b_self, b_end, b_weight;
  std::tie(b_self, b_end, b_weight) = expand_outplace(self, end, weight, "lerp__cuda");
  TORCH_CHECK(b_self.sizes() == self.sizes(),
           "output with shape ", self.sizes(),
           " doesn't match the broadcast shape ", b_self.sizes());
  TORCH_CHECK(weight.dim() <= std::max(self.dim(), end.dim()),
           "weight should be of dimension max(self.dim(), end.dim()) or lesser");
  AT_DISPATCH_FLOATING_TYPES(self.scalar_type(), "lerp__cuda", [&]{
    lerp_cuda<scalar_t>(self, b_self, b_end, b_weight);
  });
  return self;
}

Tensor& lerp_cuda_scalar_(Tensor& self, const Tensor& end, Scalar weight) {
  Tensor b_self, b_end;
  std::tie(b_self, b_end) = expand_outplace(self, end, "lerp__cuda");
  TORCH_CHECK(b_self.sizes() == self.sizes(),
           "output with shape ", self.sizes(),
           " doesn't match the broadcast shape ", b_self.sizes());
  AT_DISPATCH_FLOATING_TYPES(self.scalar_type(), "lerp__cuda", [&]{
    lerp_cuda<scalar_t>(self, b_self, b_end, weight.to<scalar_t>());
  });
  return self;
}

Tensor lerp_cuda_tensor(const Tensor& self, const Tensor& end, const Tensor& weight) {
  Tensor b_self, b_end, b_weight;
  TORCH_CHECK(weight.dim() <= std::max(self.dim(), end.dim()),
           "weight should be of dimension max(self.dim(), end.dim()) or lesser");
  std::tie(b_self, b_end, b_weight) = expand_outplace(self, end, weight, "lerp_cuda");
  Tensor result = at::empty_like(b_self, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  AT_DISPATCH_FLOATING_TYPES(self.scalar_type(), "lerp_cuda", [&]{
    lerp_cuda<scalar_t>(result, b_self, b_end, b_weight);
  });
  return result;
}

Tensor lerp_cuda_scalar(const Tensor& self, const Tensor& end, Scalar weight) {
  Tensor b_self, b_end;
  std::tie(b_self, b_end) = expand_outplace(self, end, "lerp_cuda");
  Tensor result = at::empty_like(b_self, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  AT_DISPATCH_FLOATING_TYPES(self.scalar_type(), "lerp_cuda", [&]{
    lerp_cuda<scalar_t>(result, b_self, b_end, weight.to<scalar_t>());
  });
  return result;
}

} // namespace native
} // namespace at
