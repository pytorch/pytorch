#include "ATen/NativeFunctions.h"
#include "ATen/Dispatch.h"

#include "ATen/cuda/CUDAApplyUtils.cuh"
#include "ATen/cuda/CUDATensorMethods.cuh"
#include "ATen/cuda/CUDATypeConversion.cuh"


namespace at { namespace native {

template <typename scalar_t>
void hard_shrink_cuda_kernel(at::Tensor& out_t, const at::Tensor& lambda_t, const at::Tensor& zero_t) {
  at::cuda::CUDA_tensor_apply3<scalar_t, scalar_t, scalar_t>(
      out_t,
      lambda_t,
      zero_t,
      [] __device__ (scalar_t& out_t_val,
         const scalar_t& lambda_t_val,
         const scalar_t& zero_t_val) {
           if (out_t_val >= -lambda_t_val && out_t_val <= lambda_t_val) {
             out_t_val = zero_t_val;
           }
  });
}

template <typename scalar_t>
void hard_shrink_backward_cuda_kernel(at::Tensor& out_t, const at::Tensor& lambda_t, const at::Tensor& zero_t, const at::Tensor& self) {
  at::cuda::CUDA_tensor_apply4<scalar_t, scalar_t, scalar_t, scalar_t>(
      out_t,
      lambda_t,
      zero_t,
      self,
      [] __device__ (scalar_t& out_t_val,
         const scalar_t& lambda_t_val,
         const scalar_t& zero_t_val,
         const scalar_t& self_val) {
           if (self_val >= -lambda_t_val && self_val <= lambda_t_val) {
             out_t_val = zero_t_val;
           }
  });
}

Tensor hard_shrink_cuda(const Tensor & self, Scalar lambda) {
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
  AT_DISPATCH_FLOATING_TYPES(self.type(), "hard_shrink_cuda", [&] {
    using cuda_scalar_t = cuda::type<scalar_t>;
    hard_shrink_cuda_kernel<cuda_scalar_t>(out_t, lambda_t, zero_t);
  });
  return out_t;
}

Tensor hard_shrink_backward_cuda(const Tensor & grad, const Tensor & self, Scalar lambda) {
  auto lambda_t = at::zeros_like(self).fill_(lambda);
  auto zero_t = at::zeros_like(self);
  auto out_t = grad.clone();
  AT_DISPATCH_FLOATING_TYPES(self.type(), "hard_shrink_backward_cuda", [&] {
    using cuda_scalar_t = cuda::type<scalar_t>;
    hard_shrink_backward_cuda_kernel<cuda_scalar_t>(out_t, lambda_t, zero_t, self);
  });
  return out_t;
}

}}  // namespace at::native
