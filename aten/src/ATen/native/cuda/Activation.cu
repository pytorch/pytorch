#include "ATen/ATen.h"
#include "ATen/NativeFunctions.h"
#include "ATen/Dispatch.h"
#include "ATen/cuda/CUDAApplyUtils.cuh"

namespace at { namespace native {

template <typename scalar_t>
void hardshrink_cuda_kernel(const Tensor& self, Tensor& out_tensor, scalar_t lambd) {
  at::cuda::CUDA_tensor_apply2<scalar_t, scalar_t>(
    self,
    out_tensor,
    [=] __device__ (
      scalar_t& self_val,
      scalar_t& out_tensor_val) {
        out_tensor_val = (self_val >= -lambd && self_val <= lambd) ? scalar_t(0) : self_val;
  });
}

template <typename scalar_t>
void hardshrink_backward_cuda_kernel(const Tensor& self, Tensor& out_tensor, scalar_t lambd, const Tensor& grad) {
  at::cuda::CUDA_tensor_apply3<scalar_t, scalar_t, scalar_t>(
    self,
    grad,
    out_tensor,
    [=] __device__ (
      scalar_t& self_val,
      scalar_t& grad_val,
      scalar_t& out_tensor_val) {
        out_tensor_val = (self_val >= -lambd && self_val <= lambd) ? scalar_t(0) : grad_val;
  });
}

Tensor hardshrink_cuda(const Tensor & self, Scalar lambd) {
  auto out_tensor = at::empty_like(self);
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(self.type(), "hardshrink_cuda", [&] {
    hardshrink_cuda_kernel<scalar_t>(self, out_tensor, lambd.to<scalar_t>());
  });
  return out_tensor;
}

Tensor hardshrink_backward_cuda(const Tensor & grad, const Tensor & self, Scalar lambd) {
  auto out_tensor = at::empty_like(grad);
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(self.type(), "hardshrink_backward_cuda", [&] {
    hardshrink_backward_cuda_kernel<scalar_t>(self, out_tensor, lambd.to<scalar_t>(), grad);
  });
  return out_tensor;
}

}}  // namespace at::native
