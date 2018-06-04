#include "ATen/ATen.h"
#include "ATen/NativeFunctions.h"
#include "ATen/Dispatch.h"
#include "ATen/cuda/CUDAApplyUtils.cuh"
#include "ATen/cuda/CUDATensorMethods.cuh"
#include "ATen/cuda/CUDATypeConversion.cuh"
#include "THCUNN/THCHalfAutoNumerics.cuh"

namespace at { namespace native {

template <typename scalar_t>
void hardshrink_cuda_kernel(Tensor& out_tensor, Tensor& lambd_tensor) {
  at::cuda::CUDA_tensor_apply2<scalar_t, scalar_t>(
      out_tensor,
      lambd_tensor,
      [] __device__ (scalar_t& out_tensor_val,
         scalar_t& lambd_tensor_val,
         bool early_exit) {
           if (out_tensor_val >= -lambd_tensor_val && out_tensor_val <= lambd_tensor_val) {
             out_tensor_val = ScalarConvert<double, scalar_t>::to(0.0);
           }
  });
}

template <typename scalar_t>
void hardshrink_backward_cuda_kernel(Tensor& out_tensor, Tensor& lambd_tensor, const Tensor& self) {
  at::cuda::CUDA_tensor_apply3<scalar_t, scalar_t, scalar_t>(
      out_tensor,
      lambd_tensor,
      self,
      [] __device__ (scalar_t& out_tensor_val,
         scalar_t& lambd_tensor_val,
         scalar_t& self_val) {
           if (self_val >= -lambd_tensor_val && self_val <= lambd_tensor_val) {
             out_tensor_val = ScalarConvert<double, scalar_t>::to(0.0);
           }
  });
}

Tensor hardshrink_cuda(const Tensor & self, Scalar lambd) {
  auto lambd_tensor = at::zeros_like(self).fill_(lambd);
  auto out_tensor = self.clone();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(self.type(), "hardshrink_cuda", [&] {
    using cuda_scalar_t = cuda::into_type<scalar_t>;
    hardshrink_cuda_kernel<cuda_scalar_t>(out_tensor, lambd_tensor);
  });
  return out_tensor;
}

Tensor hardshrink_backward_cuda(const Tensor & grad, const Tensor & self, Scalar lambd) {
  auto lambd_tensor = at::zeros_like(self).fill_(lambd);
  auto out_tensor = grad.clone();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(self.type(), "hardshrink_backward_cuda", [&] {
    using cuda_scalar_t = cuda::into_type<scalar_t>;
    hardshrink_backward_cuda_kernel<cuda_scalar_t>(out_tensor, lambd_tensor, self);
  });
  return out_tensor;
}

}}  // namespace at::native
