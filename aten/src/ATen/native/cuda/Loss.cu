#include "ATen/ATen.h"
#include "ATen/NativeFunctions.h"
#include "ATen/Dispatch.h"
#include "ATen/cuda/CUDAApplyUtils.cuh"
#include <THC/THCNumerics.cuh>


namespace at { namespace native {

Tensor kl_div_loss_backward_cuda(const Tensor& grad, const Tensor& input, const Tensor& target, int64_t reduction) {
  Tensor grad_input = grad.type().zeros_like(input);
  Tensor grad_expand = grad.expand_as(input);
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.type(), "kl_div_loss_backward", [&]() {
    at::cuda::CUDA_tensor_apply3<scalar_t, scalar_t, scalar_t>(
        grad_input,
        target,
        grad_expand,
        [] __device__(
            scalar_t& grad_input_val, const scalar_t& target_val, const scalar_t& grad_val) {
          if (target_val > 0) {
            grad_input_val = -target_val * grad_val;
          }
        });
  });
  return grad_input;
}
}}  // namespace at::native
