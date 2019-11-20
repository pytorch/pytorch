#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>


namespace {

using namespace at;

template<typename scalar_t>
void kl_div_backward_kernel(const Tensor& grad_input, const Tensor& target, const Tensor& grad) {
  at::cuda::CUDA_tensor_apply3<scalar_t, scalar_t, scalar_t>(
      grad_input,
      target,
      grad,
      [] __device__(
          scalar_t& grad_input_val, const scalar_t& target_val, const scalar_t& grad_val) {
        if (target_val > 0) {
          grad_input_val = -target_val * grad_val;
        }
      });
}
} // namespace

namespace at { namespace native {

Tensor kl_div_backward_cuda(const Tensor& grad, const Tensor& input, const Tensor& target, int64_t reduction) {
  auto grad_input = at::zeros_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  Tensor grad_expand = grad.expand_as(input);
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "kl_div_backward_cuda", [&]() {
    kl_div_backward_kernel<scalar_t>(grad_input, target, grad_expand);
  });
  if (reduction == at::Reduction::Mean) {
    return grad_input / input.numel();
  }
  return grad_input;
}
}}  // namespace at::native
