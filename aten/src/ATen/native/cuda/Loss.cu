#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>

constexpr float EPSILON = 1e-12;

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

template<typename scalar_t>
void binary_cross_entropy_out_kernel(Tensor& loss, const Tensor& input, const Tensor& target) {
  at::cuda::CUDA_tensor_apply3<scalar_t, scalar_t, scalar_t>(
    loss,
    input,
    target,
    [] __device__(
      scalar_t& loss_val,
      const scalar_t& input_val,
      const scalar_t& target_val
    ) {
      const scalar_t zero = 0;
      const scalar_t one = 1;
      const scalar_t neg_100 = -100;

      CUDA_KERNEL_ASSERT(input_val >= zero && input_val <= one);

      scalar_t log_input_val = log(input_val);
      scalar_t log_1_minus_input_val = log(one - input_val);

      log_input_val = max(log_input_val, neg_100);
      log_1_minus_input_val = max(log_1_minus_input_val, neg_100);

      loss_val = ((target_val - one) * log_1_minus_input_val) - (target_val * log_input_val);
    }
  );
}

template<typename scalar_t>
void binary_cross_entropy_backward_out_kernel(Tensor& grad_input, const Tensor& grad, const Tensor& input, const Tensor& target) {
  at::cuda::CUDA_tensor_apply4<scalar_t, scalar_t, scalar_t, scalar_t>(
    grad_input,
    grad,
    input,
    target,
    [] __device__(
      scalar_t& grad_input_val,
      const scalar_t& grad_val,
      const scalar_t& input_val,
      const scalar_t& target_val
    ) {
      const scalar_t one = 1;
      const scalar_t epsilon = EPSILON;

      scalar_t grad_input_denominator = max(
        (one - input_val) * input_val,
        epsilon
      );

      grad_input_val = grad_val * (input_val - target_val) / grad_input_denominator;
    }
  );
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

Tensor binary_cross_entropy_cuda(const Tensor& input, const Tensor& target, const Tensor& weight, int64_t reduction) {
    Tensor loss = at::empty_like(input);
    return at::native::binary_cross_entropy_out_cuda(loss, input, target, weight, reduction);
}

Tensor& binary_cross_entropy_out_cuda(Tensor& loss, const Tensor& input, const Tensor& target, const Tensor& weight, int64_t reduction) {
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "binary_cross_entropy_out_cuda", [&]() {
    binary_cross_entropy_out_kernel<scalar_t>(loss, input, target);
  });
  if (weight.defined()) {
    loss.mul_(weight);
  }

  if (reduction != at::Reduction::None) {
    Tensor loss_reduced;
    if (reduction == at::Reduction::Mean) {
      loss_reduced = loss.mean();
    } else if (reduction == at::Reduction::Sum) {
      loss_reduced = loss.sum();
    }
    loss.resize_as_(loss_reduced).copy_(loss_reduced);
  }

  return loss;
}

Tensor binary_cross_entropy_backward_cuda(const Tensor& grad, const Tensor& input, const Tensor& target, const Tensor& weight, int64_t reduction) {
  Tensor grad_input = at::empty_like(input);
  return at::native::binary_cross_entropy_backward_out_cuda(grad_input, grad, input, target, weight, reduction);
}

Tensor& binary_cross_entropy_backward_out_cuda(Tensor& grad_input, const Tensor& grad, const Tensor& input, const Tensor& target, const Tensor& weight, int64_t reduction) {
  Tensor grad_expand = grad.expand_as(input);
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "binary_cross_entropy_backward_out_cuda", [&]() {
    binary_cross_entropy_backward_out_kernel<scalar_t>(grad_input, grad_expand, input, target);
  });

  if (weight.defined()) {
    grad_input.mul_(weight);
  }
  if (reduction == at::Reduction::Mean) {
    grad_input.div_(input.numel());
  }
  return grad_input;
}

}}  // namespace at::native
