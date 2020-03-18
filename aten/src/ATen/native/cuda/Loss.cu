#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/Loops.cuh>

constexpr float EPSILON = 1e-12;

namespace {

using namespace at;

void binary_cross_entropy_backward_out_kernel(Tensor& grad_input, const Tensor& grad, const Tensor& input, const Tensor& target) {
  at::TensorIterator iter;
  iter.add_output(grad_input);
  iter.add_input(grad);
  iter.add_input(input);
  iter.add_input(target);
  iter.build();
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, iter.common_dtype(), "binary_cross_entropy_backward_out_cuda", [&]() {
    AT_SKIP_BFLOAT16_IF_NOT_ROCM(scalar_t, "binary_cross_entropy_backward_out_cuda", [&] {
      at::native::gpu_kernel(iter, [] GPU_LAMBDA (
          scalar_t grad_val,
          scalar_t input_val,
          scalar_t target_val
        ) -> scalar_t {
          const scalar_t one = 1;
          const scalar_t epsilon = EPSILON;

          scalar_t grad_input_denominator = max(
            (one - input_val) * input_val,
            epsilon
          );

          return grad_val * (input_val - target_val) / grad_input_denominator;
        }
      );
    });
  });
}

} // namespace

namespace at { namespace native {

Tensor kl_div_backward_cuda(const Tensor& grad, const Tensor& input, const Tensor& target, int64_t reduction) {
  auto grad_input = at::empty_like(input);
  TensorIterator iter;
  iter.add_output(grad_input);
  iter.add_input(target);
  iter.add_input(grad);
  iter.build();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "kl_div_backward_cuda", [&]() {
    scalar_t inv = (reduction == at::Reduction::Mean) ? scalar_t(1.0 / input.numel()) : scalar_t(1.0);
    gpu_kernel(iter,
      [inv] GPU_LAMBDA (scalar_t target_val, scalar_t grad_val) {
        return (target_val > 0) ? scalar_t(-target_val * grad_val * inv) : scalar_t(0.0);
      });
  });
  return grad_input;
}

Tensor binary_cross_entropy_cuda(const Tensor& input, const Tensor& target, const Tensor& weight, int64_t reduction) {
    Tensor loss = at::empty_like(input);
    return at::native::binary_cross_entropy_out_cuda(loss, input, target, weight, reduction);
}

Tensor& binary_cross_entropy_out_cuda(Tensor& loss, const Tensor& input, const Tensor& target, const Tensor& weight, int64_t reduction) {
  Tensor loss_squeezed = at::squeeze(loss);

  TensorIterator iter;
  iter.add_output(loss_squeezed);
  iter.add_input(at::squeeze(input));
  iter.add_input(at::squeeze(target));
  iter.build();
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, iter.common_dtype(), "binary_cross_entropy_out_cuda", [&]() {
    AT_SKIP_BFLOAT16_IF_NOT_ROCM(scalar_t, "binary_cross_entropy_out_cuda", [&] {
      gpu_kernel(iter,
        [] GPU_LAMBDA (scalar_t input_val, scalar_t target_val) -> scalar_t {
          const scalar_t zero = 0;
          const scalar_t one = 1;
          const scalar_t neg_100 = -100;

          CUDA_KERNEL_ASSERT(input_val >= zero && input_val <= one);

          scalar_t log_input_val = std::log(input_val);
          scalar_t log_1_minus_input_val = std::log(one - input_val);

          log_input_val = std::max(log_input_val, neg_100);
          log_1_minus_input_val = std::max(log_1_minus_input_val, neg_100);

          return ((target_val - one) * log_1_minus_input_val) - (target_val * log_input_val);
        }
      );
    });
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
  binary_cross_entropy_backward_out_kernel(grad_input, grad_expand, input, target);

  if (weight.defined()) {
    grad_input.mul_(weight);
  }
  if (reduction == at::Reduction::Mean) {
    grad_input.div_(input.numel());
  }
  return grad_input;
}

}}  // namespace at::native
