#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/TensorUtils.h>
#include <ATen/native/PointwiseOps.h>
#include <ATen/native/TensorIterator.h>

namespace at {
namespace native {

DEFINE_DISPATCH(binary_cross_entropy_backward_stub);

namespace {

// Returns a contiguous tensor if the source tensor
// is defined. Otherwise returns the undefined
// source tensor unmodified.
inline Tensor optional_contiguous(const Tensor& source) {
  return source.defined() ? source.contiguous() : source;
}

// Returns the address of the first element of a tensor
// or nullptr if the tensor is undefined.
template <typename scalar_t>
inline scalar_t* optional_data(const Tensor& source) {
  return source.defined() ? source.data_ptr<scalar_t>() : nullptr;
}

#define EPS 1e-12

template <typename scalar_t>
static inline scalar_t safe_log(scalar_t a) {
  if (a == scalar_t(0)) {
    return std::log(EPS);          
  }
  return std::log(a);    
}

template <typename scalar_t>
static void binary_cross_entropy_out_frame(
    Tensor& output,
    const Tensor& input,
    const Tensor& target,
    const Tensor& weight,
    int64_t reduction) {
  const scalar_t* input_data = input.data_ptr<scalar_t>();
  const scalar_t* target_data = target.data_ptr<scalar_t>();
  auto weight_contiguous = optional_contiguous(weight);
  const scalar_t* weight_data = optional_data<scalar_t>(weight_contiguous);

  if (reduction == Reduction::None) {
    output.resize_(input.sizes());
    scalar_t* output_data = output.data_ptr<scalar_t>();
    at::parallel_for(0, input.numel(), 1, [&](int64_t start, int64_t end) {
      for (int64_t i = start; i < end; i++) {
        scalar_t x = input_data[i];
        scalar_t y = target_data[i];
        TORCH_CHECK(x >= scalar_t(0) && x <= scalar_t(1), "input valus should be between 0~1");
        scalar_t w = weight_data != nullptr ? weight_data[i] : static_cast<scalar_t>(1);
        output_data[i] = -(safe_log<scalar_t>(x) * y + safe_log<scalar_t>(scalar_t(1) - x) * (scalar_t(1) - y)) * w;
      }
    });
  } else {
    output.resize_({});
    scalar_t* output_data = output.data_ptr<scalar_t>();
    *output_data = scalar_t(0);
    for (int64_t i = 0; i < input.numel(); i++) {
      scalar_t x = input_data[i];
      scalar_t y = target_data[i];
      TORCH_CHECK(x >= scalar_t(0) && x <= scalar_t(1), "input valus should be between 0~1");
      scalar_t w = weight_data != nullptr ? weight_data[i] : static_cast<scalar_t>(1);
      *output_data -= (safe_log<scalar_t>(x) * y + safe_log<scalar_t>(scalar_t(1) - x) * (scalar_t(1) - y)) * w;
    }
    if (reduction == Reduction::Mean) {
      *output_data /= input.numel();
    }
  }
}

void binary_cross_entropy_out_cpu_template(
    Tensor& output,
    const Tensor& input,
    const Tensor& target,
    const Tensor& weight,
    int64_t reduction) {
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.scalar_type(), "binary_cross_entropy_out_frame", [&] {
        binary_cross_entropy_out_frame<scalar_t>(
            output,
            input.contiguous(),
            target.contiguous(),
            weight,
            reduction);
  });
}

} // namespace

Tensor& binary_cross_entropy_out_cpu(
    Tensor& output,
    const Tensor& input,
    const Tensor& target,
    const Tensor& weight,
    int64_t reduction) {
  binary_cross_entropy_out_cpu_template(output, input, target, weight, reduction);
  return output; 
}

Tensor binary_cross_entropy_cpu(
    const Tensor& input,
    const Tensor& target,
    const Tensor& weight,
    int64_t reduction) {
  auto output = at::empty({0}, input.options());
  binary_cross_entropy_out_cpu_template(output, input, target, weight, reduction);
  return output;
}

Tensor binary_cross_entropy_backward_cpu(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& target,
    const Tensor& weight,
    int64_t reduction) {
  Tensor grad_input = at::zeros_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT); 
  return at::native::binary_cross_entropy_backward_out_cpu(grad_input,
    grad_output, input, target, weight, reduction);
}

Tensor& binary_cross_entropy_backward_out_cpu(
    Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& target,
    const Tensor& weight,
    int64_t reduction) {
  auto norm = reduction == Reduction::Mean ? 1. / input.numel() : 1.;
  auto iter = at::TensorIterator();
  iter.set_check_mem_overlap(true);
  iter.add_output(grad_input);
  iter.add_input(input);
  iter.add_input(target);
  iter.add_input(grad_output);
  iter.build();
  binary_cross_entropy_backward_stub(iter.device_type(), iter, norm);
  if (weight.defined()) {
    grad_input.mul_(weight);
  }
  return grad_input;
}

}} // namespace at::native
