#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/TensorMeta.h>
#include <ATen/quantized/Quantizer.h>
#include <ATen/native/Padding.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_empty_affine_quantized.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/reflection_pad1d_backward_native.h>
#include <ATen/ops/reflection_pad1d_native.h>
#include <ATen/ops/reflection_pad2d_backward_native.h>
#include <ATen/ops/reflection_pad2d_native.h>
#include <ATen/ops/reflection_pad3d_backward_native.h>
#include <ATen/ops/reflection_pad3d_native.h>
#include <ATen/ops/zeros_like.h>
#endif

namespace at::meta {

TORCH_META_FUNC(reflection_pad1d)(const Tensor& input, IntArrayRef padding) {
  auto output_size = at::native::padding::pad_shape_check(
      input, padding, /*dim=*/1, /*is_reflection=*/true);
  set_output_raw_strided(0, output_size, {}, input.options());
}

TORCH_META_FUNC(reflection_pad1d_backward)(const Tensor& grad_output,
    const Tensor& input,
    IntArrayRef padding) {
  at::native::padding::check_pad_within_input(input, padding, /*dim=*/1);
  at::native::padding::pad_backward_shape_check(grad_output, input, padding, /*dim=*/1);
  set_output_raw_strided(0, input.sizes(), {}, input.options());
}

TORCH_META_FUNC(reflection_pad3d)(const Tensor& input, IntArrayRef padding) {
  auto output_size = at::native::padding::pad_shape_check(
      input, padding, /*dim=*/3, /*is_reflection=*/true);
  set_output_raw_strided(0, output_size, {}, input.options());
}

TORCH_META_FUNC(reflection_pad3d_backward)(
    const Tensor& grad_output,
    const Tensor& input,
    IntArrayRef padding
) {
  TORCH_CHECK(
      padding.size() == 6,
      "padding size is expected to be 6, but got: ",
      padding.size());
  TORCH_CHECK(input.dim() > 3);
  TORCH_CHECK(grad_output.dim() == input.dim());

  at::native::padding::pad_backward_shape_check(grad_output, input, padding, /*dim=*/3);

  set_output_raw_strided(0, input.sizes(), {}, input.options());
}
} // namespace at::meta

namespace at::native {

namespace {

void reflection_pad2d_out_template(
    Tensor &output, const Tensor &input, IntArrayRef padding) {
  auto output_size = at::native::padding::pad_shape_check(
      input, padding, /*dim=*/2, /*is_reflection=*/true);

  /* resize output */
  if (input.dim() == 3 || input.is_quantized()) {
    // quantized tensor can not be resized with argument `memory_format`
    output.resize_(output_size);
  } else {
    output.resize_(output_size, input.suggest_memory_format());
  }
  reflection_pad2d_kernel(kCPU, output, input, padding);
}

void reflection_pad2d_backward_out_template(
    Tensor &grad_input, const Tensor &grad_output,
    const Tensor &input, IntArrayRef padding) {
  at::native::padding::pad_backward_shape_check(grad_output, input, padding, /*dim=*/2);
  reflection_pad2d_backward_kernel(kCPU, grad_input, grad_output, padding);
}

} // namespace

Tensor& reflection_pad1d_out_quantized_cpu(const Tensor& input, IntArrayRef padding,
    Tensor& output) {
  TORCH_CHECK(input.qscheme() == kPerTensorAffine, "Only per tensor quantization is supported");
  set_quantizer_(output, make_per_tensor_affine_quantizer(input.q_scale(), input.q_zero_point(), input.scalar_type()));
  reflection_pad1d_kernel(kCPU, output, input, padding);
  return output;
}

TORCH_IMPL_FUNC(reflection_pad1d_out_cpu)
(const Tensor& input, IntArrayRef padding, const Tensor& output) {
  reflection_pad1d_kernel(kCPU, output, input, padding);
}

TORCH_IMPL_FUNC(reflection_pad1d_backward_out_cpu)(const Tensor& grad_output,
    const Tensor& input,
    IntArrayRef padding,
    const Tensor& grad_input) {
  if (grad_output.numel() == 0) {
    return;
  }

  grad_input.zero_();
  reflection_pad1d_backward_kernel(kCPU, grad_input, grad_output, padding);
}

Tensor& reflection_pad2d_out_cpu(const Tensor& input, IntArrayRef padding,
    Tensor& output) {
  reflection_pad2d_out_template(output, input, padding);
  return output;
}

Tensor reflection_pad2d_cpu(const Tensor& input, IntArrayRef padding) {
  Tensor output = at::empty({0}, input.options());
  reflection_pad2d_out_template(output, input, padding);
  return output;
}

Tensor reflection_pad2d_quantized_cpu(const Tensor& input, IntArrayRef padding) {
  TORCH_CHECK(input.qscheme() == kPerTensorAffine, "Only per tensor quantization is supported");
  Tensor output = at::_empty_affine_quantized({0}, input.options(),
                                           input.q_scale(),
                                           input.q_zero_point());
  reflection_pad2d_out_template(output, input, padding);
  return output;
}

Tensor& reflection_pad2d_backward_out_cpu(const Tensor& grad_output,
    const Tensor& input,
    IntArrayRef padding,
    Tensor& grad_input) {
  grad_input.resize_as_(input, input.suggest_memory_format());
  grad_input.zero_();
  reflection_pad2d_backward_out_template(
    grad_input, grad_output, input, padding);
  return grad_input;
}

Tensor reflection_pad2d_backward_cpu(
    const Tensor& grad_output,
    const Tensor& input,
    IntArrayRef padding) {
  auto grad_input = at::zeros_like(input, input.suggest_memory_format());
  reflection_pad2d_backward_out_template(
    grad_input, grad_output, input, padding);
  return grad_input;
}

TORCH_IMPL_FUNC(reflection_pad3d_out_cpu)
(const Tensor& input, IntArrayRef padding, const Tensor& output) {
  // TODO: move this to TORCH_META_FUNC when CUDA has channels last support
  output.resize_(output.sizes(), input.suggest_memory_format());

  reflection_pad3d_kernel(kCPU, output, input, padding);
}

TORCH_IMPL_FUNC(reflection_pad3d_backward_out_cpu)(const Tensor& grad_output,
    const Tensor& input,
    IntArrayRef padding,
    const Tensor& grad_input) {
  if (grad_output.numel() == 0) {
    return;
  }

  // TODO: move this to TORCH_META_FUNC when CUDA has channels last support
  grad_input.resize_(input.sizes(), input.suggest_memory_format());

  grad_input.zero_();
  reflection_pad3d_backward_kernel(kCPU, grad_input, grad_output, padding);
}

DEFINE_DISPATCH(reflection_pad1d_kernel);
DEFINE_DISPATCH(reflection_pad1d_backward_kernel);
DEFINE_DISPATCH(reflection_pad2d_kernel);
DEFINE_DISPATCH(reflection_pad2d_backward_kernel);
DEFINE_DISPATCH(reflection_pad3d_kernel);
DEFINE_DISPATCH(reflection_pad3d_backward_kernel);

} // namespace at::native
