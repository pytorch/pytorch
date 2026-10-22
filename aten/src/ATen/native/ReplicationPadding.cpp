#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/TensorMeta.h>
#include <ATen/native/Padding.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/replication_pad1d_backward_native.h>
#include <ATen/ops/replication_pad1d_native.h>
#include <ATen/ops/replication_pad2d_backward_native.h>
#include <ATen/ops/replication_pad2d_native.h>
#include <ATen/ops/replication_pad3d_backward_native.h>
#include <ATen/ops/replication_pad3d_native.h>
#include <ATen/ops/zeros_like.h>
#endif

namespace at::meta {

TORCH_META_FUNC(replication_pad1d) (
  const Tensor& input, IntArrayRef paddingSize
) {
  auto output_size = at::native::padding::pad_shape_check(
      input, paddingSize, /*dim=*/1, /*is_reflection=*/false);
  set_output_raw_strided(0, output_size, {}, input.options());
}

TORCH_META_FUNC(replication_pad1d_backward) (
  const Tensor& gradOutput,
  const Tensor& input,
  IntArrayRef paddingSize
) {
  TORCH_CHECK(paddingSize.size() == 2, "padding size is expected to be 2, but got: ", paddingSize.size());
  at::native::padding::pad_backward_shape_check(gradOutput, input, paddingSize, /*dim=*/1);
  set_output_raw_strided(0, input.sizes(), {}, input.options());
}

TORCH_META_FUNC(replication_pad2d) (
  const Tensor& input, IntArrayRef paddingSize
) {
  auto output_size = at::native::padding::pad_shape_check(
      input, paddingSize, /*dim=*/2, /*is_reflection=*/false);
  set_output_raw_strided(0, output_size, {}, input.options());
}

TORCH_META_FUNC(replication_pad3d) (
  const Tensor& input, IntArrayRef paddingSize
) {
  auto output_size = at::native::padding::pad_shape_check(
      input, paddingSize, /*dim=*/3, /*is_reflection=*/false);
  set_output_raw_strided(0, output_size, {}, input.options());
}

} // namespace at::meta

namespace at::native {

namespace {

void replication_pad2d_backward_out_cpu_template(
    Tensor& gradInput,
    const Tensor& gradOutput,
    const Tensor& input,
    IntArrayRef paddingSize)
{
  at::native::padding::check_valid_input(input, paddingSize, /*dim=*/2);
  at::native::padding::pad_backward_shape_check(gradOutput, input, paddingSize, /*dim=*/2);

  if (gradInput.numel() == 0) {
    return;
  }

  replication_pad2d_backward_kernel(kCPU, gradInput, gradOutput, paddingSize);
}

void replication_pad3d_backward_out_cpu_template(
    Tensor& gradInput,
    const Tensor& gradOutput,
    const Tensor& input,
    IntArrayRef paddingSize)
{
  at::native::padding::check_valid_input(input, paddingSize, /*dim=*/3);
  at::native::padding::pad_backward_shape_check(gradOutput, input, paddingSize, /*dim=*/3);

  if (gradInput.numel() == 0) {
    return;
  }

  replication_pad3d_backward_kernel(kCPU, gradInput, gradOutput, paddingSize);
}

} // anonymous namespace

TORCH_IMPL_FUNC(replication_pad1d_out_cpu) (
  const Tensor& input, IntArrayRef paddingSize, const Tensor& output
) {
  replication_pad1d_kernel(kCPU, output, input, paddingSize);
}

TORCH_IMPL_FUNC(replication_pad1d_backward_out_cpu) (
  const Tensor& gradOutput, const Tensor& input, IntArrayRef paddingSize, const Tensor& gradInput
) {
  if (gradInput.numel() == 0) {
    return;
  }
  gradInput.zero_();

  replication_pad1d_backward_kernel(kCPU, gradInput, gradOutput, paddingSize);
}

TORCH_IMPL_FUNC(replication_pad2d_out_cpu) (
  const Tensor& input, IntArrayRef paddingSize, const Tensor& output
) {
  // TODO: move this to TORCH_META_FUNC when CUDA has channels last support
  output.resize_(output.sizes(), input.suggest_memory_format());

  replication_pad2d_kernel(kCPU, output, input, paddingSize);
}

Tensor& replication_pad2d_backward_out_cpu(const Tensor& gradOutput,
    const Tensor& input,
    IntArrayRef paddingSize,
    Tensor& gradInput)
{
  gradInput.resize_as_(input, input.suggest_memory_format());
  gradInput.zero_();
  replication_pad2d_backward_out_cpu_template(
      gradInput, gradOutput, input, paddingSize);
  return gradInput;
}

Tensor replication_pad2d_backward_cpu(
    const Tensor& gradOutput,
    const Tensor& input,
    IntArrayRef paddingSize)
{
  auto gradInput = at::zeros_like(input, input.suggest_memory_format());
  replication_pad2d_backward_out_cpu_template(
      gradInput, gradOutput, input, paddingSize);
  return gradInput;
}

TORCH_IMPL_FUNC(replication_pad3d_out_cpu) (
  const Tensor& input, IntArrayRef paddingSize, const Tensor& output
) {
  // TODO: move this to TORCH_META_FUNC when CUDA has channels last support
  output.resize_(output.sizes(), input.suggest_memory_format());

  replication_pad3d_kernel(kCPU, output, input, paddingSize);
}

Tensor& replication_pad3d_backward_out_cpu(const Tensor& gradOutput,
    const Tensor& input,
    IntArrayRef paddingSize,
    Tensor& gradInput)
{
  gradInput.resize_as_(input, input.suggest_memory_format());
  gradInput.zero_();
  replication_pad3d_backward_out_cpu_template(
      gradInput, gradOutput, input, paddingSize);
  return gradInput;
}

Tensor replication_pad3d_backward_cpu(
    const Tensor& gradOutput,
    const Tensor& input,
    IntArrayRef paddingSize)
{
  auto gradInput = at::zeros_like(input, input.suggest_memory_format());
  replication_pad3d_backward_out_cpu_template(
      gradInput, gradOutput, input, paddingSize);
  return gradInput;
}

DEFINE_DISPATCH(replication_pad1d_kernel);
DEFINE_DISPATCH(replication_pad1d_backward_kernel);
DEFINE_DISPATCH(replication_pad2d_kernel);
DEFINE_DISPATCH(replication_pad2d_backward_kernel);
DEFINE_DISPATCH(replication_pad3d_kernel);
DEFINE_DISPATCH(replication_pad3d_backward_kernel);

} // namespace at::native
