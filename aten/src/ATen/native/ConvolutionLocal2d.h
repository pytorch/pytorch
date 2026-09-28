#pragma once

#include <ATen/core/Tensor.h>
#include <c10/util/ArrayRef.h>

namespace at::native {

// Locally connected 2d convolution: a conv2d whose kernel is not shared across
// output positions. Layouts are
//   input   [N, C_in, H_in, W_in]
//   weight  [H_out, W_out, C_out, C_in, kH, kW]
//   bias    [C_out, H_out, W_out]
//   output  [N, C_out, H_out, W_out]
struct Conv2dLocalParams {
  int64_t batch;
  int64_t in_channels;
  int64_t in_height;
  int64_t in_width;
  int64_t out_channels;
  int64_t out_height;
  int64_t out_width;
  int64_t kernel_height;
  int64_t kernel_width;
  int64_t stride_height;
  int64_t stride_width;
  int64_t pad_height;
  int64_t pad_width;
  int64_t dilation_height;
  int64_t dilation_width;
};

// bias and grad_output may be undefined.
inline Conv2dLocalParams conv2d_local_shape_check(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    const Tensor& grad_output,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation) {
  TORCH_CHECK(stride.size() == 2, "conv2d_local: stride must have 2 elements, got ", stride);
  TORCH_CHECK(padding.size() == 2, "conv2d_local: padding must have 2 elements, got ", padding);
  TORCH_CHECK(dilation.size() == 2, "conv2d_local: dilation must have 2 elements, got ", dilation);
  TORCH_CHECK(stride[0] > 0 && stride[1] > 0, "conv2d_local: stride must be positive, got ", stride);
  TORCH_CHECK(padding[0] >= 0 && padding[1] >= 0, "conv2d_local: padding must be non-negative, got ", padding);
  TORCH_CHECK(dilation[0] > 0 && dilation[1] > 0, "conv2d_local: dilation must be positive, got ", dilation);
  TORCH_CHECK(input.dim() == 4, "conv2d_local: expected 4D input (N, C_in, H_in, W_in), got ", input.sizes());
  TORCH_CHECK(weight.dim() == 6, "conv2d_local: expected 6D weight (H_out, W_out, C_out, C_in, kH, kW), got ", weight.sizes());
  TORCH_CHECK(input.scalar_type() == weight.scalar_type(), "conv2d_local: input dtype ", input.scalar_type(), " does not match weight dtype ", weight.scalar_type());

  Conv2dLocalParams p{};
  p.batch = input.size(0);
  p.in_channels = input.size(1);
  p.in_height = input.size(2);
  p.in_width = input.size(3);
  p.out_channels = weight.size(2);
  p.kernel_height = weight.size(4);
  p.kernel_width = weight.size(5);
  p.stride_height = stride[0];
  p.stride_width = stride[1];
  p.pad_height = padding[0];
  p.pad_width = padding[1];
  p.dilation_height = dilation[0];
  p.dilation_width = dilation[1];

  TORCH_CHECK(weight.size(3) == p.in_channels, "conv2d_local: weight expects ", weight.size(3), " input channels but input has ", p.in_channels);
  TORCH_CHECK(p.kernel_height > 0 && p.kernel_width > 0, "conv2d_local: kernel size must be positive, got weight shape ", weight.sizes());

  const int64_t span_h = p.in_height + 2 * p.pad_height - p.dilation_height * (p.kernel_height - 1) - 1;
  const int64_t span_w = p.in_width + 2 * p.pad_width - p.dilation_width * (p.kernel_width - 1) - 1;
  TORCH_CHECK(span_h >= 0 && span_w >= 0, "conv2d_local: kernel size (", p.kernel_height, ", ", p.kernel_width, ") with dilation ", dilation, " is larger than the padded input ", input.sizes(), " with padding ", padding);
  p.out_height = span_h / p.stride_height + 1;
  p.out_width = span_w / p.stride_width + 1;
  TORCH_CHECK(weight.size(0) == p.out_height && weight.size(1) == p.out_width, "conv2d_local: weight is sized for output (", weight.size(0), ", ", weight.size(1), ") but input ", input.sizes(), " with stride ", stride, ", padding ", padding, ", dilation ", dilation, " produces output (", p.out_height, ", ", p.out_width, ")");

  if (bias.defined()) {
    TORCH_CHECK(bias.sizes() == IntArrayRef({p.out_channels, p.out_height, p.out_width}), "conv2d_local: expected bias of shape (", p.out_channels, ", ", p.out_height, ", ", p.out_width, "), got ", bias.sizes());
    TORCH_CHECK(bias.scalar_type() == input.scalar_type(), "conv2d_local: bias dtype ", bias.scalar_type(), " does not match input dtype ", input.scalar_type());
  }
  if (grad_output.defined()) {
    TORCH_CHECK(grad_output.sizes() == IntArrayRef({p.batch, p.out_channels, p.out_height, p.out_width}), "conv2d_local: expected grad_output of shape (", p.batch, ", ", p.out_channels, ", ", p.out_height, ", ", p.out_width, "), got ", grad_output.sizes());
    TORCH_CHECK(grad_output.scalar_type() == input.scalar_type(), "conv2d_local: grad_output dtype ", grad_output.scalar_type(), " does not match input dtype ", input.scalar_type());
  }
  return p;
}

} // namespace at::native
