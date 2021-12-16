#pragma once
#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/TensorUtils.h>

#include <ATen/native/CPUBlas.h>
#include <ATen/native/ConvUtils.h>
#include <ATen/native/vol2col.h>

namespace at {
namespace native {
namespace {

static inline void slow_conv_transpose3d_shape_check(
    const Tensor& input,
    const Tensor& grad_output,
    const Tensor& weight,
    const Tensor& bias,
    int kernel_depth,
    int kernel_width,
    int kernel_height,
    int stride_depth,
    int stride_width,
    int stride_height,
    int padding_depth,
    int padding_width,
    int padding_height,
    int dilation_depth,
    int dilation_width,
    int dilation_height,
    int output_padding_depth,
    int output_padding_width,
    int output_padding_height,
    int weight_nullable) {
  TORCH_CHECK(
      input.numel() != 0 && (input.dim() == 4 || input.dim() == 5),
      "non-empty 4D or 5D (batch mode) tensor expected for input, but got: ",
      input.sizes());
  TORCH_CHECK(
      stride_depth > 0 && stride_width > 0 && stride_height > 0,
      "stride should be greater than zero, but got stride_depth: ",
      stride_depth,
      " stride_height: ",
      stride_height,
      " stride_width: ",
      stride_width);
  TORCH_CHECK(
      dilation_depth > 0 && dilation_width > 0 && dilation_height > 0,
      "dilation should be greater than zero, but got dilation_depth: ",
      dilation_depth,
      ", dilation_height: ",
      dilation_height,
      ", dilation_width: ",
      dilation_width);
  TORCH_CHECK(
      (output_padding_depth < stride_depth ||
       output_padding_depth < dilation_depth) &&
          (output_padding_width < stride_width ||
           output_padding_width < dilation_width) &&
          (output_padding_height < stride_height ||
           output_padding_height < dilation_height),
      "output padding must be smaller than either stride or dilation,",
      " but got output_padding_depth: ",
      output_padding_depth,
      " output_padding_height: ",
      output_padding_height,
      " output_padding_width: ",
      output_padding_width,
      " stride_depth: ",
      stride_depth,
      " stride_height: ",
      stride_height,
      " stride_width: ",
      stride_width,
      " dilation_depth: ",
      dilation_depth,
      " dilation_height: ",
      dilation_height,
      " dilation_width: ",
      dilation_width);

  // number of input & output planes and kernel size is indirectly defined by
  // the weight tensor
  if (weight.defined()) {
    /* TODO: TORCH_CHECK just have 2 args: condition and message */
    TORCH_CHECK(
        weight.numel() != 0 && weight.dim() == 5,
        "non-empty 5D (n_output_plane x n_input_plane x kernel_depth",
        " x kernel_height x kernel_width) tensor ",
        "expected for weight, but got: ",
        weight.sizes());
    if (bias.defined()) {
      check_dim_size(bias, 1, 0, weight.size(1));
    }
  } else if (!weight_nullable) {
    AT_ERROR("weight tensor is expected to be non-nullable");
  }

  int ndim = input.dim();
  int dimf = 0;
  int dimd = 1;
  int dimh = 2;
  int dimw = 3;

  if (ndim == 5) {
    dimf++;
    dimd++;
    dimh++;
    dimw++;
  }

  if (weight.defined()) {
    const int64_t n_input_plane = weight.size(0);
    check_dim_size(input, ndim, dimf, n_input_plane);
  }

  const int64_t input_width = input.size(dimw);
  const int64_t input_height = input.size(dimh);
  const int64_t input_depth = input.size(dimd);
  const int64_t output_depth = (input_depth - 1) * stride_depth -
      2 * padding_depth + (dilation_depth * (kernel_depth - 1) + 1) +
      output_padding_depth;
  const int64_t output_height = (input_height - 1) * stride_height -
      2 * padding_height + (dilation_height * (kernel_height - 1) + 1) +
      output_padding_height;
  const int64_t output_width = (input_width - 1) * stride_width -
      2 * padding_width + (dilation_width * (kernel_width - 1) + 1) +
      output_padding_width;

  if (output_depth < 1 || output_width < 1 || output_height < 1) {
    AT_ERROR(
        "Given input size per channel: (",
        input_depth,
        " x ",
        input_height,
        " x ",
        input_width,
        "). "
        "Calculated output size per channel: (",
        output_depth,
        " x ",
        output_height,
        " x ",
        output_width,
        "). Output size is too small");
  }

  if (grad_output.defined()) {
    if (weight.defined()) {
      const int64_t n_output_plane = weight.size(1);
      check_dim_size(grad_output, ndim, dimf, n_output_plane);
    } else if (bias.defined()) {
      const int64_t n_output_plane = bias.size(0);
      check_dim_size(grad_output, ndim, dimf, n_output_plane);
    }
    check_dim_size(grad_output, ndim, dimd, output_depth);
    check_dim_size(grad_output, ndim, dimh, output_height);
    check_dim_size(grad_output, ndim, dimw, output_width);
  }
}

} // namespace
}} // namespace at::native
