#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/TensorUtils.h>
#include <ATen/Utils.h>
#include <ATen/div_rtn.h>

#include <ATen/cuda/CUDAContext.h>

#include <ATen/native/cuda/im2col.cuh>
#include <ATen/native/im2col_shape_check.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/col2im_native.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/im2col_native.h>
#endif

namespace at::native {
namespace {

void col2im_out_cuda_template(
    Tensor& output,
    const Tensor& input_,
    IntArrayRef output_size,
    IntArrayRef kernel_size,
    IntArrayRef dilation,
    IntArrayRef padding,
    IntArrayRef stride) {
  TensorArg input_arg{input_, "input", 1};
  TensorArg output_arg{output, "output", 2};
  checkAllSameGPU(__func__, {input_arg, output_arg});

  TORCH_CHECK(
      output_size.size() == 2,
      "It is expected output_size equals to 2, but got size ",
      output_size.size());

  TORCH_CHECK(
      kernel_size.size() == 2,
      "It is expected kernel_size equals to 2, but got size ",
      kernel_size.size());

  TORCH_CHECK(
      dilation.size() == 2,
      "It is expected dilation equals to 2, but got size ",
      dilation.size());

  TORCH_CHECK(
      padding.size() == 2,
      "It is expected padding equals to 2, but got size ",
      padding.size());

  TORCH_CHECK(
      stride.size() == 2,
      "It is expected stride equals to 2, but got size ",
      stride.size());

  int64_t output_height = output_size[0];
  int64_t output_width = output_size[1];
  int64_t kernel_height = kernel_size[0];
  int64_t kernel_width = kernel_size[1];
  int64_t dilation_height = dilation[0];
  int64_t dilation_width = dilation[1];
  int64_t pad_height = padding[0];
  int64_t pad_width = padding[1];
  int64_t stride_height = stride[0];
  int64_t stride_width = stride[1];

  col2im_shape_check(
      input_,
      Tensor(),
      output_height,
      output_width,
      kernel_height,
      kernel_width,
      dilation_height,
      dilation_width,
      pad_height,
      pad_width,
      stride_height,
      stride_width);

  Tensor input = input_.contiguous();

  bool batched_input = true;
  if (input.dim() == 2) {
    // Force batch
    batched_input = false;
    input = input.unsqueeze(0);
  }

  int64_t batch_size = input.size(0);
  int64_t n_input_plane = input.size(1);
  int64_t n_output_plane = n_input_plane / (kernel_width * kernel_height);
  int64_t input_batch_stride = input.stride(0);

  output.resize_({batch_size, n_output_plane, output_height, output_width});
  int64_t output_batch_stride = output.stride(0);

  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND3(kHalf, kBFloat16, kBool,
      input.scalar_type(), "col2im_out_cuda", [&] {
    int64_t height_col = (output_height + 2 * pad_height -
                          (dilation_height * (kernel_height - 1) + 1)) /
            stride_height +
        1;
    int64_t width_col = (output_width + 2 * pad_width -
                         (dilation_width * (kernel_width - 1) + 1)) /
            stride_width +
        1;

    col2im_batched(
        at::cuda::getCurrentCUDAStream(),
        input.const_data_ptr<scalar_t>(),
        input_batch_stride,
        batch_size,
        n_output_plane,
        output_height,
        output_width,
        height_col,
        width_col,
        kernel_height,
        kernel_width,
        pad_height,
        pad_width,
        stride_height,
        stride_width,
        dilation_height,
        dilation_width,
        output.mutable_data_ptr<scalar_t>(),
        output_batch_stride);

  });
  if (!batched_input) {
    output = output.squeeze(0);
  }
}

} // namespace

Tensor& col2im_out_cuda(const Tensor& input,
    IntArrayRef output_size,
    IntArrayRef kernel_size,
    IntArrayRef dilation,
    IntArrayRef padding,
    IntArrayRef stride,
    Tensor& output) {
  col2im_out_cuda_template(
      output, input, output_size, kernel_size, dilation, padding, stride);
  return output;
}

Tensor col2im_cuda(
    const Tensor& input,
    IntArrayRef output_size,
    IntArrayRef kernel_size,
    IntArrayRef dilation,
    IntArrayRef padding,
    IntArrayRef stride) {
  Tensor output = at::empty_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);

  col2im_out_cuda_template(
      output, input, output_size, kernel_size, dilation, padding, stride);
  return output;
}

} // namespace at::native
