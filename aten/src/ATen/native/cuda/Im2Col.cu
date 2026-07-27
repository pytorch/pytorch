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
#include <ATen/ops/empty_like.h>
#include <ATen/ops/col2im_native.h>
#include <ATen/ops/im2col_native.h>
#endif

namespace at::native {
namespace {

static void im2col_out_cuda_template(
    Tensor& output,
    const Tensor& input_,
    IntArrayRef kernel_size,
    IntArrayRef dilation,
    IntArrayRef padding,
    IntArrayRef stride) {
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

  int64_t kernel_height = kernel_size[0];
  int64_t kernel_width = kernel_size[1];
  int64_t dilation_height = dilation[0];
  int64_t dilation_width = dilation[1];
  int64_t pad_height = padding[0];
  int64_t pad_width = padding[1];
  int64_t stride_height = stride[0];
  int64_t stride_width = stride[1];

  TensorArg input_arg{input_, "input", 1};
  TensorArg output_arg{output, "output", 2};
  checkAllSameGPU(__func__, {input_arg, output_arg});

  im2col_shape_check(
      input_,
      Tensor(),
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

  if (input.dim() == 3) {
    batched_input = false;
    input = input.unsqueeze(0);
  }

  int64_t batch_size = input.size(0);
  int64_t n_input_plane = input.size(1);
  int64_t input_height = input.size(2);
  int64_t input_width = input.size(3);

  int64_t output_height = (input_height + 2 * pad_height -
                           (dilation_height * (kernel_height - 1) + 1)) /
          stride_height +
      1;
  int64_t output_width = (input_width + 2 * pad_width -
                          (dilation_width * (kernel_width - 1) + 1)) /
          stride_width +
      1;
  int64_t n_output_plane = n_input_plane * kernel_width * kernel_height;
  int64_t output_length = output_height * output_width;

  output.resize_({batch_size, n_output_plane, output_length});

  // Launch kernel
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND3(kHalf, kBFloat16, kBool,
      input.scalar_type(), "im2col_out_cuda", [&] {
    Tensor input_n;
    Tensor output_n;

    for (int64_t elt = 0; elt < batch_size; elt++) {
      input_n = input.select(0, elt);
      output_n = output.select(0, elt);

      im2col<scalar_t>(
          at::cuda::getCurrentCUDAStream(),
          input_n.const_data_ptr<scalar_t>(),
          n_input_plane,
          input_height,
          input_width,
          output_height,
          output_width,
          kernel_height,
          kernel_width,
          pad_height,
          pad_width,
          stride_height,
          stride_width,
          dilation_height,
          dilation_width,
          output_n.mutable_data_ptr<scalar_t>());
    }

  });
  if (!batched_input) {
    output = output.squeeze(0);
  }
}

} // namespace

Tensor& im2col_out_cuda(const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef dilation,
    IntArrayRef padding,
    IntArrayRef stride,
    Tensor& output) {
  im2col_out_cuda_template(
      output, input, kernel_size, dilation, padding, stride);
  return output;
}

Tensor im2col_cuda(
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef dilation,
    IntArrayRef padding,
    IntArrayRef stride) {
  Tensor output = at::empty_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  im2col_out_cuda_template(
      output, input, kernel_size, dilation, padding, stride);
  return output;
}

} // namespace at::native
