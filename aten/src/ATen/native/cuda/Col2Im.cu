#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/NativeFunctions.h>
#include <ATen/TensorUtils.h>
#include <ATen/Utils.h>
#include <ATen/div_rtn.h>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>

#include <ATen/native/cuda/im2col.cuh>
#include <ATen/native/im2col_shape_check.h>

namespace at {
namespace native {
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
  checkAllSameGPU("col2im_out_cuda", {input_arg, output_arg});

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
    input.resize_({1, input.size(0), input.size(1)});
  }

  int64_t batch_size = input.size(0);
  int64_t n_input_plane = input.size(1);
  int64_t n_output_plane = n_input_plane / (kernel_width * kernel_height);

  output.resize_({batch_size, n_output_plane, output_height, output_width});
  output.zero_();

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "col2im_out_cuda", [&] {
    using accscalar_t = at::acc_type<scalar_t, true>;

    Tensor input_n;
    Tensor output_n;

    int64_t height_col = (output_height + 2 * pad_height -
                          (dilation_height * (kernel_height - 1) + 1)) /
            stride_height +
        1;
    int64_t width_col = (output_width + 2 * pad_width -
                         (dilation_width * (kernel_width - 1) + 1)) /
            stride_width +
        1;

    for (int64_t elt = 0; elt < batch_size; elt++) {
      input_n = input.select(0, elt);
      output_n = output.select(0, elt);

      col2im<scalar_t, accscalar_t>(
          at::cuda::getCurrentCUDAStream(),
          input_n.data_ptr<scalar_t>(),
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
          output_n.data_ptr<scalar_t>());
    }

    if (!batched_input) {
      output.resize_({n_output_plane, output_height, output_width});
    }
  });
}

void col2im_backward_out_cuda_template(
    Tensor& grad_input,
    const Tensor& grad_output,
    IntArrayRef kernel_size,
    IntArrayRef dilation,
    IntArrayRef padding,
    IntArrayRef stride) {
  // im2col_out_cuda checks size of kernel_size, dilation, padding and stride
  im2col_out_cuda(
      grad_input, grad_output, kernel_size, dilation, padding, stride);
}

} // namespace

Tensor& col2im_out_cuda(
    Tensor& output,
    const Tensor& input,
    IntArrayRef output_size,
    IntArrayRef kernel_size,
    IntArrayRef dilation,
    IntArrayRef padding,
    IntArrayRef stride) {
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

Tensor& col2im_backward_out_cuda(
    Tensor& grad_input,
    const Tensor& grad_output,
    IntArrayRef kernel_size,
    IntArrayRef dilation,
    IntArrayRef padding,
    IntArrayRef stride) {
  col2im_backward_out_cuda_template(
      grad_input, grad_output, kernel_size, dilation, padding, stride);
  return grad_input;
}

Tensor col2im_backward_cuda(
    const Tensor& grad_output,
    IntArrayRef kernel_size,
    IntArrayRef dilation,
    IntArrayRef padding,
    IntArrayRef stride) {
  Tensor grad_input = at::empty_like(grad_output, LEGACY_CONTIGUOUS_MEMORY_FORMAT);

  col2im_backward_out_cuda_template(
      grad_input, grad_output, kernel_size, dilation, padding, stride);
  return grad_input;
}

} // namespace native
} // namespace at
