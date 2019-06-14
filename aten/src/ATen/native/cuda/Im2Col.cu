#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/NativeFunctions.h>
#include <ATen/TensorUtils.h>
#include <ATen/Utils.h>
#include <ATen/div_rtn.h>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>

#include <ATen/native/cuda/im2col.cuh>

namespace at {
namespace native {
namespace {

static inline void im2col_shape_check(
    const Tensor& input,
    const Tensor& grad_output,
    int64_t kernel_height,
    int64_t kernel_width,
    int64_t dilation_height,
    int64_t dilation_width,
    int64_t pad_height,
    int64_t pad_width,
    int64_t stride_height,
    int64_t stride_width) {
  TORCH_CHECK(
      kernel_width > 0 && kernel_height > 0,
      "kernel size should be greater than zero, but got kernel_height: ",
      kernel_height,
      " kernel_width: ",
      kernel_width);

  TORCH_CHECK(
      dilation_width > 0 && dilation_height > 0,
      "dilation should be greater than zero, but got dilation_height: ",
      dilation_height,
      " dilation_width: ",
      dilation_width);
  
  TORCH_CHECK(
      pad_width >= 0 && pad_height >= 0,
      "padding should be non-negative, but got pad_height: ",
      pad_height,
      " pad_width: ",
      pad_width);
  
  TORCH_CHECK(
      stride_width > 0 && stride_height > 0,
      "stride should be greater than zero, but got stride_height: ",
      stride_height,
      " stride_width: ",
      stride_width);

  int64_t ndim = input.ndimension();

  TORCH_CHECK(
      input.numel() != 0 && (ndim == 3 || ndim == 4),
      "Expected non-empty 3D or 4D input tensor, but got input of size ",
      input.sizes());

  int64_t dim_batch = 0;

  if (ndim == 3) {
    dim_batch = -1;
  }

  int64_t n_input_plane = input.size(dim_batch + 1);
  int64_t input_height = input.size(dim_batch + 2);
  int64_t input_width = input.size(dim_batch + 3);
  int64_t output_height = div_rtn<int64_t>(
                              input_height + 2 * pad_height -
                                  (dilation_height * (kernel_height - 1) + 1),
                              stride_height) +
      1;
  int64_t output_width = div_rtn<int64_t>(
                             input_width + 2 * pad_width -
                                 (dilation_width * (kernel_width - 1) + 1),
                             stride_width) +
      1;

  if (output_height < 1 || output_width < 1) {
    AT_ERROR(
        "Given input with spatial size (",
        input_height,
        ", ",
        input_width,
        "), kernel_size=(",
        kernel_height,
        ", ",
        kernel_width,
        "), dilation=(",
        dilation_height,
        ", ",
        dilation_width,
        "), padding=(",
        pad_height,
        ", ",
        pad_width,
        "), calculated shape of the array of sliding blocks as (",
        output_height,
        ", ",
        output_width,
        "), which is too small (non-positive).");
  }
}

static void im2col_out_cuda_template(
    Tensor& output,
    Tensor& input_,
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
  checkAllSameGPU("im2col_cuda", {input_arg, output_arg});

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
    input.resize_({1, input.size(0), input.size(1), input.size(2)});
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
  output.zero_();

  // Launch kernel
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "im2col_out_cuda", [&] {
    Tensor input_n;
    Tensor output_n;

    for (int64_t elt = 0; elt < batch_size; elt++) {
      input_n = input.select(0, elt);
      output_n = output.select(0, elt);

      im2col<scalar_t>(
          at::cuda::getCurrentCUDAStream(),
          input_n.data<scalar_t>(),
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
          output_n.data<scalar_t>());
    }

    if (!batched_input) {
      output.resize_({n_output_plane, output_length});
    }
  });
}

static void im2col_backward_out_cuda_template(
    Tensor& grad_input,
    Tensor& grad_output,
    IntArrayRef input_size,
    IntArrayRef kernel_size,
    IntArrayRef dilation,
    IntArrayRef padding,
    IntArrayRef stride) {
  TORCH_CHECK(
      input_size.size() == 2,
      "It is expected input_size equals to 2, but got size ",
      input_size.size());
  // col2im_out_cuda checks size of kernel_size, dilation, padding and stride
  grad_input = col2im_cuda(
      grad_output,
      input_size,
      kernel_size,
      dilation,
      padding,
      stride);
}

} // namespace

Tensor im2col_cuda(
    const Tensor& input_,
    IntArrayRef kernel_size,
    IntArrayRef dilation,
    IntArrayRef padding,
    IntArrayRef stride) {
  Tensor input = input_;
  Tensor output = at::empty_like(input);
  im2col_out_cuda_template(
      output, input, kernel_size, dilation, padding, stride);
  return output;
}

Tensor im2col_backward_cuda(
    const Tensor& grad_output_,
    IntArrayRef input_size,
    IntArrayRef kernel_size,
    IntArrayRef dilation,
    IntArrayRef padding,
    IntArrayRef stride) {
  Tensor grad_output = grad_output_;
  Tensor grad_input = at::empty_like(grad_output);
  im2col_backward_out_cuda_template(
      grad_input,
      grad_output,
      input_size,
      kernel_size,
      dilation,
      padding,
      stride);
  return grad_input;
}

} // namespace native
} // namespace at
