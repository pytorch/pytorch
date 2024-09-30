#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/TensorMeta.h>
#include <ATen/quantized/Quantizer.h>
#include <ATen/native/Padding.h>
#include <c10/util/irange.h>

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
  int64_t dim_plane = 0;
  int64_t dim_w = 1;
  int64_t nbatch = 1;

  if (input.ndimension() == 3) {
    nbatch = input.size(0);
    dim_w++;
    dim_plane++;
  }

  at::native::padding::check_valid_input<1>(input, padding);

  /* sizes */
  auto pad_l = padding[0];
  auto pad_r = padding[1];

  int64_t nplane = input.size(dim_plane);
  int64_t input_w = input.size(dim_w);
  int64_t output_w = input_w + pad_l + pad_r;

  TORCH_CHECK(
      pad_l < input_w && pad_r < input_w,
      "Argument #4: Padding size "
      "should be less than the corresponding input dimension, but got: padding (",
      pad_l,
      ", ",
      pad_r,
      ") at dimension ",
      dim_w,
      " of input ",
      input.sizes());

  TORCH_CHECK(
      output_w >= 1,
      "input (W: ",
      input_w,
      ") is too small. Calculated output W: ",
      output_w);

  if (input.ndimension() == 2) {
    set_output_raw_strided(0, {nplane, output_w}, {}, input.options());
  } else {
    set_output_raw_strided(0, {nbatch, nplane, output_w}, {}, input.options());
  }
}

TORCH_META_FUNC(reflection_pad1d_backward)(const Tensor& grad_output,
    const Tensor& input,
    IntArrayRef padding) {
  int64_t dim_w = 1;
  if (input.ndimension() == 3) {
    dim_w++;
  }

  /* sizes */
  auto pad_l = padding[0];
  auto pad_r = padding[1];
  int64_t input_w = input.size(dim_w);
  int64_t output_w  = input_w + pad_l + pad_r;

  TORCH_CHECK(
      pad_l < input_w && pad_r < input_w,
      "Argument #4: Padding size "
      "should be less than the corresponding input dimension, but got: padding (",
      pad_l,
      ", ",
      pad_r,
      ") at dimension ",
      dim_w,
      " of input ",
      input.sizes());

  TORCH_CHECK(output_w == grad_output.size(dim_w), "grad_output width unexpected."
    " Expected: ", output_w, ", Got: ", grad_output.size(dim_w));

  set_output_raw_strided(0, input.sizes(), {}, input.options());
}

TORCH_META_FUNC(reflection_pad3d)(const Tensor& input, IntArrayRef padding) {
  int64_t pad_left = padding[0];
  int64_t pad_right = padding[1];
  int64_t pad_top = padding[2];
  int64_t pad_bottom = padding[3];
  int64_t pad_front = padding[4];
  int64_t pad_back = padding[5];
  int64_t dim_w = 3;
  int64_t dim_h = 2;
  int64_t dim_d = 1;
  int64_t dim_plane = 0;

  at::native::padding::check_valid_input<3>(input, padding);

  bool batch_mode = (input.dim() == 5);
  if (batch_mode) {
    dim_w++;
    dim_h++;
    dim_d++;
    dim_plane++;
  }

  int64_t nplane = input.size(dim_plane);
  int64_t input_d = input.size(dim_d);
  int64_t input_h = input.size(dim_h);
  int64_t input_w = input.size(dim_w);
  int64_t output_d = input_d + pad_front + pad_back;
  int64_t output_h = input_h + pad_top + pad_bottom;
  int64_t output_w = input_w + pad_left + pad_right;

  TORCH_CHECK(
      pad_left < input_w && pad_right < input_w,
      "Argument #4: Padding size "
      "should be less than the corresponding input dimension, but got: padding (",
      pad_left, ", ", pad_right, ") at dimension ", dim_w, " of input ", input.sizes());
  TORCH_CHECK(
      pad_top < input_h && pad_bottom < input_h,
      "Argument #6: Padding size "
      "should be less than the corresponding input dimension, but got: padding (",
      pad_top, ", ", pad_bottom, ") at dimension ", dim_h, " of input ", input.sizes());
  TORCH_CHECK(
      pad_front < input_d && pad_back < input_d,
      "Argument #8: Padding size "
      "should be less than the corresponding input dimension, but got: padding (",
      pad_front, ", ", pad_back, ") at dimension ", dim_d, " of input ", input.sizes());

  TORCH_CHECK(output_w >= 1 || output_h >=1 || output_d >= 1,
      "input (D: ", input_d, " H: ", input_h, ", W: ", input_w,
      ") is too small."
      " Calculated output D: ", output_d, " H: ", output_h, " W: ", output_w);

  if (batch_mode) {
    set_output_raw_strided(0, {input.size(0), nplane, output_d, output_h, output_w}, {}, input.options());
  } else {
    set_output_raw_strided(0, {nplane, output_d, output_h, output_w}, {}, input.options());
  }
}

TORCH_META_FUNC(reflection_pad3d_backward)(
    const Tensor& grad_output,
    const Tensor& input,
    IntArrayRef padding
) {
  TORCH_CHECK(padding.size() == 6, "padding size is expected to be 6");
  TORCH_CHECK(input.dim() > 3);
  TORCH_CHECK(grad_output.dim() == input.dim());

  int64_t pad_left = padding[0];
  int64_t pad_right = padding[1];
  int64_t pad_top = padding[2];
  int64_t pad_bottom = padding[3];
  int64_t pad_front = padding[4];
  int64_t pad_back = padding[5];
  int64_t dim_w = 3;
  int64_t dim_h = 2;
  int64_t dim_d = 1;

  if (input.dim() == 5) {
    // batch mode
    dim_w++;
    dim_h++;
    dim_d++;
  }

  int64_t input_d = input.size(dim_d);
  int64_t input_h = input.size(dim_h);
  int64_t input_w = input.size(dim_w);
  int64_t output_d = input_d + pad_front + pad_back;
  int64_t output_h = input_h + pad_top + pad_bottom;
  int64_t output_w = input_w + pad_left + pad_right;

  TORCH_CHECK(output_w == grad_output.size(dim_w), "grad_output width unexpected."
    " Expected: ", output_w, ", Got: ", grad_output.size(dim_w));
  TORCH_CHECK(output_h == grad_output.size(dim_h), "grad_output height unexpected."
    " Expected: ", output_h, ", Got: ", grad_output.size(dim_h));
  TORCH_CHECK(output_d == grad_output.size(dim_d), "grad_output depth unexpected."
    " Expected: ", output_d, ", Got: ", grad_output.size(dim_d));

  set_output_raw_strided(0, input.sizes(), {}, input.options());
}
} // namespace at::meta

namespace at::native {

namespace {

void reflection_pad2d_out_template(
    Tensor &output, const Tensor &input, IntArrayRef padding) {
  int dim_w = 2;
  int dim_h = 1;
  int dim_slices = 0;
  int64_t nbatch = 1;

  at::native::padding::check_valid_input<2>(input, padding);

  int ndim = input.dim();
  if (ndim == 4) {
    nbatch = input.size(0);
    dim_w++;
    dim_h++;
    dim_slices++;
  }

  /* sizes */
  int64_t pad_l = padding[0];
  int64_t pad_r = padding[1];
  int64_t pad_t = padding[2];
  int64_t pad_b = padding[3];

  int64_t nplane = input.size(dim_slices);
  int64_t input_h = input.size(dim_h);
  int64_t input_w = input.size(dim_w);
  int64_t output_h = input_h + pad_t + pad_b;
  int64_t output_w  = input_w + pad_l + pad_r;

  TORCH_CHECK(pad_l < input_w && pad_r < input_w,
    "Argument #4: Padding size should be less than the corresponding "
    "input dimension, but got: padding (", pad_l, ", ", pad_r,
    ") at dimension ", dim_w, " of input ", input.sizes());

  TORCH_CHECK(pad_t < input_h && pad_b < input_h,
    "Argument #6: Padding size should be less than the corresponding "
    "input dimension, but got: padding (", pad_t, ", ", pad_b,
    ") at dimension ", dim_h, " of input ", input.sizes());

  TORCH_CHECK(output_w >= 1 || output_h >= 1,
    "input (H: ", input_h, ", W: ", input_w, ") is too small. Calculated "
    "output H: ", output_h, " W: ", output_w);

  /* resize output */
  if (ndim == 3) {
    output.resize_({nplane, output_h, output_w});
  } else {
    if (input.is_quantized()) {
      // quantized tensor can not be resized with argument `memory_format`
      output.resize_({nbatch, nplane, output_h, output_w});
    } else {
      output.resize_({nbatch, nplane, output_h, output_w}, input.suggest_memory_format());
    }
  }
  reflection_pad2d_kernel(kCPU, output, input, padding);
}

void reflection_pad2d_backward_out_template(
    Tensor &grad_input, const Tensor &grad_output,
    const Tensor &input, IntArrayRef padding) {
  int dim_w = 2;
  int dim_h = 1;

  if (input.ndimension() == 4) {
    dim_w++;
    dim_h++;
  }

  /* sizes */
  int64_t pad_l = padding[0];
  int64_t pad_r = padding[1];
  int64_t pad_t = padding[2];
  int64_t pad_b = padding[3];

  int64_t input_h = input.size(dim_h);
  int64_t input_w = input.size(dim_w);
  int64_t output_h = input_h + pad_t + pad_b;
  int64_t output_w  = input_w + pad_l + pad_r;

  TORCH_CHECK(output_w == grad_output.size(dim_w),
    "gradOutput width unexpected. Expected: ", output_w, ", Got: ",
    grad_output.size(dim_w));

  TORCH_CHECK(output_h == grad_output.size(dim_h),
    "gradOutput height unexpected. Expected: ", output_h, ", Got: ",
    grad_output.size(dim_h));

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
