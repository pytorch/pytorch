#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/Padding.h>

namespace at {

namespace meta {

TORCH_META_FUNC(reflection_pad1d)(const Tensor& input, IntArrayRef padding) {
  int64_t dim_plane = 0;
  int64_t dim_w = 1;
  int64_t nbatch = 1;

  // allow dim=0 only in the batch dimension.
  TORCH_CHECK(
      (input.ndimension() == 2 && input.size(1) != 0) ||
          (input.ndimension() == 3 && input.size(1) != 0 && input.size(2) != 0),
      "2D or 3D (batch mode) tensor expected for input, but got: ",
      input);

  if (input.ndimension() == 3) {
    nbatch = input.size(0);
    dim_w++;
    dim_plane++;
  }

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
      2,
      "input (W: ",
      input_w,
      ")is too small. Calculated output W: ",
      output_w);

  if (input.ndimension() == 2) {
    set_output({nplane, output_w}, input.options());
  } else {
    set_output({nbatch, nplane, output_w}, input.options());
  }
}

TORCH_META_FUNC(reflection_pad1d_backward) (
    const Tensor& grad_output,
    const Tensor& input,
    IntArrayRef padding) {
  int64_t dim_plane = 0;
  int64_t dim_w = 1;

  if (input.ndimension() == 3) {
    dim_w++;
    dim_plane++;
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

  set_output(input.sizes(), input.options());
}

TORCH_META_FUNC(reflection_pad3d)(const Tensor& input, IntArrayRef padding) {
  TORCH_CHECK(padding.size() == 6, "padding size is expected to be 6");
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

  // allow batch size of 0-dim.
  bool valid_dims =
      input.size(1) != 0 && input.size(2) != 0 && input.size(3) != 0;
  bool valid_single = input.dim() == 4 && input.size(0) != 0 && valid_dims;
  bool valid_batch = input.dim() == 5 && valid_dims && input.size(4) != 0;

  TORCH_CHECK(
    valid_single || valid_batch,
      "Expected 4D or 5D (batch mode) tensor with possibly 0 batch size and other non-zero dimensions for input, but got: ",
  input.sizes());

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

  auto memory_format = input.suggest_memory_format();
  if (batch_mode) {
    set_output({input.size(0), nplane, output_d, output_h, output_w}, input.options().memory_format(memory_format));
  } else {
    set_output({nplane, output_d, output_h, output_w}, input.options());
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

  if (input.dim() == 5)
  {
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
    " Expected: ", output_h, ", Got: ", grad_output.size(dim_d));

  auto memory_format = input.suggest_memory_format();
  set_output(input.sizes(), input.options().memory_format(memory_format));
}
} // namespace meta

namespace native {

namespace {

void reflection_pad1d_out_template(
    const Tensor& output, const Tensor& input, IntArrayRef padding) {
  int64_t dim_plane = 0;
  int64_t dim_w = 1;
  int64_t nbatch = 1;
  // allow dim=0 only in the batch dimension.
  TORCH_CHECK(
      (input.ndimension() == 2 && input.size(1) != 0) ||
      (input.ndimension() == 3 && input.size(1) != 0 && input.size(2) != 0),
      "2D or 3D (batch mode) tensor expected for input, but got: ", input.sizes());

  if (input.ndimension() == 3) {
    nbatch = input.size(0);
    dim_w++;
    dim_plane++;
  }

  /* sizes */
  auto pad_l = padding[0];
  auto pad_r = padding[1];

  int64_t nplane = input.size(dim_plane);
  int64_t input_w = input.size(dim_w);
  int64_t output_w  = input_w + pad_l + pad_r;

  TORCH_CHECK(pad_l < input_w && pad_r < input_w, "Argument #4: Padding size "
    "should be less than the corresponding input dimension, but got: padding (",
    pad_l, ", ", pad_r, ") at dimension ", dim_w, " of input ", input.sizes());

  TORCH_CHECK(output_w >= 1 , 2,
    "input (W: ", input_w, ")is too small. Calculated output W: ", output_w);

  /* resize output */
  if (input.ndimension() == 2) {
    output.resize_({nplane, output_w});
  } else {
    output.resize_({nbatch, nplane, output_w});
  }

  reflection_pad1d_kernel(kCPU, output, input, padding);
}

void reflection_pad2d_out_template(
    Tensor &output, const Tensor &input, IntArrayRef padding) {
  int dim_w = 2;
  int dim_h = 1;
  int dim_slices = 0;
  int64_t nbatch = 1;

  bool valid_dims = input.size(1) != 0 && input.size(2) != 0;
  TORCH_CHECK(
      (input.ndimension() == 3 && valid_dims) ||
      (input.ndimension() == 4 && valid_dims && input.size(3) != 0),
      "3D or 4D (batch mode) tensor expected for input, but got: ", input.sizes());

  if (input.ndimension() == 4) {
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
    ") at dimension ", dim_w, " of input ", input.ndimension());

  TORCH_CHECK(pad_t < input_h && pad_b < input_h,
    "Argument #6: Padding size should be less than the corresponding "
    "input dimension, but got: padding (", pad_t, ", ", pad_b,
    ") at dimension ", dim_h, " of input ", input.ndimension());

  TORCH_CHECK(output_w >= 1 || output_h >= 1,
    "input (H: ", input_h, ", W: ", input_w, ")is too small. Calculated "
    "output H: ", output_h, " W: ", output_w);

  /* resize output */
  if (input.ndimension() == 3) {
    output.resize_({nplane, output_h, output_w});
  } else {
    output.resize_({nbatch, nplane, output_h, output_w}, input.suggest_memory_format());
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

  /* resize grad_input */
  grad_input.resize_(input.sizes(), input.suggest_memory_format());

  grad_input.zero_();
  reflection_pad2d_backward_kernel(kCPU, grad_input, grad_output, padding);
}

} // namespace

Tensor& reflection_pad1d_out_cpu(
    const Tensor& input, IntArrayRef padding, Tensor& output) {
  reflection_pad1d_out_template(output, input, padding);
  return output;
}

Tensor reflection_pad1d_cpu(const Tensor& input, IntArrayRef padding) {
  Tensor output;
  if (input.is_quantized()) {
    if (input.qscheme() == kPerTensorAffine) {
      output = at::_empty_affine_quantized({0}, input.options(),
                                           input.q_scale(),
                                           input.q_zero_point());
    } else {
      TORCH_CHECK(false, "Only per tensor quantization is supported");
    }
  } else {
    output = at::empty({0}, input.options());
  }
  reflection_pad1d_out_template(output, input, padding);
  return output;
}

TORCH_IMPL_FUNC(reflection_pad1d_out_cpu) (
    const Tensor& input, IntArrayRef padding, const Tensor& output) {
  reflection_pad1d_out_template(output, input, padding);
}

TORCH_IMPL_FUNC(reflection_pad1d_backward_out_cpu) (
    const Tensor& grad_output, const Tensor& input, IntArrayRef padding, const Tensor& grad_input) {
  grad_input.zero_();
  reflection_pad1d_backward_kernel(kCPU, grad_input, grad_output, padding);
}

Tensor& reflection_pad2d_out_cpu(
    const Tensor& input, IntArrayRef padding, Tensor& output) {
  reflection_pad2d_out_template(output, input, padding);
  return output;
}

Tensor reflection_pad2d_cpu(const Tensor& input, IntArrayRef padding) {
  Tensor output;
  if (input.is_quantized()) {
    if (input.qscheme() == kPerTensorAffine) {
      output = at::_empty_affine_quantized({0}, input.options(),
                                           input.q_scale(),
                                           input.q_zero_point());
    } else {
      TORCH_CHECK(false, "Only per tensor quantization is supported");
    }
  } else {
    output = at::empty({0}, input.options());
  }
  reflection_pad2d_out_template(output, input, padding);
  return output;
}

Tensor& reflection_pad2d_backward_out_cpu(
    const Tensor& grad_output, const Tensor& input, IntArrayRef padding, Tensor& grad_input) {
  reflection_pad2d_backward_out_template(
    grad_input, grad_output, input, padding);
  return grad_input;
}

Tensor reflection_pad2d_backward_cpu(
    const Tensor& grad_output, const Tensor& input, IntArrayRef padding) {
  auto grad_input = at::empty({0}, input.options());
  reflection_pad2d_backward_out_template(
    grad_input, grad_output, input, padding);
  return grad_input;
}

TORCH_IMPL_FUNC(reflection_pad3d_out_cpu) (
    const Tensor& input, IntArrayRef padding, const Tensor& output) {
  reflection_pad3d_kernel(kCPU, output, input, padding);
}

TORCH_IMPL_FUNC(reflection_pad3d_backward_out_cpu) (
    const Tensor& grad_output, const Tensor& input, IntArrayRef padding, const Tensor& grad_input) {
  grad_input.zero_();
  reflection_pad3d_backward_kernel(kCPU, grad_input, grad_output, padding);
}

DEFINE_DISPATCH(reflection_pad1d_kernel);
DEFINE_DISPATCH(reflection_pad1d_backward_kernel);
DEFINE_DISPATCH(reflection_pad2d_kernel);
DEFINE_DISPATCH(reflection_pad2d_backward_kernel);
DEFINE_DISPATCH(reflection_pad3d_kernel);
DEFINE_DISPATCH(reflection_pad3d_backward_kernel);

} // namespace native
} // namespace at
