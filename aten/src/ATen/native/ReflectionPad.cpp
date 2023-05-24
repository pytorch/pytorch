#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/TensorMeta.h>
#include <ATen/quantized/Quantizer.h>
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
    set_output_raw_strided(0, {nplane, output_w}, {}, input.options());
  } else {
    set_output_raw_strided(0, {nbatch, nplane, output_w}, {}, input.options());
  }
}

TORCH_META_FUNC(reflection_pad1d_backward)(const Tensor& grad_output,
    const Tensor& input,
    IntArrayRef padding) {
  int64_t dim_w = 1;
  int64_t nbatch = 1;

  if (input.ndimension() == 3) {
    nbatch = input.size(0);
    (void)nbatch;
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

  set_output_raw_strided(0, input.sizes(), {}, input.options());
}
} // namespace meta

namespace native {

namespace {

template <typename scalar_t>
static void reflection_pad1d_out_frame(
    scalar_t *input_p, scalar_t *output_p,
    int64_t nplane,
    int64_t input_w, int64_t output_w,
    int64_t pad_l) {
  int64_t i_start_x = std::max(int64_t(0), -pad_l);
  int64_t o_start_x = std::max(int64_t(0), pad_l);

  at::parallel_for(0, nplane, 0, [&](int64_t start, int64_t end) {
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    int64_t ip_x;
    for (const auto k : c10::irange(start, end)) {
      for (const auto j : c10::irange(output_w)) {
        if (j < pad_l) {
          ip_x = pad_l * 2 - j;
        } else if (j >= pad_l && j < input_w + pad_l) {
          ip_x = j;
        } else {
          ip_x = (input_w + pad_l - 1) * 2 - j;
        }
        ip_x = ip_x - o_start_x + i_start_x;

        scalar_t *dest_p = output_p + k*output_w + j;
        scalar_t *src_p = input_p + k*input_w + ip_x;
        *dest_p = *src_p;
      }
    }
  });
}

template <typename scalar_t>
inline void reflection_pad1d_out_loop(
    scalar_t *input_p, scalar_t *output_p,
    int64_t nbatch, int64_t nplane,
    int64_t input_w, int64_t output_w,
    int64_t pad_l) {
  at::parallel_for(0, nbatch, 0, [&](int64_t start, int64_t end) {
    for (const auto p : c10::irange(start, end)) {
      reflection_pad1d_out_frame<scalar_t>(
        input_p + p * nplane * input_w,
        output_p + p * nplane * output_w,
        nplane,
        input_w, output_w,
        pad_l);
    }
  });
}

void reflection_pad1d_out_template(
    const Tensor& output, const Tensor& input_, IntArrayRef padding) {
  /* get contiguous input */
  Tensor input = input_.contiguous();

  if (input.ndimension() == 2) {
    if (input.is_quantized()) {
      AT_DISPATCH_QINT_TYPES(input.scalar_type(), "qreflection_pad1d", [&]() {
        reflection_pad1d_out_frame<scalar_t>(
          input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
          input.size(0),
          input.size(1), output.size(-1),
          padding[0]);
      });
    } else {
      AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(input.scalar_type(), "reflection_pad1d", [&] {
        reflection_pad1d_out_frame<scalar_t>(
          input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
          input.size(0),
          input.size(1), output.size(-1),
          padding[0]);
      });
    }
  } else {
    if (input.is_quantized()) {
      AT_DISPATCH_QINT_TYPES(input.scalar_type(), "qreflection_pad1d", [&]() {
        reflection_pad1d_out_loop<scalar_t>(
          input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
          output.size(0), input.size(1),
          input.size(2), output.size(-1),
          padding[0]);
      });
    } else {
      AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(input.scalar_type(), "reflection_pad1d", [&] {
        reflection_pad1d_out_loop<scalar_t>(
          input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
          output.size(0), input.size(1),
          input.size(2), output.size(-1),
          padding[0]);
      });
    }
  }
}

template <typename scalar_t>
static void reflection_pad1d_backward_out_frame(
    scalar_t * grad_input, scalar_t * grad_output,
    int64_t nplane,
    int64_t input_w, int64_t output_w,
    int64_t pad_l) {
  int64_t i_start_x = std::max(int64_t(0), -pad_l);
  int64_t o_start_x = std::max(int64_t(0), pad_l);

  at::parallel_for(0, nplane, 0, [&](int64_t start, int64_t end) {
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    int64_t ip_x;
    for (const auto k : c10::irange(start, end)) {
      for (const auto j : c10::irange(output_w)) {
        if (j < pad_l) {
          ip_x = pad_l * 2 - j;
        } else if (j >= pad_l && j < input_w + pad_l) {
          ip_x = j;
        } else {
          ip_x = (input_w + pad_l - 1) * 2 - j;
        }
        ip_x = ip_x - o_start_x + i_start_x;

        scalar_t *src_p = grad_output + k*output_w + j;
        scalar_t *dest_p = grad_input + k*input_w + ip_x;
        *dest_p += *src_p;
      }
    }
  });
}

template <typename scalar_t>
inline void reflection_pad1d_backward_out_loop(
    scalar_t *grad_input, scalar_t *grad_output,
    int64_t nbatch, int64_t nplane,
    int64_t input_w, int64_t output_w,
    int64_t pad_l) {
  at::parallel_for(0, nbatch, 0, [&](int64_t start, int64_t end) {
    for (const auto p : c10::irange(start, end)) {
      reflection_pad1d_backward_out_frame<scalar_t>(
        grad_input + p * nplane * input_w,
        grad_output + p * nplane * output_w,
        nplane,
        input_w, output_w,
        pad_l);
    }
  });
}

template <typename scalar_t>
static void reflection_pad2d_out_frame(
    scalar_t * input_p, scalar_t * output_p,
    int64_t nplane,
    int64_t input_w, int64_t input_h,
    int64_t output_w, int64_t output_h,
    int64_t pad_l, int64_t pad_t) {
  auto i_start_x = std::max(int64_t(0), -pad_l);
  auto i_start_y = std::max(int64_t(0), -pad_t);
  auto o_start_x = std::max(int64_t(0), pad_l);
  auto o_start_y = std::max(int64_t(0), pad_t);

  at::parallel_for(0, nplane, 0, [&](int64_t start, int64_t end) {
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    int64_t ip_x, ip_y;
    for (const auto k : c10::irange(start, end)) {
      for (const auto i : c10::irange(output_h)) {
        for (const auto j : c10::irange(output_w)) {
          if (j < pad_l) {
            ip_x = pad_l * 2 - j;
          } else if (j >= pad_l && j < input_w + pad_l) {
            ip_x = j;
          } else {
            ip_x = (input_w + pad_l - 1) * 2 - j;
          }
          ip_x = ip_x - o_start_x + i_start_x;

          if (i < pad_t) {
            ip_y = pad_t * 2 - i;
          } else if (i >= pad_t && i < input_h + pad_t) {
            ip_y = i;
          } else {
            ip_y = (input_h + pad_t - 1) * 2 - i;
          }
          ip_y = ip_y - o_start_y + i_start_y;

          scalar_t *dest_p = output_p + k*output_w*output_h + i * output_w + j;
          scalar_t *src_p = input_p + k*input_w*input_h + ip_y * input_w + ip_x;
          *dest_p = *src_p;
        }
      }
    }
  });
}

template <typename scalar_t>
inline void reflection_pad2d_out_loop(
    scalar_t * input_p, scalar_t * output_p,
    int64_t nbatch, int64_t nplane,
    int64_t input_w, int64_t input_h,
    int64_t output_w, int64_t output_h,
    int64_t pad_l, int64_t pad_t) {
  at::parallel_for(0, nbatch, 0, [&](int64_t start, int64_t end) {
    for (const auto p : c10::irange(start, end)) {
      reflection_pad2d_out_frame(
        input_p + p * nplane * input_w * input_h,
        output_p + p * nplane * output_w * output_h,
        nplane,
        input_w, input_h, output_w, output_h,
        pad_l, pad_t);
    }
  });
}

void reflection_pad2d_out_template(
    Tensor &output, const Tensor &input_, IntArrayRef padding) {
  int dim_w = 2;
  int dim_h = 1;
  int dim_slices = 0;
  int64_t nbatch = 1;

  bool valid_dims = input_.size(1) != 0 && input_.size(2) != 0;
  TORCH_CHECK(
      (input_.ndimension() == 3 && valid_dims) ||
      (input_.ndimension() == 4 && valid_dims && input_.size(3) != 0),
      "3D or 4D (batch mode) tensor expected for input, but got: ", input_);

  if (input_.ndimension() == 4) {
    nbatch = input_.size(0);
    dim_w++;
    dim_h++;
    dim_slices++;
  }

  /* sizes */
  int64_t pad_l = padding[0];
  int64_t pad_r = padding[1];
  int64_t pad_t = padding[2];
  int64_t pad_b = padding[3];

  int64_t nplane = input_.size(dim_slices);
  int64_t input_h = input_.size(dim_h);
  int64_t input_w = input_.size(dim_w);
  int64_t output_h = input_h + pad_t + pad_b;
  int64_t output_w  = input_w + pad_l + pad_r;

  TORCH_CHECK(pad_l < input_w && pad_r < input_w,
    "Argument #4: Padding size should be less than the corresponding "
    "input dimension, but got: padding (", pad_l, ", ", pad_r,
    ") at dimension ", dim_w, " of input ", input_.ndimension());

  TORCH_CHECK(pad_t < input_h && pad_b < input_h,
    "Argument #6: Padding size should be less than the corresponding "
    "input dimension, but got: padding (", pad_t, ", ", pad_b,
    ") at dimension ", dim_h, " of input ", input_.ndimension());

  TORCH_CHECK(output_w >= 1 || output_h >= 1,
    "input (H: ", input_h, ", W: ", input_w, ")is too small. Calculated "
    "output H: ", output_h, " W: ", output_w);

  /* get contiguous input */
  Tensor input = input_.contiguous();

  if (input.ndimension() == 3) {
    /* resize output */
    output.resize_({nplane, output_h, output_w});
    if (input.is_quantized()) {
      AT_DISPATCH_QINT_TYPES(input.scalar_type(), "qreflection_pad2d", [&] {
        reflection_pad2d_out_frame(
          input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
          nplane,
          input_w, input_h, output_w, output_h,
          pad_l, pad_t);
      });
    } else {
      AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(input.scalar_type(), "reflection_pad2d", [&] {
        reflection_pad2d_out_frame(
          input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
          nplane,
          input_w, input_h, output_w, output_h,
          pad_l, pad_t);
      });
    }
  } else {
    /* resize output */
    output.resize_({nbatch, nplane, output_h, output_w});
    if (input.is_quantized()) {
      AT_DISPATCH_QINT_TYPES(input.scalar_type(), "qreflection_pad2d", [&] {
        reflection_pad2d_out_loop(
          input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
          nbatch, nplane,
          input_w, input_h, output_w, output_h,
          pad_l, pad_t);
      });
    } else {
      AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(input.scalar_type(), "reflection_pad2d", [&] {
        reflection_pad2d_out_loop(
          input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
          nbatch, nplane,
          input_w, input_h, output_w, output_h,
          pad_l, pad_t);
      });
    }
  }
}

template <typename scalar_t>
static void reflection_pad2d_backward_out_frame(
    scalar_t *grad_input, scalar_t *grad_output,
    int64_t nplane,
    int64_t input_w, int64_t input_h,
    int64_t output_w, int64_t output_h,
    int64_t pad_l, int64_t pad_t) {
  auto i_start_x = std::max(int64_t(0), -pad_l);
  auto i_start_y = std::max(int64_t(0), -pad_t);
  auto o_start_x = std::max(int64_t(0), pad_l);
  auto o_start_y = std::max(int64_t(0), pad_t);

  at::parallel_for(0, nplane, 0, [&](int64_t start, int64_t end) {
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    int64_t ip_x, ip_y;
    for (const auto k : c10::irange(start, end)) {
      for (const auto i : c10::irange(output_h)) {
        for (const auto j : c10::irange(output_w)) {
          if (j < pad_l) {
            ip_x = pad_l * 2 - j;
          } else if (j >= pad_l && j < input_w + pad_l) {
            ip_x = j;
          } else {
            ip_x = (input_w + pad_l - 1) * 2 - j;
          }
          ip_x = ip_x - o_start_x + i_start_x;

          if (i < pad_t) {
            ip_y = pad_t * 2 - i;
          } else if (i >= pad_t && i < input_h + pad_t) {
            ip_y = i;
          } else {
            ip_y = (input_h + pad_t - 1) * 2 - i;
          }
          ip_y = ip_y - o_start_y + i_start_y;

          scalar_t *src_p =
            grad_output + k * output_w * output_h + i * output_w + j;
          scalar_t *dest_p =
            grad_input + k * input_w * input_h + ip_y * input_w + ip_x;
          *dest_p += *src_p;
        }
      }
    }
  });
}

template <typename scalar_t>
inline void reflection_pad2d_backward_out_loop(
    scalar_t *grad_input, scalar_t *grad_output,
    int64_t nbatch, int64_t nplane,
    int64_t input_w, int64_t input_h,
    int64_t output_w, int64_t output_h,
    int64_t pad_l, int64_t pad_t) {
  at::parallel_for(0, nbatch, 0, [&](int64_t start, int64_t end) {
    for (const auto p : c10::irange(start, end)) {
      reflection_pad2d_backward_out_frame(
        grad_input + p * nplane * input_h * input_w,
        grad_output + p * nplane * output_h * output_w,
        nplane,
        input_w, input_h, output_w, output_h,
        pad_l, pad_t);
    }
  });
}

void reflection_pad2d_backward_out_template(
    Tensor &grad_input, const Tensor &grad_output_,
    const Tensor &input, IntArrayRef padding) {
  int dim_w = 2;
  int dim_h = 1;
  int dim_plane = 0;
  int64_t nbatch = 1;

  if (input.ndimension() == 4) {
    nbatch = input.size(0);
    dim_w++;
    dim_h++;
    dim_plane++;
  }

  /* sizes */
  int64_t pad_l = padding[0];
  int64_t pad_r = padding[1];
  int64_t pad_t = padding[2];
  int64_t pad_b = padding[3];

  int64_t nplane = input.size(dim_plane);
  int64_t input_h = input.size(dim_h);
  int64_t input_w = input.size(dim_w);
  int64_t output_h = input_h + pad_t + pad_b;
  int64_t output_w  = input_w + pad_l + pad_r;

  TORCH_CHECK(output_w == grad_output_.size(dim_w),
    "gradOutput width unexpected. Expected: ", output_w, ", Got: ",
    grad_output_.size(dim_w));

  TORCH_CHECK(output_h == grad_output_.size(dim_h),
    "gradOutput height unexpected. Expected: ", output_h, ", Got: ",
    grad_output_.size(dim_h));

  /* get contiguous gradOutput */
  Tensor grad_output = grad_output_.contiguous();

  /* backprop */
  if (input.ndimension() == 3) {
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      grad_output.scalar_type(), "reflection_pad2d_backward", [&] {
        reflection_pad2d_backward_out_frame(
          grad_input.data_ptr<scalar_t>(), grad_output.data_ptr<scalar_t>(),
          nplane,
          input_w, input_h, output_w, output_h,
          pad_l, pad_t);
      }
    );
  } else {
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      grad_output.scalar_type(), "reflection_pad2d_backward", [&] {
        reflection_pad2d_backward_out_loop(
          grad_input.data_ptr<scalar_t>(), grad_output.data_ptr<scalar_t>(),
          nbatch, nplane,
          input_w, input_h, output_w, output_h,
          pad_l, pad_t);
      }
    );
  }
}
template <typename F>
inline void parallel_reflection_pad3d(
    int64_t nplane,
    int64_t input_w, int64_t input_h, int64_t input_d,
    int64_t output_w, int64_t output_h, int64_t output_d,
    int64_t pad_left, int64_t pad_top, int64_t pad_front,
    const F& f) {

  auto i_start_x = std::max(int64_t(0), -pad_left);
  auto i_start_y = std::max(int64_t(0), -pad_top);
  auto i_start_z = std::max(int64_t(0), -pad_front);
  auto o_start_x = std::max(int64_t(0), pad_left);
  auto o_start_y = std::max(int64_t(0), pad_top);
  auto o_start_z = std::max(int64_t(0), pad_front);

  at::parallel_for(0, nplane, 0, [&](int64_t start, int64_t end) {
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    int64_t ip_x, ip_y, ip_z;
    for (const auto k : c10::irange(start, end)) {
      for (const auto op_z : c10::irange(output_d)) {
        for (const auto op_y : c10::irange(output_h)) {
          for (const auto op_x : c10::irange(output_w)) {
            if (op_x < pad_left) {
              ip_x = pad_left * 2 - op_x;
            } else if (op_x >= pad_left && op_x < input_w + pad_left) {
              ip_x = op_x;
            } else {
              ip_x = (input_w + pad_left - 1) * 2 - op_x;
            }
            ip_x = ip_x - o_start_x + i_start_x;

            if (op_y < pad_top) {
              ip_y = pad_top * 2 - op_y;
            } else if (op_y >= pad_top && op_y < input_h + pad_top) {
              ip_y = op_y;
            } else {
              ip_y = (input_h + pad_top - 1) * 2 - op_y;
            }
            ip_y = ip_y - o_start_y + i_start_y;

            if (op_z < pad_front) {
              ip_z = pad_front * 2 - op_z;
            } else if (op_z >= pad_front && op_z < input_d + pad_front) {
              ip_z = op_z;
            } else {
              ip_z = (input_d + pad_front - 1) * 2 - op_z;
            }
            ip_z = ip_z - o_start_z + i_start_z;

            f(k, op_z, op_y, op_x, ip_z, ip_y, ip_x);
          }
        }
      }
    }
  });
}

template <typename scalar_t>
static void reflection_pad3d_out_frame(
    scalar_t *input_p, scalar_t *output_p,
    int64_t nplane,
    int64_t input_w, int64_t input_h, int64_t input_d,
    int64_t output_w, int64_t output_h, int64_t output_d,
    int64_t pad_left, int64_t pad_top, int64_t pad_front)
{
  parallel_reflection_pad3d(
      nplane,
      input_w,
      input_h,
      input_d,
      output_w,
      output_h,
      output_d,
      pad_left,
      pad_top,
      pad_front,
      [&](int64_t k,
          int64_t op_z,
          int64_t op_y,
          int64_t op_x,
          int64_t ip_z,
          int64_t ip_y,
          int64_t ip_x) {
        scalar_t* dest_p = output_p + k * output_w * output_h * output_d +
            op_z * output_w * output_h + op_y * output_w + op_x;
        scalar_t* src_p = input_p + k * input_w * input_h * input_d +
            ip_z * input_w * input_h + ip_y * input_w + ip_x;
        *dest_p = *src_p;
      });
}

template <typename scalar_t>
static void reflection_pad3d_out_loop(
    scalar_t *input_p, scalar_t *output_p,
    int64_t nbatch, int64_t nplane,
    int64_t input_w, int64_t input_h, int64_t input_d,
    int64_t output_w, int64_t output_h, int64_t output_d,
    int64_t pad_left, int64_t pad_top, int64_t pad_front)
{
  at::parallel_for(0, nbatch, 0, [&](int64_t start, int64_t end) {
    for (const auto p : c10::irange(start, end)) {
      reflection_pad3d_out_frame(
          input_p + p * nplane * input_w * input_h * input_d,
          output_p + p * nplane * output_w * output_h * output_d,
          nplane,
          input_w,
          input_h,
          input_d,
          output_w,
          output_h,
          output_d,
          pad_left,
          pad_top,
          pad_front);
    }
  });
}

template <typename scalar_t>
static void reflection_pad3d_backward_out_frame(
    scalar_t *grad_input, scalar_t *grad_output,
    int64_t nplane,
    int64_t input_w, int64_t input_h, int64_t input_d,
    int64_t output_w, int64_t output_h, int64_t output_d,
    int64_t pad_left, int64_t pad_top, int64_t pad_front
) {
  parallel_reflection_pad3d(
      nplane,
      input_w,
      input_h,
      input_d,
      output_w,
      output_h,
      output_d,
      pad_left,
      pad_top,
      pad_front,
      [&](int64_t k,
          int64_t op_z,
          int64_t op_y,
          int64_t op_x,
          int64_t ip_z,
          int64_t ip_y,
          int64_t ip_x) {
        scalar_t* src_p = grad_output + k * output_w * output_h * output_d +
            op_z * output_w * output_h + op_y * output_w + op_x;
        scalar_t* dest_p = grad_input + k * input_w * input_h * input_d +
            ip_z * input_w * input_h + ip_y * input_w + ip_x;
        *dest_p += *src_p;
      });
}

template <typename scalar_t>
static void reflection_pad3d_backward_out_loop(
    scalar_t *grad_input, scalar_t *grad_output,
    int64_t nbatch, int64_t nplane,
    int64_t input_w, int64_t input_h, int64_t input_d,
    int64_t output_w, int64_t output_h, int64_t output_d,
    int64_t pad_left, int64_t pad_top, int64_t pad_front
) {
  at::parallel_for(0, nbatch, 0, [&](int64_t start, int64_t end) {
    for (const auto p : c10::irange(start, end)) {
      reflection_pad3d_backward_out_frame<scalar_t>(
          grad_input + p * nplane * input_w * input_h * input_d,
          grad_output + p * nplane * output_w * output_h * output_d,
          nplane,
          input_w,
          input_h,
          input_d,
          output_w,
          output_h,
          output_d,
          pad_left,
          pad_top,
          pad_front);
    }
  });
}

} // namespace

// TODO: I tihnk this function should be removed since we implement it with
// TORCH_IMPL_FUNC below
static Tensor& reflection_pad1d_out_cpu(const Tensor& input, IntArrayRef padding,
    Tensor& output) {
  reflection_pad1d_out_template(output, input, padding);
  return output;
}

Tensor& reflection_pad1d_out_quantized_cpu(const Tensor& input, IntArrayRef padding,
    Tensor& output) {
  TORCH_CHECK(input.qscheme() == kPerTensorAffine, "Only per tensor quantization is supported");
  set_quantizer_(output, make_per_tensor_affine_quantizer(input.q_scale(), input.q_zero_point(), input.scalar_type()));
  reflection_pad1d_out_template(output, input, padding);
  return output;
}

TORCH_IMPL_FUNC(reflection_pad1d_out_cpu)
(const Tensor& input, IntArrayRef padding, const Tensor& output) {
  reflection_pad1d_out_template(output, input, padding);
}

TORCH_IMPL_FUNC(reflection_pad1d_backward_out_cpu)(const Tensor& grad_output_,
    const Tensor& input,
    IntArrayRef padding,
    const Tensor& grad_input) {
  grad_input.zero_();

  int64_t dim_plane = 0;
  int64_t dim_w = 1;
  int64_t nbatch = 1;

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

  /* get contiguous grad_output */
  Tensor grad_output = grad_output_.contiguous();

  /* backprop */
  if (input.ndimension() == 2) {
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      grad_input.scalar_type(), "reflection_pad1d_backward_cpu", [&] {
        reflection_pad1d_backward_out_frame(
          grad_input.data_ptr<scalar_t>(), grad_output.data_ptr<scalar_t>(),
          nplane,
          input_w, output_w,
          pad_l);
        }
    );
  } else {
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      grad_input.scalar_type(), "reflection_pad1d_backward_cpu", [&] {
        reflection_pad1d_backward_out_loop(
          grad_input.data_ptr<scalar_t>(),
          grad_output.data_ptr<scalar_t>(),
          nbatch, nplane,
          input_w, output_w,
          pad_l);
      }
    );
  }
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
  grad_input.resize_as_(input);
  grad_input.zero_();
  reflection_pad2d_backward_out_template(
    grad_input, grad_output, input, padding);
  return grad_input;
}

Tensor reflection_pad2d_backward_cpu(
    const Tensor& grad_output,
    const Tensor& input,
    IntArrayRef padding) {
  auto grad_input = at::zeros_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  reflection_pad2d_backward_out_template(
    grad_input, grad_output, input, padding);
  return grad_input;
}

TORCH_IMPL_FUNC(reflection_pad3d_out_cpu)
(const Tensor& input_, IntArrayRef padding, const Tensor& output) {
  int64_t pad_left = padding[0];
  int64_t pad_top = padding[2];
  int64_t pad_front = padding[4];
  int64_t dim_w = 3;
  int64_t dim_h = 2;
  int64_t dim_d = 1;
  int64_t dim_plane = 0;
  bool batch_mode = (input_.dim() == 5);

  if (batch_mode) {
    dim_w++;
    dim_h++;
    dim_d++;
    dim_plane++;
  }

  int64_t nplane = input_.size(dim_plane);
  int64_t input_w = input_.size(dim_w);
  int64_t input_h = input_.size(dim_h);
  int64_t input_d = input_.size(dim_d);
  int64_t output_w = output.size(dim_w);
  int64_t output_h = output.size(dim_h);
  int64_t output_d = output.size(dim_d);

  auto input = input_.contiguous();

  if (batch_mode) {
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
        kHalf, kBFloat16, input.scalar_type(), "reflection_pad3d_cpu", [&] {
          auto input_data = input.data_ptr<scalar_t>();
          auto output_data = output.data_ptr<scalar_t>();
          auto nbatch = input.size(0);
          reflection_pad3d_out_loop(
              input_data,
              output_data,
              nbatch,
              nplane,
              input_w,
              input_h,
              input_d,
              output_w,
              output_h,
              output_d,
              pad_left,
              pad_top,
              pad_front);
        });
  } else {
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
        kHalf, kBFloat16, input.scalar_type(), "reflection_pad3d_cpu", [&] {
          auto input_data = input.data_ptr<scalar_t>();
          auto output_data = output.data_ptr<scalar_t>();
          reflection_pad3d_out_frame(
              input_data,
              output_data,
              nplane,
              input_w,
              input_h,
              input_d,
              output_w,
              output_h,
              output_d,
              pad_left,
              pad_top,
              pad_front);
        });
  }
}

TORCH_IMPL_FUNC(reflection_pad3d_backward_out_cpu)(const Tensor& grad_output,
    const Tensor& input,
    IntArrayRef padding,
    const Tensor& grad_input) {
  int64_t pad_left = padding[0];
  int64_t pad_top = padding[2];
  int64_t pad_front = padding[4];
  int64_t dim_w = 3;
  int64_t dim_h = 2;
  int64_t dim_d = 1;
  int64_t dim_plane = 0;
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
  int64_t output_d = grad_output.size(dim_d);
  int64_t output_h = grad_output.size(dim_h);
  int64_t output_w = grad_output.size(dim_w);

  auto grad_output_ = grad_output.contiguous();
  if (grad_output_.numel() == 0) {
    return;
  }

  grad_input.zero_();

  if (batch_mode) {
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
        kHalf, kBFloat16, input.scalar_type(), "reflection_pad3d_backward_cpu", [&] {
          reflection_pad3d_backward_out_loop<scalar_t>(
              grad_input.data_ptr<scalar_t>(),
              grad_output_.data_ptr<scalar_t>(),
              input.size(0),
              nplane,
              input_w,
              input_h,
              input_d,
              output_w,
              output_h,
              output_d,
              pad_left,
              pad_top,
              pad_front);
        });
  } else {
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
        kHalf, kBFloat16, input.scalar_type(), "reflection_pad3d_backward_cpu", [&] {
          reflection_pad3d_backward_out_frame<scalar_t>(
              grad_input.data_ptr<scalar_t>(),
              grad_output_.data_ptr<scalar_t>(),
              nplane,
              input_w,
              input_h,
              input_d,
              output_w,
              output_h,
              output_d,
              pad_left,
              pad_top,
              pad_front);
        });
  }
}
} // namespace native
} // namespace at
