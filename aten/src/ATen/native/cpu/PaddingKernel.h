#pragma once

#include <ATen/core/Tensor.h>
#include <ATen/native/DispatchStub.h>

namespace at::native {

using padding_fn = void (*)(const Tensor&, const Tensor&, IntArrayRef);

// reflection padding
DECLARE_DISPATCH(padding_fn, reflection_pad1d_kernel);
DECLARE_DISPATCH(padding_fn, reflection_pad1d_backward_kernel);
DECLARE_DISPATCH(padding_fn, reflection_pad2d_kernel);
DECLARE_DISPATCH(padding_fn, reflection_pad2d_backward_kernel);
DECLARE_DISPATCH(padding_fn, reflection_pad3d_kernel);
DECLARE_DISPATCH(padding_fn, reflection_pad3d_backward_kernel);

// replication padding
DECLARE_DISPATCH(padding_fn, replication_pad1d_kernel);
DECLARE_DISPATCH(padding_fn, replication_pad1d_backward_kernel);
DECLARE_DISPATCH(padding_fn, replication_pad2d_kernel);
DECLARE_DISPATCH(padding_fn, replication_pad2d_backward_kernel);
DECLARE_DISPATCH(padding_fn, replication_pad3d_kernel);
DECLARE_DISPATCH(padding_fn, replication_pad3d_backward_kernel);

namespace padding {

template <int dim>
static inline void check_valid_input(const Tensor& input) {

  int input_dim = input.dim();

  bool is_batch_mode = input_dim == (dim + 2);

  bool valid_batch_mode = is_batch_mode;
  bool valid_non_batch_mode = !is_batch_mode;

  if (is_batch_mode) {
    // allow batch size of 0-dim.
    for (const auto d : c10::irange(1, input_dim)) {
      valid_batch_mode = valid_batch_mode && input.size(d) != 0;
    }
  } else {
    for (const auto d : c10::irange(0, input_dim)) {
      valid_non_batch_mode = valid_non_batch_mode && input.size(d) != 0;
    }
  }

  TORCH_CHECK(valid_batch_mode || valid_non_batch_mode,
      "Expected ", dim + 1, "D or ", dim + 2,
      "D (batch mode) tensor with possibly 0 batch size and other non-zero dimensions for input, but got: ",
      input.sizes());
}

template <int dim>
static inline void check_valid_input_backward(
    const Tensor& grad_output, const Tensor& input, IntArrayRef padding) {

  if constexpr (dim == 1) {
    int64_t pad_l = padding[0];
    int64_t pad_r = padding[1];

    int dimw = -1;
    int64_t input_width = input.size(dimw);
    int64_t output_width = input_width + pad_l + pad_r;

    TORCH_CHECK(output_width == grad_output.size(dimw),
        "gradOutput width unexpected. Expected: ", output_width,
        " Got: ", grad_output.size(dimw));
  } else if constexpr(dim == 2) {
    int64_t pad_l = padding[0];
    int64_t pad_r = padding[1];
    int64_t pad_t = padding[2];
    int64_t pad_b = padding[3];

    int dimh = -2;
    int dimw = -1;

    int64_t input_height = input.size(dimh);
    int64_t input_width = input.size(dimw);
    int64_t output_height = input_height + pad_t + pad_b;
    int64_t output_width  = input_width + pad_l + pad_r;

    TORCH_CHECK(output_width == grad_output.size(dimw),
        "gradOutput width unexpected. Expected: ", output_width, ", Got: ",
        grad_output.size(dimw));
    TORCH_CHECK(output_height == grad_output.size(dimh),
        "gradOutput height unexpected. Expected: ", output_height, ", Got: ",
        grad_output.size(dimh));
  } else {
    int64_t pleft = padding[0];
    int64_t pright = padding[1];
    int64_t ptop = padding[2];
    int64_t pbottom = padding[3];
    int64_t pfront = padding[4];
    int64_t pback = padding[5];

    int dimd = -3;
    int dimh = -2;
    int dimw = -1;

    int64_t input_depth = input.size(dimd);
    int64_t input_height = input.size(dimh);
    int64_t input_width = input.size(dimw);
    int64_t output_depth = input_depth + pfront + pback;
    int64_t output_height = input_height + ptop + pbottom;
    int64_t output_width  = input_width + pleft + pright;

    TORCH_CHECK(output_width == grad_output.size(dimw),
        "gradOutput width unexpected. Expected: ", output_width, ", Got: ",
        grad_output.size(dimw));
    TORCH_CHECK(output_height == grad_output.size(dimh),
        "gradOutput height unexpected. Expected: ", output_height, ", Got: ",
        grad_output.size(dimh));
    TORCH_CHECK(output_depth == grad_output.size(dimd),
        "gradOutput depth unexpected. Expected: ", output_depth, ", Got: ",
        grad_output.size(dimd));
  }
}

} // namespace padding

} // at::native
