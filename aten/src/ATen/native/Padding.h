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

namespace padding {

template <int dim>
static inline void check_valid_input(const Tensor& input, IntArrayRef padding) {

  TORCH_CHECK(padding.size() == 2 * dim,
      "padding size is expected to be ", 2 * dim,
      ", but got: ", padding.size());

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

  // allow empty batch size but not other dimensions.
  TORCH_CHECK(valid_batch_mode || valid_non_batch_mode,
      "Expected ", dim + 1, "D or ", dim + 2,
      "D (batch mode) tensor with possibly 0 batch size and other non-zero dimensions for input, but got: ",
      input.sizes());
}

} // namespace padding

} // at::native
