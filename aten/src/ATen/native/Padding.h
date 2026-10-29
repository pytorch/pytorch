#pragma once

#include <ATen/core/DimVector.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/DispatchStub.h>
#include <c10/util/irange.h>

#include <array>

namespace at::native {

using padding_fn = void (*)(const Tensor&, const Tensor&, IntArrayRef);

// reflection padding
DECLARE_DISPATCH(padding_fn, reflection_pad1d_kernel)
DECLARE_DISPATCH(padding_fn, reflection_pad1d_backward_kernel)
DECLARE_DISPATCH(padding_fn, reflection_pad2d_kernel)
DECLARE_DISPATCH(padding_fn, reflection_pad2d_backward_kernel)
DECLARE_DISPATCH(padding_fn, reflection_pad3d_kernel)
DECLARE_DISPATCH(padding_fn, reflection_pad3d_backward_kernel)

// replication padding
DECLARE_DISPATCH(padding_fn, replication_pad1d_kernel)
DECLARE_DISPATCH(padding_fn, replication_pad1d_backward_kernel)
DECLARE_DISPATCH(padding_fn, replication_pad2d_kernel)
DECLARE_DISPATCH(padding_fn, replication_pad2d_backward_kernel)
DECLARE_DISPATCH(padding_fn, replication_pad3d_kernel)
DECLARE_DISPATCH(padding_fn, replication_pad3d_backward_kernel)

// Shape checks shared by reflection_pad{1,2,3}d and replication_pad{1,2,3}d, on
// every backend. The messages match the meta implementations in
// _meta_registrations.py (_pad{1,2,3}d_common and friends) so that eager and
// torch.compile report failures identically; keep the two in sync.
//
// `padding` follows the torch.nn.functional.pad convention: pair `i` covers the
// input dimension `input.dim() - 1 - i`, i.e. the pairs run inwards-out while
// the D/H/W labels in the messages run the other way.
namespace padding {

inline void check_valid_input(const Tensor& input, IntArrayRef padding, int64_t dim) {
  TORCH_CHECK(static_cast<int64_t>(padding.size()) == 2 * dim,
      "padding size is expected to be ", 2 * dim,
      ", but got: ", padding.size());

  const int64_t input_dim = input.dim();
  const bool is_batch_mode = input_dim == dim + 2;

  // Allow an empty batch size, but no other empty dimension.
  bool valid = is_batch_mode || input_dim == dim + 1;
  for (const auto d : c10::irange(is_batch_mode ? 1 : 0, input_dim)) {
    valid = valid && input.size(d) != 0;
  }

  TORCH_CHECK(valid,
      "Expected ", dim + 1, "D or ", dim + 2,
      "D (batch mode) tensor with possibly 0 batch size and other non-zero dimensions for input, but got: ",
      input.sizes());
}

// TODO(#196457): remove this overload once the torch-xpu-ops pin in
// third_party/xpu.txt passes the rank as an argument. Its ReflectionPadKernels
// still calls check_valid_input<2>(input, padding). torch-xpu-ops compiles every
// TU with -DUSE_XPU (see its cmake/BuildFlags.cmake), while in-tree USE_XPU is
// PRIVATE to torch_xpu, so this stays out of torch_cpu.
#ifdef USE_XPU
template <int dim>
inline void check_valid_input(const Tensor& input, IntArrayRef padding) {
  check_valid_input(input, padding, dim);
}
#endif

// A reflection cannot reach past the edge of the input, so every pad must be
// strictly smaller than the dimension it pads. Replication has no such limit.
inline void check_pad_within_input(const Tensor& input, IntArrayRef padding, int64_t dim) {
  const int64_t ndim = input.dim();
  for (const auto i : c10::irange(dim)) {
    const auto d = ndim - 1 - i;
    TORCH_CHECK(padding[2 * i] < input.size(d) && padding[2 * i + 1] < input.size(d),
        "Argument #", 4 + 2 * i, ": Padding size should be less than the corresponding "
        "input dimension, but got: padding (", padding[2 * i], ", ", padding[2 * i + 1],
        ") at dimension ", d, " of input ", input.sizes());
  }
}

// Validates the forward inputs and returns the full output shape; only the
// trailing `dim` dimensions differ from the input.
inline DimVector pad_shape_check(
    const Tensor& input, IntArrayRef padding, int64_t dim, bool is_reflection) {
  check_valid_input(input, padding, dim);
  if (is_reflection) {
    check_pad_within_input(input, padding, dim);
  }

  const int64_t ndim = input.dim();
  DimVector output_size(input.sizes());
  bool valid_output = true;

  for (const auto i : c10::irange(dim)) {
    const auto d = ndim - 1 - i;
    output_size[d] = input.size(d) + padding[2 * i] + padding[2 * i + 1];
    valid_output = valid_output && output_size[d] >= 1;
  }

  const auto in_spatial = input.sizes().slice(ndim - dim);
  const auto out_spatial = IntArrayRef(output_size).slice(ndim - dim);
  if (dim == 1) {
    TORCH_CHECK(valid_output,
        "input (W: ", in_spatial[0], ") is too small."
        " Calculated output W: ", out_spatial[0]);
  } else if (dim == 2) {
    TORCH_CHECK(valid_output,
        "Calculated output H: ", out_spatial[0], " W: ", out_spatial[1],
        " must be >= 1 in every dimension"
        " (input H: ", in_spatial[0], ", W: ", in_spatial[1], ")");
  } else {
    TORCH_CHECK(valid_output,
        "Calculated output D: ", out_spatial[0], " H: ", out_spatial[1], " W: ", out_spatial[2],
        " must be >= 1 in every dimension"
        " (input D: ", in_spatial[0], ", H: ", in_spatial[1], ", W: ", in_spatial[2], ")");
  }
  return output_size;
}

// Validates gradOutput against the shape the forward pass would have produced.
// The channel check guards a segfault: the kernels iterate over the input's
// channels and index gradOutput with the same counter (pytorch/pytorch#142834).
inline void pad_backward_shape_check(
    const Tensor& grad_output, const Tensor& input, IntArrayRef padding, int64_t dim) {
  constexpr std::array<const char*, 3> labels = {"depth", "height", "width"};
  const int64_t ndim = input.dim();

  const auto channel_dim = ndim - dim - 1;
  TORCH_CHECK(input.size(channel_dim) == grad_output.size(channel_dim),
      "grad_output channel unexpected. Expected: ", input.size(channel_dim),
      ", Got: ", grad_output.size(channel_dim));

  for (const auto i : c10::irange(dim)) {
    const auto d = ndim - 1 - i;
    const auto output_size = input.size(d) + padding[2 * i] + padding[2 * i + 1];
    TORCH_CHECK(output_size == grad_output.size(d),
        "grad_output ", labels[labels.size() - 1 - i], " unexpected. Expected: ", output_size,
        ", Got: ", grad_output.size(d));
  }
}

} // namespace padding

} // namespace at::native
