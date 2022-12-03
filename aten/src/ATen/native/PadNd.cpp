#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/PadNd.h>
#include <ATen/core/Tensor.h>

#include <c10/util/irange.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_empty_affine_quantized.h>
#include <ATen/ops/_pad_circular.h>
#include <ATen/ops/_pad_circular_native.h>
#include <ATen/ops/_pad_enum_native.h>
#include <ATen/ops/constant_pad_nd.h>
#include <ATen/ops/constant_pad_nd_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/pad_native.h>
#include <ATen/ops/reflection_pad1d.h>
#include <ATen/ops/reflection_pad2d.h>
#include <ATen/ops/reflection_pad3d.h>
#include <ATen/ops/replication_pad1d.h>
#include <ATen/ops/replication_pad2d.h>
#include <ATen/ops/replication_pad3d.h>
#endif

namespace at { namespace native {

Tensor constant_pad_nd(const Tensor& self, IntArrayRef pad, const Scalar& value) {
    TORCH_CHECK(pad.size() % 2 == 0, "Length of pad must be even but instead it equals ",
             pad.size());

    auto input_sizes = self.sizes();
    auto l_inp = self.dim();

    auto l_pad = pad.size() / 2;
    auto l_diff = l_inp - l_pad;
    TORCH_CHECK(l_inp >= (int64_t)l_pad, "Length of pad should be no more than twice the number of "
             "dimensions of the input. Pad length is ", pad.size(), "while the input has ",
             l_inp, "dimensions.");

    std::vector<int64_t> new_shape;

    bool all_pads_non_positive = true;

    auto c_input = self;
    for (const auto i : c10::irange(l_diff, l_inp)) {
        auto pad_idx = 2 * (l_inp - i - 1);
        if (pad[pad_idx] < 0) {
            c_input = c_input.narrow(i, -pad[pad_idx], c_input.size(i) + pad[pad_idx]);
        } else if (pad[pad_idx] != 0) {
            all_pads_non_positive = false;
        }
        if (pad[pad_idx + 1] < 0) {
            c_input = c_input.narrow(i, 0, c_input.size(i) + pad[pad_idx + 1]);
        } else if (pad[pad_idx + 1] != 0) {
            all_pads_non_positive = false;
        }
    }

    // if none of the pads are positive we can optimize and just return the result
    // of calling .narrow() on the input
    if (all_pads_non_positive) {
        return c_input.clone();
    }


    for (size_t i = 0; i < (size_t)l_diff; i ++) {
        new_shape.emplace_back(input_sizes[i]);
    }

    for (const auto i : c10::irange((size_t)l_pad)) {
        auto pad_idx = pad.size() - ((i + 1) * 2);
        auto new_dim = input_sizes[l_diff + i] + pad[pad_idx] + pad[pad_idx + 1];
        TORCH_CHECK(new_dim > 0, "The input size ", input_sizes[l_diff + i], ", plus negative padding ",
                 pad[pad_idx], " and ", pad[pad_idx + 1], " resulted in a negative output size, "
                 "which is invalid. Check dimension ", l_diff + i, " of your input.");
        new_shape.emplace_back(new_dim);
    }

    at::Tensor output;
    const auto memory_format = self.suggest_memory_format();
    if (self.is_quantized()) {
        const auto qscheme = self.qscheme();
        TORCH_CHECK(qscheme == kPerTensorAffine || qscheme == kPerTensorSymmetric,
                    "Only per-tensor padding is supported.");
        output = at::_empty_affine_quantized(
            new_shape, self.options().memory_format(memory_format),
            self.q_scale(), self.q_zero_point(), c10::nullopt);
    } else {
        output = at::empty(new_shape, self.options().memory_format(memory_format));
    }
    output.fill_(value);

    auto c_output = output;
    for (const auto i : c10::irange(l_diff, l_inp)) {
        auto pad_idx = 2 * (l_inp - i - 1);
        if (pad[pad_idx] > 0) {
            c_output = c_output.narrow(i, pad[pad_idx], c_output.size(i) - pad[pad_idx]);
        }
        if (pad[pad_idx + 1] > 0) {
            c_output = c_output.narrow(i, 0, c_output.size(i) - pad[pad_idx + 1]);
        }
    }
    c_output.copy_(c_input);
    return output;
}

Tensor _pad_circular_symint(const Tensor &self, c10::SymIntArrayRef padding) {
  const auto in_shape = self.sym_sizes();
  const auto ndim = static_cast<int64_t>(in_shape.size()) - 2;
  TORCH_CHECK(padding.size() + 4 == in_shape.size() * 2,
              "Invalid padding size, expected ", ndim * 2, " but got ", padding.size());

  c10::SymDimVector out_shape(in_shape.size());
  out_shape[0] = in_shape[0];
  out_shape[1] = in_shape[1];

  // Get shape of padded tensor
  for (const auto i : c10::irange(ndim)) {
    const auto pad_l = padding[2 * (ndim - i - 1) + 0];
    const auto pad_r = padding[2 * (ndim - i - 1) + 1];
    const auto size = in_shape[2 + i];
    out_shape[2 + i] = size + pad_l + pad_r;

    TORCH_CHECK(
        pad_l <= size && pad_r <= size,
        "Padding value causes wrapping around more than once.");
    TORCH_CHECK(
        out_shape[2 + i] >= 0,
        "Negative padding value is resulting in an empty dimension");
  }

  auto out = self.new_empty_symint(out_shape, self.options());

  // Put original array into the padded array
  Tensor out_slice = out;
  Tensor in_slice = self;
  const SymInt zero = 0;
  for (const auto i : c10::irange(ndim)) {
    const auto dim = ndim - i + 1;
    const auto pad_l = padding[2*i + 0];
    const auto pad_r = padding[2*i + 1];
    out_slice = out_slice.slice_symint(dim, std::max(pad_l, zero), out_shape[dim] - std::max(pad_r, zero));
    in_slice = in_slice.slice_symint(dim, std::max(-pad_l, zero), in_shape[dim] - std::max(-pad_r, zero));
  }
  out_slice.copy_(in_slice);

  // The following steps first pad the beginning of the tensor (left side),
  // and then pad the end of the tensor (right side).
  // Note: Corners will be written more than once when ndim > 1.
  //
  // Only in cases where padding values are > 0 are when additional copying
  // is required.
  for (const auto i : c10::irange(ndim)) {
    const auto dim = ndim - i + 1;
    const auto pad_l = padding[2*i + 0];
    const auto pad_r = padding[2*i + 1];

    if (pad_l > 0) {
      out_slice = out.slice_symint(dim, 0, pad_l);
      in_slice = out.slice_symint(dim,
                           out_shape[dim] - pad_l - std::max(pad_r, zero),
                           out_shape[dim] - std::max(pad_r, zero));
      out_slice.copy_(in_slice);
    }

    if (pad_r > 0) {
      out_slice = out.slice_symint(dim, out_shape[dim] - pad_r, out_shape[dim]);
      in_slice = out.slice_symint(dim, std::max(pad_l, zero), std::max(pad_l, zero) + pad_r);
      out_slice.copy_(in_slice);
    }
  }

  return out;
}

Tensor _pad_enum_symint(const Tensor &self, c10::SymIntArrayRef pad, int64_t mode_int, c10::optional<double> value) {
  const auto input_dim = self.dim();
  TORCH_CHECK(pad.size() % 2 == 0, "Padding length must be divisible by 2");
  TORCH_CHECK(static_cast<int64_t>(pad.size()) <= input_dim * 2, "Padding length too large");
  auto mode = static_cast<at::padding_mode>(mode_int);

  if (mode == at::padding_mode::constant) {
    return at::constant_pad_nd_symint(self, pad, value.value_or(0.0));
  }
  TORCH_CHECK(!value.has_value() || *value == 0,
              "Padding mode \"", padding_mode_string(mode),
              "\" doesn't take in value argument");

  if (pad.size() == 2 && (input_dim == 2 || input_dim == 3)) {
    switch (mode) {
      case at::padding_mode::reflect: return at::reflection_pad1d_symint(self, pad);
      case at::padding_mode::replicate: return at::replication_pad1d_symint(self, pad);
      case at::padding_mode::circular: return at::_pad_circular_symint(self, pad);
      default: {}
    }
  } else if(pad.size() == 4 && (input_dim == 3 || input_dim == 4)) {
    switch (mode) {
      case at::padding_mode::reflect: return at::reflection_pad2d_symint(self, pad);
      case at::padding_mode::replicate: return at::replication_pad2d_symint(self, pad);
      case at::padding_mode::circular: return at::_pad_circular_symint(self, pad);
      default: {}
    }
  } else if (pad.size() == 6 && (input_dim == 4 || input_dim == 5)) {
    switch (mode) {
      case at::padding_mode::reflect: return at::reflection_pad3d_symint(self, pad);
      case at::padding_mode::replicate: return at::replication_pad3d_symint(self, pad);
      case at::padding_mode::circular: return at::_pad_circular_symint(self, pad);
      default: {}
    }
  }
  C10_THROW_ERROR(NotImplementedError,
      "Only 2D, 3D, 4D, 5D padding with non-constant padding are supported for now");
}

Tensor pad_symint(const Tensor &self, c10::SymIntArrayRef pad, c10::string_view mode, c10::optional<double> value) {
  const auto mode_enum = [&] {
    if (mode == "reflect") {
      return at::padding_mode::reflect;
    } else if (mode == "constant") {
      return at::padding_mode::constant;
    } else if (mode == "replicate") {
      return at::padding_mode::replicate;
    } else if (mode == "circular") {
      return at::padding_mode::circular;
    }
    C10_THROW_ERROR(NotImplementedError,
                    c10::str("Unrecognised padding mode ", mode));
  }();
  return at::native::_pad_enum_symint(self, pad, static_cast<int64_t>(mode_enum), value);
}

}}  // namespace at::native
