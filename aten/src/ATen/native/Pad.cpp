#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/native/Resize.h>
#include <ATen/TensorIndexing.h>

namespace at { namespace native {

// TODO: Consider creating a PadSpecifier class, which would wrap the output of
// this function. Helper functions could take the PadSpecifier instead of
// a Tensor, so that we never accidentally give them the unprocessed pad
// specifier tensor.
Tensor expand_pad_specifier(const Tensor& pad_spec, const char* arg_name, int64_t ndim) {
  auto pad_spec_ndim = pad_spec.dim();
  auto pad_spec_numel = pad_spec.numel();

  // TODO: This logic can probably be simplified
  if (pad_spec_numel == 1) {
    // size [], [1], [2] cases
    TORCH_CHECK(
      (pad_spec_ndim >= 0) && (pad_spec_ndim <= 2),
      "torch.pad: Expected ", arg_name, ".size() to be either ",
      "[], [1], [2], [1, 1], [1, 2], [input.dim(), 1], or [input.dim(), 2], but got ",
      pad_spec.sizes());
    return pad_spec.as_strided({ndim, 2}, {0, 0});

  } else if ((pad_spec_ndim == 2) && (pad_spec.size(0) == ndim) && (pad_spec.size(1) == 1)) {
    // size [ndim, 1] case
    return pad_spec.as_strided({ndim, 2}, {1, 0});

  } else if (pad_spec_numel == 2) {
    // size [1, 1] and [1, 2] cases
    TORCH_CHECK(
      (pad_spec_ndim == 1) || ((pad_spec_ndim == 2) && (pad_spec.size(-1) == 2)),
      "torch.pad: Expected ", arg_name, ".size() to be either ",
      "[], [1], [2], [1, 1], [1, 2], [input.dim(), 1], or [input.dim(), 2], but got ",
      pad_spec.sizes());
    return pad_spec.as_strided({ndim, 2}, {0, 1});

  } else {
    // size [ndim, 2] case
    TORCH_CHECK(
      (pad_spec_ndim == 2) && (pad_spec.size(0) == ndim) && (pad_spec.size(1) == 2),
      "torch.pad: Expected ", arg_name, ".size() to be either ",
      "[], [1], [2], [1, 1], [1, 2], [input.dim(), 1], or [input.dim(), 2], but got ",
      pad_spec.sizes());
    // NOTE: There is no need to restride in this case, since it's
    // already the correct shape
    return pad_spec;
  }
}

// Returns slices that can be used to index the inner section of the padded
// tensor that corresponds with the original unpadded tensor.
//
// NOTE: pad_width must be processed with expand_pad_specifier before
// calling this function
std::vector<at::indexing::TensorIndex> pad_width_to_inner_slices(const Tensor& pad_width, const Tensor& self) {
  std::vector<at::indexing::TensorIndex> slices;
  for (int64_t dim_idx = 0; dim_idx < self.dim(); dim_idx++) {
    auto pad_width_before = pad_width[dim_idx][0].item<int64_t>();
    auto self_dim_size = self.size(dim_idx);

    slices.push_back(at::indexing::TensorIndex(at::indexing::Slice(
      pad_width_before,
      pad_width_before + self_dim_size,
      at::indexing::None)));
  }
  return slices;
}

enum PadMode {
  Constant
};

// Get PadMode enum val from pad mode string
PadMode get_pad_mode_from_str(c10::string_view mode_str) {
  if (mode_str.compare("constant") == 0) {
    return PadMode::Constant;
  }
  TORCH_CHECK(false, "torch.pad: Unrecognized mode: ", mode_str);
}

// Fill the pad result with data from the input tensor
// NOTE: pad_width must be processed with expand_pad_specifier before
// calling this function
void fill_input(Tensor& result, const Tensor& self, const Tensor& pad_width) {
  std::vector<at::indexing::TensorIndex> slices = pad_width_to_inner_slices(pad_width, self);
  result.index_put_(slices, self);
}

// Fill each section of padding for each dimension with the
// corresponding constant value
//
// NOTE: pad_width and constant_values must be processed with
// expand_pad_specifier before calling this function
void fill_constant_pad(Tensor& result, const Tensor& self, const Tensor& pad_width, const Tensor& constant_values) {
  for (int64_t pad_dim_idx = 0; pad_dim_idx < self.dim(); pad_dim_idx++) {
    auto width_before = pad_width[pad_dim_idx][0].item<int64_t>();
    auto width_after = pad_width[pad_dim_idx][1].item<int64_t>();

    // Fill before-padding for this dimension
    if (width_before > 0) {
      std::vector<at::indexing::TensorIndex> slices;
      for (int64_t slice_dim_idx = 0; slice_dim_idx < self.dim(); slice_dim_idx++) {
        if (slice_dim_idx == pad_dim_idx) {
          slices.push_back(at::indexing::TensorIndex(at::indexing::Slice(
              0,
              width_before,
              at::indexing::None)));
        } else if (slice_dim_idx < pad_dim_idx) {
          slices.push_back(at::indexing::TensorIndex(at::indexing::Slice(
              0,
              result.size(slice_dim_idx),
              at::indexing::None)));
        } else {
          auto other_width_before = pad_width[slice_dim_idx][0].item<int64_t>();
          slices.push_back(at::indexing::TensorIndex(at::indexing::Slice(
              other_width_before,
              other_width_before + self.size(slice_dim_idx),
              at::indexing::None)));
        }
      }
      result.index_put_(slices, constant_values[pad_dim_idx][0]);
    }

    // Fill after-padding for this dimension
    if (width_after > 0) {
      std::vector<at::indexing::TensorIndex> slices;
      for (int64_t slice_dim_idx = 0; slice_dim_idx < self.dim(); slice_dim_idx++) {
        if (slice_dim_idx == pad_dim_idx) {
          slices.push_back(at::indexing::TensorIndex(at::indexing::Slice(
              width_before + self.size(slice_dim_idx),
              result.size(slice_dim_idx),
              at::indexing::None)));
        } else if (slice_dim_idx < pad_dim_idx) {
          slices.push_back(at::indexing::TensorIndex(at::indexing::Slice(
              0,
              result.size(slice_dim_idx),
              at::indexing::None)));
        } else {
          auto other_width_before = pad_width[slice_dim_idx][0].item<int64_t>();
          slices.push_back(at::indexing::TensorIndex(at::indexing::Slice(
              other_width_before,
              other_width_before + self.size(slice_dim_idx),
              at::indexing::None)));
        }
      }
      result.index_put_(slices, constant_values[pad_dim_idx][1]);
    }
  }
}

void check_pad_specifier_is_none(const c10::optional<Tensor>& arg, const char* arg_name, c10::string_view mode_str) {
  TORCH_CHECK(!arg.has_value(),
    "torch.pad: Unsupported keyword argument for '", std::string(mode_str),
    "' mode: ", arg_name);
}

// Calculate the size of the padded tensor
//
// NOTE: pad_width must be processed with expand_pad_specifier before
// calling this function
Tensor get_result_size(const Tensor& self, const Tensor& pad_width) {
  auto result_size_tensor = at::tensor(
    self.sizes(),
    TensorOptions().dtype(at::kLong).device(at::kCPU).memory_format(c10::MemoryFormat::Contiguous));

  result_size_tensor += pad_width.sum(1).cpu();

  return result_size_tensor;
}

// Create an IntArrayRef from a tensor
IntArrayRef tensor_to_arrayref(const Tensor& size_tensor) {
  return IntArrayRef(size_tensor.data_ptr<int64_t>(), size_tensor.numel());
}

// Check that a tensor is on the expected device
void check_device(const Tensor& arg, const char* arg_name, at::Device self_device) {
  TORCH_CHECK(arg.device() == self_device,
    "torch.pad: Expected '", arg_name, "' to be on the same device as ",
    "'input' (", self_device, ") but got ", arg.device());
}

Tensor& pad_out_impl(
  const Tensor& self,
  const Tensor& pad_width,
  c10::string_view mode_str,
  const c10::optional<Tensor>& constant_values_opt,
  Tensor& result
) {
  // pad_width must be Long, on CPU, and non-negative
  TORCH_CHECK(pad_width.device() == at::kCPU,
    "torch.pad: Expected 'pad_width' to be on CPU, but got ", pad_width.device());
  TORCH_CHECK(pad_width.scalar_type() == at::ScalarType::Long,
    "torch.pad: Expected 'pad_width' to be Long dtype, but got ",
    pad_width.scalar_type());
  TORCH_CHECK(pad_width.ge(0).all().item<bool>(),
    "torch.pad: Expected 'pad_width' to be non-negative");

  PadMode mode = get_pad_mode_from_str(mode_str);

  if (mode != PadMode::Constant) {
    // constant_values should be none if not using constant mode
    check_pad_specifier_is_none(constant_values_opt, "constant_values", mode_str);
  } else {
    // If using constant_values is given, it should be on same device as self
    if (constant_values_opt.has_value()) {
      check_device(constant_values_opt.value(), "constant_values", self.device());
    }
  }

  // If `out` is given, it must match `self`'s dtype and device
  if (result.defined()) {
    TORCH_CHECK(result.scalar_type() == self.scalar_type(),
      "torch.pad: Expected 'out' dtype ", self.scalar_type(),
      " but got ", result.scalar_type());
    check_device(result, "out", self.device());
  }

  Tensor pad_width_ = at::native::expand_pad_specifier(pad_width, "pad_width", self.dim());

  auto result_size_tensor = get_result_size(self, pad_width_);
  auto result_size = tensor_to_arrayref(result_size_tensor);
  if (result.defined()) {
    at::native::resize_output(result, result_size);
  } else {
    // TODO: In constant mode, if constant_values is a single scalar, we could
    // use at::full here for better performance than fill_constant_pad
    result = at::empty(result_size, self.options());
  }

  if (mode == PadMode::Constant) {
    fill_input(result, self, pad_width_);

    Tensor constant_values = at::native::expand_pad_specifier(
      constant_values_opt.value_or(
        at::zeros({1}, self.options().device(at::kCPU))),
      "constant_values",
      self.dim());

    TORCH_CHECK(
      constant_values.scalar_type() == self.dtype(),
      "torch.pad: Expected constant_values.dtype to match input.dtype (",
      self.dtype(), ") but got ", constant_values.dtype());

    fill_constant_pad(result, self, pad_width_, constant_values);
  }
  return result;
}

Tensor& pad_out(
  const Tensor& self,
  const Tensor& pad_width,
  c10::string_view mode_str,
  const c10::optional<Tensor>& constant_values_opt,
  Tensor& result
) {
  return pad_out_impl(self, pad_width, mode_str, constant_values_opt, result);
}

Tensor pad(
  const Tensor& self,
  const Tensor& pad_width,
  c10::string_view mode_str,
  const c10::optional<Tensor>& constant_values_opt
) {
  Tensor result;
  return pad_out_impl(self, pad_width, mode_str, constant_values_opt, result);
}

}} // namespace at::native
