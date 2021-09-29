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

template<typename T>
void check_arg_is_none(const c10::optional<T>& arg, const char* arg_name, c10::string_view mode_str) {
  TORCH_CHECK(!arg.has_value(),
    "torch.pad: Unsupported keyword argument for '", std::string(mode_str),
    "' mode: ", arg_name);
}

// Create an IntArrayRef from a tensor
IntArrayRef tensor_to_arrayref(const Tensor& size_tensor) {
  return IntArrayRef(size_tensor.data_ptr<int64_t>(), size_tensor.numel());
}

// Check that a tensor is on the expected device
void check_device(const at::Tensor& arg, const char* arg_name, at::Device self_device) {
  TORCH_CHECK(arg.device() == self_device,
    "torch.pad: Expected '", arg_name, "' to be on the same device as ",
    "'input' (", self_device, ") but got ", arg.device());
}

Tensor& pad_out_impl(
  const Tensor& self,
  const Tensor& pad_width,
  c10::string_view mode_str,
  const c10::optional<Scalar>& constant_values_opt,
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
    check_arg_is_none(constant_values_opt, "constant_values", mode_str);
  }

  // If `out` is given, it must match `self`'s dtype and device
  if (result.defined()) {
    TORCH_CHECK(result.scalar_type() == self.scalar_type(),
      "torch.pad: Expected 'out' dtype ", self.scalar_type(),
      " but got ", result.scalar_type());
    check_device(result, "out", self.device());
  }

  Tensor pad_width_ = at::native::expand_pad_specifier(pad_width, "pad_width", self.dim());

  if (mode == PadMode::Constant) {
    Scalar constant_values = constant_values_opt.value_or(
      c10::Scalar(0));

    c10::ScalarType scalar_type = self.scalar_type();

    if (c10::isIntegralType(scalar_type, true)) {
      TORCH_CHECK(constant_values.isIntegral(true),
        "torch.pad: Expected 'constant_values' to be of integer type");
    } else if (c10::isFloatingType(scalar_type)) {
      TORCH_CHECK(!constant_values.isComplex(),
        "torch.pad: Expected 'constant_values' to be of floating point",
        " or integer type");
    }

    Tensor pad_width_flipped = at::flip(pad_width_, {0});
    Tensor result_ = at::constant_pad_nd(
      self,
      tensor_to_arrayref(pad_width_flipped),
      constant_values);

    if (result.defined()) {
      at::native::resize_output(result, result_.sizes());
      result.copy_(result_);

    } else {
      result = result_;
    }
  }
  return result;
}

Tensor& pad_out(
  const Tensor& self,
  const Tensor& pad_width,
  c10::string_view mode_str,
  const c10::optional<Scalar>& constant_values_opt,
  Tensor& result
) {
  return pad_out_impl(self, pad_width, mode_str, constant_values_opt, result);
}

Tensor pad(
  const Tensor& self,
  const Tensor& pad_width,
  c10::string_view mode_str,
  const c10::optional<Scalar>& constant_values_opt
) {
  Tensor result;
  return pad_out_impl(self, pad_width, mode_str, constant_values_opt, result);
}

}} // namespace at::native
