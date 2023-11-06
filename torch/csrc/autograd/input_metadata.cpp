#include <torch/csrc/autograd/input_metadata.h>

// TODO: we may be able to move some imports from input_metadata.h to here, but
// it seems that function.h transitively depends on some of them.

namespace torch {
namespace autograd {

namespace {

MetadataShape compute_variant_shape(const at::Tensor& input) {
  if (input.is_nested() &&
    !input.unsafeGetTensorImpl()->is_python_dispatch()) {
    auto nested_size = input._nested_tensor_size();
    return MetadataShape{std::in_place_type<at::Tensor>, nested_size};
  }
  return MetadataShape{std::in_place_type<SymIntSmallVec>, input.sym_sizes()};
}

bool is_python_dispatch(const at::Tensor& tensor) {
  return tensor.unsafeGetTensorImpl()->is_python_dispatch();
}

bool is_cpp_nested_tensor(const at::Tensor& tensor) {
  return tensor.is_nested() && !is_python_dispatch(tensor);
}

} // namespace

InputMetadata::InputMetadata(
    const at::TensorOptions& options,
    MetadataShape input_shape,
    bool is_tensor_subclass,
    bool is_nested)
    : options_{options},
      shape_{std::move(input_shape)},
      is_tensor_subclass_{is_tensor_subclass},
      is_nested_{is_nested},
      was_default_constructed_{false} {
  auto device_ = options.device();
  stream_ = c10::impl::getDeviceGuardImpl(device_.type())->getStream(device_);
}

InputMetadata::InputMetadata(const at::Tensor& t)
    : InputMetadata(
          t.options(),
          compute_variant_shape(t),
          is_python_dispatch(t),
          t.is_nested()) {}

at::Tensor InputMetadata::zeros_like() const {
  TORCH_CHECK(
      !is_nested_, "Zeros is not currently supported for nested tensors.")
  return at::zeros_symint(shape_as_dim_vector(), options_);
}

bool InputMetadata::is_same_shape(const at::Tensor& grad) const {
  if (!is_nestedness_same(grad)) {
    return false;
  }
  if (is_cpp_nested_tensor()) {
    return grad._nested_tensor_size().is_same_size(shape_as_tensor());
  }
  return grad.sym_sizes().equals(shape_as_dim_vector());
}

bool InputMetadata::is_expandable_to_shape(const at::Tensor& grad) const {
  if (!maybe_expandable_to(grad)) {
    return false;
  }
  return at::is_expandable_to(shape_as_dim_vector(), grad.sym_sizes());
}

at::Tensor InputMetadata::reduce_grad(at::Tensor& grad) const {
  // reduce_grad should only be called if is_expandable_to_shape returns true.
  TORCH_INTERNAL_ASSERT(maybe_expandable_to(grad));
  return at::sum_to(std::move(grad), shape_as_dim_vector());
}

std::stringstream InputMetadata::incompatible_shape_error_message(
    const size_t index,
    const at::Tensor& grad) const {
  std::stringstream ss;
  ss << "invalid gradient at index " << index << " - got ";
  if (::torch::autograd::is_cpp_nested_tensor(grad)) {
    ss << grad._nested_tensor_size();
  } else {
    ss << grad.sym_sizes();
  }
  ss << " but expected shape compatible with ";
  if (is_cpp_nested_tensor()) {
    ss << shape_as_tensor();
  } else {
    ss << shape_as_dim_vector();
  }
  return ss;
}

bool InputMetadata::is_cpp_nested_tensor() const {
  bool ret = std::holds_alternative<at::Tensor>(shape_);
  TORCH_INTERNAL_ASSERT(ret == (is_nested_ && !is_tensor_subclass_))
  return ret;
}

c10::SymIntArrayRef InputMetadata::shape_as_dim_vector() const {
  const auto& dim_shape = std::get<SymIntSmallVec>(shape_);
  return c10::SymIntArrayRef(dim_shape.data(), dim_shape.size());
}

// Danger: not thread safe, caller must protect with lock
SymIntSmallVec& InputMetadata::mutable_shape_as_dim_vector() {
  return std::get<SymIntSmallVec>(shape_);
}

bool InputMetadata::is_nestedness_same(const at::Tensor& grad) const {
  return (
    grad.is_nested() == is_nested_ &&
    ::torch::autograd::is_cpp_nested_tensor(grad) == is_cpp_nested_tensor()
  );
}

at::Tensor InputMetadata::shape_as_tensor() const {
  return std::get<at::Tensor>(shape_);
}

bool InputMetadata::maybe_expandable_to(const at::Tensor& grad) const {
  // This is the initial step to determine whether or not the tensor represented by
  // input_metadata is expandable to grad based on is-nestedness information alone.
  // If this function returns true, then is_expandable_to_shape will be called.
  // We support the following 3 types of expansion:
  bool grad_is_nested = grad.is_nested();
  if (!is_nested_ && !grad_is_nested) {
    // Normal case (no NestedTensors are involved)
    // (1) plain Tensor -> plain Tensor
    return true;
  } else {
    // (2) python NT -> python NT
    // (3) plain Tensor -> python NT
    return (
      grad_is_nested && is_python_dispatch(grad) && (!is_nested_ || is_tensor_subclass_)
    );
  }
}

} // namespace autograd
} // namespace torch
