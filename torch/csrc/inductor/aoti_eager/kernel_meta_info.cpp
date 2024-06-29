#if !defined(C10_MOBILE) && !defined(ANDROID)
#include <torch/csrc/inductor/aoti_eager/kernel_meta_info.h>
#include <iostream>

namespace torch::inductor {

TensorMetadata::TensorMetadata(const at::Tensor& src_tensor)
    : is_symbolic_(false),
      dtype_(src_tensor.scalar_type()),
      device_(src_tensor.device()),
      dispatch_key_set_(src_tensor.key_set()),
      sizes_(src_tensor.sizes().vec()),
      strides_(src_tensor.strides().vec()),
      requires_grad_(src_tensor.requires_grad()) {}

TensorMetadata::TensorMetadata(
    bool is_symbolic,
    c10::ScalarType dtype,
    c10::Device device,
    c10::DispatchKeySet dispatch_key_set,
    std::vector<int64_t> sizes,
    std::vector<int64_t> strides,
    bool requires_grad)
    : is_symbolic_(is_symbolic),
      dtype_(dtype),
      device_(device),
      dispatch_key_set_(dispatch_key_set),
      sizes_(sizes),
      strides_(strides),
      requires_grad_(requires_grad) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      !is_symbolic_, "Not support symbolic shape now");
}

void TensorMetadata::build_guard(const torch::dynamo::LocalState& local_state) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      !is_symbolic_, "Not support symbolic shape now");
  std::vector<std::optional<c10::SymInt>> sym_sizes;
  std::vector<std::optional<c10::SymInt>> sym_strides;
  std::transform(
      sizes_.begin(),
      sizes_.end(),
      std::back_inserter(sym_sizes),
      [](int64_t size) { return std::optional<c10::SymInt>(size); });
  std::transform(
      strides_.begin(),
      strides_.end(),
      std::back_inserter(sym_strides),
      [](int64_t stride) { return std::optional<c10::SymInt>(stride); });
  tensor_check_ = torch::dynamo::TensorCheck(
      local_state,
      nullptr,
      dispatch_key_set_,
      dtype_,
      device_.index(),
      requires_grad_,
      sym_sizes,
      sym_strides);
}

bool TensorMetadata::operator==(const TensorMetadata& other) const {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      !is_symbolic_, "Not support symbolic shape now");

  if (tensor_check_.has_value()) {
    auto sizes_ = c10::IntArrayRef(other.sizes_);
    auto strides_ = c10::IntArrayRef(other.strides_);
    auto sym_sizes = c10::SymIntArrayRef(
        reinterpret_cast<const c10::SymInt*>(sizes_.data()), sizes_.size());
    auto sym_strides = c10::SymIntArrayRef(
        reinterpret_cast<const c10::SymInt*>(strides_.data()), strides_.size());

    torch::dynamo::LocalState local_state;
    local_state.overrideDispatchKeySet(dispatch_key_set_);
    auto _tensor_check = tensor_check_.value();
    auto res = _tensor_check.check(
        local_state,
        other.dispatch_key_set_,
        other.dtype_,
        other.device_,
        sym_sizes,
        sym_strides,
        other.requires_grad_ /* Should we need to care about grad requirement?*/);
    return res;
  } else {
    return this->is_symbolic_ == other.is_symbolic_ &&
        this->dtype_ == other.dtype_ && this->device_ == other.device_ &&
        this->dispatch_key_set_ == other.dispatch_key_set_ &&
        this->requires_grad_ == other.requires_grad_ &&
        this->sizes_ == other.sizes_ && this->strides_ == other.strides_;
  }
}

std::ostream& operator<<(
    std::ostream& stream,
    const TensorMetadata& tensor_metadata) {
  stream << "is_symbolic_: " << tensor_metadata.is_symbolic_ << std::endl;
  stream << "dtype_: " << tensor_metadata.dtype_ << std::endl;
  stream << "device_: " << tensor_metadata.device_ << std::endl;
  stream << "sizes_: ";
  for (const auto& size : tensor_metadata.sizes_) {
    stream << size << " ";
  }
  stream << std::endl;
  stream << "strides_: ";
  for (const auto& stride : tensor_metadata.strides_) {
    stream << stride << " ";
  }

  stream << "requires_grad_: " << tensor_metadata.requires_grad_ << std::endl;
  stream << "dispatch_key_set_: " << tensor_metadata.dispatch_key_set_
         << std::endl;
  stream << "tensor_check_: " << tensor_metadata.tensor_check_.has_value()
         << std::endl;
  stream << std::endl;
  return stream;
}

ParameterMetadata::ParameterMetadata(
    TensorMetadata tensor_metadata,
    uint64_t input_order)
    : tag_(TENSOR), value_(tensor_metadata), order_(input_order) {}

ParameterMetadata::ParameterMetadata(
    const at::Tensor& tensor,
    uint64_t input_order)
    : tag_(TENSOR), order_(input_order) {
  value_ = TensorMetadata(tensor);
}

ParameterMetadata::ParameterMetadata(
    const std::vector<TensorMetadata>& tensor_metadata_list,
    uint64_t input_order)
    : tag_(TENSOR_LIST), value_(tensor_metadata_list), order_(input_order) {}

ParameterMetadata::ParameterMetadata(
    const std::vector<at::Tensor>& tensor_list,
    uint64_t input_order)
    : tag_(TENSOR_LIST), order_(input_order) {
  std::vector<TensorMetadata> tensor_metadata_list;
  for (const auto& tensor : tensor_list) {
    tensor_metadata_list.push_back(TensorMetadata(tensor));
  }
  value_ = tensor_metadata_list;
}

ParameterMetadata::ParameterMetadata(
    const c10::Scalar& scalar,
    uint64_t input_order)
    : tag_(SCALAR), order_(input_order) {
  value_ = scalar;
}

ParameterMetadata::ParameterMetadata(
    const std::string& str,
    uint64_t input_order)
    : tag_(STRING), order_(input_order) {
  value_ = str;
}

bool ParameterMetadata::operator==(const ParameterMetadata& other) const {
  // Same type
  if (tag_ != other.tag_) {
    return false;
  }

  // Same order of the input parameters
  if (order_ != other.order_) {
    return false;
  }

  switch (tag_) {
    case TENSOR:
      return std::get<TensorMetadata>(value_) ==
          std::get<TensorMetadata>(other.value_);
    case TENSOR_LIST:
      return std::get<std::vector<TensorMetadata>>(value_) ==
          std::get<std::vector<TensorMetadata>>(other.value_);
    case SCALAR:
      TORCH_INTERNAL_ASSERT(
          std::get<c10::Scalar>(other.value_).isFloatingPoint() ||
          std::get<c10::Scalar>(other.value_).isIntegral(true /*includeBool*/));
      return equal_to(std::get<c10::Scalar>(other.value_));
    case STRING:
      return std::get<std::string>(value_) ==
          std::get<std::string>(other.value_);
    default:
      return false;
  }
}

bool ParameterMetadata::equal_to(const c10::Scalar& scalar) const {
  TORCH_INTERNAL_ASSERT(scalar.isFloatingPoint() || scalar.isIntegral(true));
  if (tag_ != SCALAR) {
    return false;
  }

  auto self_scalar = std::get<c10::Scalar>(value_);
  if (scalar.isFloatingPoint() && self_scalar.isFloatingPoint()) {
    return self_scalar.toDouble() == scalar.toDouble();
  } else if (scalar.isIntegral(true) && self_scalar.isIntegral(true)) {
    return self_scalar.toInt() == scalar.toInt();
  }

  return false;
}

} // namespace torch::inductor
#endif
