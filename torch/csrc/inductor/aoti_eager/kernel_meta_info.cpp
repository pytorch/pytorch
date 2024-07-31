#if !defined(C10_MOBILE) && !defined(ANDROID)
#include <torch/csrc/inductor/aoti_eager/kernel_meta_info.h>
#include <iostream>
#include <utility>

namespace torch::inductor {

TensorMetadata::TensorMetadata(const at::Tensor& src_tensor)
    : is_symbolic_(false),
      dtype_(src_tensor.scalar_type()),
      device_(src_tensor.device()),
      dispatch_key_set_(src_tensor.key_set()),
      requires_grad_(src_tensor.requires_grad()) {
  auto sizes = src_tensor.sizes().vec();
  auto strides = src_tensor.strides().vec();
  std::transform(
      sizes.begin(), sizes.end(), std::back_inserter(sizes_), [](int64_t size) {
        return c10::SymInt(size);
      });
  std::transform(
      strides.begin(),
      strides.end(),
      std::back_inserter(strides_),
      [](int64_t stride) { return c10::SymInt(stride); });
}

TensorMetadata::TensorMetadata(
    bool is_symbolic,
    c10::ScalarType dtype,
    c10::Device device,
    c10::DispatchKeySet dispatch_key_set,
    std::vector<int64_t> sizes,
    std::vector<int64_t> strides,
    std::vector<int64_t> dim_order,
    bool requires_grad)
    : is_symbolic_(is_symbolic),
      dtype_(dtype),
      device_(device),
      dispatch_key_set_(dispatch_key_set),
      dim_order_(dim_order),
      requires_grad_(requires_grad) {
  std::transform(
      sizes.begin(), sizes.end(), std::back_inserter(sizes_), [](int64_t size) {
        return c10::SymInt(size);
      });
  std::transform(
      strides.begin(),
      strides.end(),
      std::back_inserter(strides_),
      [](int64_t stride) { return c10::SymInt(stride); });
}

TensorMetadata::TensorMetadata(
    bool is_symbolic,
    c10::ScalarType dtype,
    c10::Device device,
    c10::DispatchKeySet dispatch_key_set,
    std::vector<std::optional<c10::SymInt>> sizes,
    std::vector<std::optional<c10::SymInt>> strides,
    std::vector<int64_t> dim_order,
    bool requires_grad)
    : is_symbolic_(is_symbolic),
      dtype_(dtype),
      device_(device),
      dispatch_key_set_(dispatch_key_set),
      sizes_(std::move(sizes)),
      strides_(std::move(strides)),
      dim_order_(std::move(dim_order)),
      requires_grad_(requires_grad) {}

void TensorMetadata::build_guard(const torch::dynamo::LocalState& local_state) {
  tensor_check_ = torch::dynamo::TensorCheck(
      local_state,
      nullptr,
      dispatch_key_set_,
      dtype_,
      device_.index(),
      requires_grad_,
      sizes_,
      strides_);
}

bool TensorMetadata::operator==(const TensorMetadata& other) const {
  TORCH_INTERNAL_ASSERT(tensor_check_.has_value());
  torch::dynamo::LocalState local_state;
  local_state.overrideDispatchKeySet(dispatch_key_set_);

  if (sizes_.size() != other.sizes_.size() ||
      strides_.size() != other.strides_.size()) {
    return false;
  }

  auto _sym_sizes = std::vector<c10::SymInt>();
  auto _sym_strides = std::vector<c10::SymInt>();
  for (auto sym_size_val : other.sizes_) {
    TORCH_INTERNAL_ASSERT(sym_size_val.has_value());
    _sym_sizes.push_back(sym_size_val.value());
  }
  for (auto sym_stride_val : other.strides_) {
    TORCH_INTERNAL_ASSERT(sym_stride_val.has_value());
    _sym_strides.push_back(sym_stride_val.value());
  }

  auto sym_sizes = c10::SymIntArrayRef(
      reinterpret_cast<const c10::SymInt*>(_sym_sizes.data()),
      _sym_sizes.size());
  auto sym_strides = c10::SymIntArrayRef(
      reinterpret_cast<const c10::SymInt*>(_sym_strides.data()),
      _sym_strides.size());

  auto _tensor_check = tensor_check_.value();
  auto res = _tensor_check.check(
      local_state,
      other.dispatch_key_set_,
      other.dtype_,
      other.device_,
      sym_sizes,
      sym_strides,
      other.requires_grad_ /* Should we need to care about grad requirement?*/);
  if (!res)
    return res;

  if (is_symbolic_) {
    TORCH_INTERNAL_ASSERT(dim_order_.has_value());
    auto dim_order = dim_order_.value();
    if (dim_order.empty()) {
      return res;
    }

    auto cur_dim_idx = dim_order[0];
    if (other.strides_[cur_dim_idx].value().expect_int() != 1) {
      return false;
    }

    // Check tensor layout
    for (size_t dim_order_idx = 1; dim_order_idx < dim_order.size();
         dim_order_idx++) {
      cur_dim_idx = dim_order[dim_order_idx];
      auto cont_dim_idx = dim_order[dim_order_idx - 1];
      auto dim_cont_size = other.sizes_[cont_dim_idx].value().expect_int();
      auto dim_cont_stride = other.strides_[cont_dim_idx].value().expect_int();
      auto dim_cur_stride = other.strides_[cur_dim_idx].value().expect_int();
      if (dim_cur_stride != dim_cont_size * dim_cont_stride)
        return false;
    }
  }

  return true;
}

std::ostream& operator<<(
    std::ostream& stream,
    const TensorMetadata& tensor_metadata) {
  stream << "is_symbolic_: " << tensor_metadata.is_symbolic_ << '\n';
  stream << "dtype_: " << tensor_metadata.dtype_ << '\n';
  stream << "device_: " << tensor_metadata.device_ << '\n';
  stream << "sizes_: ";
  for (const auto& size : tensor_metadata.sizes_) {
    stream << size.value() << " ";
  }
  stream << '\n';
  stream << "strides_: ";
  for (const auto& stride : tensor_metadata.strides_) {
    stream << stride.value() << " ";
  }

  stream << "requires_grad_: " << tensor_metadata.requires_grad_ << '\n';
  stream << "dispatch_key_set_: " << tensor_metadata.dispatch_key_set_ << '\n';
  stream << "tensor_check_: " << tensor_metadata.tensor_check_.has_value()
         << '\n';
  stream << '\n';
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
  tensor_metadata_list.reserve(tensor_list.size());
  for (const auto& tensor : tensor_list) {
    tensor_metadata_list.emplace_back(tensor);
  }
  value_ = tensor_metadata_list;
}

ParameterMetadata::ParameterMetadata(
    const c10::Scalar& scalar,
    uint64_t input_order)
    : tag_(SCALAR), value_(scalar), order_(input_order) {}

ParameterMetadata::ParameterMetadata(
    const std::string& str,
    uint64_t input_order)
    : tag_(STRING), value_(str), order_(input_order) {}

ParameterMetadata::ParameterMetadata(
    const c10::Device& device,
    uint64_t input_order)
    : tag_(DEVICE), value_(device), order_(input_order) {}

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
    case DEVICE:
      return std::get<c10::Device>(value_) ==
          std::get<c10::Device>(other.value_);
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
