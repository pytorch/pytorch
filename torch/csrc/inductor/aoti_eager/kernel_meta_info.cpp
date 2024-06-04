#if !defined(C10_MOBILE) && !defined(ANDROID)
#include <torch/csrc/inductor/aoti_eager/kernel_meta_info.h>
#include <iostream>

namespace torch::inductor {

TensorMetadata::TensorMetadata(const at::Tensor& src_tensor)
    : is_symbolic_(false),
      device_(src_tensor.device()),
      sizes_(src_tensor.sizes().vec()),
      strides_(src_tensor.sizes().vec()) {}

TensorMetadata::TensorMetadata(
    bool is_symbolic,
    c10::ScalarType dtype,
    c10::Device device,
    std::vector<int64_t> sizes,
    std::vector<int64_t> strides)
    : is_symbolic_(is_symbolic),
      dtype_(dtype),
      scalar_value_((float)1.0),
      device_(device),
      sizes_(sizes),
      strides_(strides) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      !is_symbolic_, "Not support symbolic shape now");
}

TensorMetadata::TensorMetadata(
    bool is_symbolic,
    c10::ScalarType dtype,
    c10::IValue scalar_value,
    c10::Device device,
    std::vector<int64_t> sizes,
    std::vector<int64_t> strides)
    : is_symbolic_(is_symbolic),
      dtype_(dtype),
      scalar_value_(scalar_value),
      device_(device),
      sizes_(sizes),
      strides_(strides) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      !is_symbolic_, "Not support symbolic shape now");
}

bool TensorMetadata::operator==(const TensorMetadata& other) const {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      !is_symbolic_, "Not support symbolic shape now");
  return this->is_symbolic_ == other.is_symbolic_ &&
      this->dtype_ == other.dtype_ &&
      this->scalar_value_ == other.scalar_value_ &&
      this->device_.type() == other.device_.type() &&
      this->sizes_ == other.sizes_ && this->strides_ == other.strides_;
}

std::ostream& operator<<(
    std::ostream& stream,
    const TensorMetadata& tensor_metadata) {
  stream << "is_symbolic_: " << tensor_metadata.is_symbolic_ << std::endl;
  stream << "dtype_: " << tensor_metadata.dtype_ << std::endl;
  stream << "scalar_value_: " << tensor_metadata.scalar_value_.type()->str()
         << "(" << tensor_metadata.scalar_value_ << ")" << std::endl;
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
  stream << std::endl;
  return stream;
}

size_t TensorMetadataHash::operator()(
    const TensorMetadata& tensor_metadata) const {
  auto hash = std::hash<bool>()(tensor_metadata.is_symbolic_);
  hash = c10::hash_combine(
      hash, std::hash<c10::ScalarType>()(tensor_metadata.dtype_));
  hash =
      c10::hash_combine(hash, c10::IValue::hash(tensor_metadata.scalar_value_));
  hash = c10::hash_combine(
      hash, std::hash<c10::DeviceType>()(tensor_metadata.device_.type()));

  for (auto& e : tensor_metadata.sizes_) {
    hash = c10::hash_combine(hash, std::hash<int64_t>()(e));
  }

  for (auto& e : tensor_metadata.strides_) {
    hash = c10::hash_combine(hash, std::hash<int64_t>()(e));
  }
  return hash;
}

size_t AOTIKernelMetadataHash::operator()(
    const AOTIKernelMetadata& aoti_kernel_metadata) const {
  size_t hash = 0;
  for (auto& e : aoti_kernel_metadata) {
    hash = c10::hash_combine(hash, TensorMetadataHash()(e));
  }
  return hash;
}

} // namespace torch::inductor
#endif
