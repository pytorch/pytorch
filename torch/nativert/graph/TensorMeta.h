#pragma once

#include <c10/core/Device.h>
#include <c10/util/Logging.h>

#include <c10/core/Layout.h>
#include <c10/core/MemoryFormat.h>
#include <c10/core/ScalarType.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/ArrayRef.h>

#include <torch/csrc/utils/generated_serialization_types.h>
#include <torch/nativert/executor/Placement.h>

namespace torch::nativert {

c10::ScalarType convertJsonScalarType(
    const torch::_export::ScalarType& scalarType);
c10::MemoryFormat convertJsonMemoryFormat(
    const torch::_export::MemoryFormat& memoryFormat);
c10::Layout convertJsonLayout(const torch::_export::Layout& layout);
c10::Device convertJsonDevice(const torch::_export::Device& device);

class TensorMeta {
 public:
  explicit TensorMeta(const torch::_export::TensorMeta& tensorMeta);

  c10::IntArrayRef sizes() const {
    TORCH_CHECK(!hasSymbolicShape_, "TensorMeta has symbolic shape");
    return sizes_;
  }

  c10::IntArrayRef strides() const {
    TORCH_CHECK(!hasSymbolicShape_, "TensorMeta has symbolic shape");
    return strides_;
  }

  c10::Layout layout() const {
    return layout_;
  }

  c10::ScalarType dtype() const {
    return dtype_;
  }

  bool requires_grad() const {
    return requiresGrad_;
  }

  int64_t storage_offset() const {
    return storage_offset_;
  }

  int64_t dim() const {
    return sizes_.size();
  }

  bool hasSymbolicShape() const {
    return hasSymbolicShape_;
  }

  int64_t numel() const {
    TORCH_CHECK(!hasSymbolicShape_, "TensorMeta has symbolic shape");
    return numel_;
  }

  c10::Device device() const {
    return device_;
  }

  // override device according to placement
  void setDevice(c10::Device device) {
    device_ = device;
  }

  c10::TensorOptions asTensorOptions() const {
    return c10::TensorOptions().dtype(dtype_).layout(layout_).requires_grad(
        requiresGrad_);
  }

  // override device according to placement
  void applyDevicePlacement(const Placement& placement) {
    device_ = placement.getMappedDevice(device_);
  }

  // NYI
  // c10::SymIntArrayRef sym_sizes() const {}
  // c10::SymIntArrayRef sym_strides() const {}
  // c10::SymInt sym_storage_offset() const {}
  // c10::SymInt sym_numel() const {}

 private:
  bool hasSymbolicShape_ = false;

  std::vector<int64_t> sizes_;
  std::vector<int64_t> strides_;
  int64_t storage_offset_ = 0;
  int64_t numel_ = 1;

  c10::ScalarType dtype_;
  c10::Layout layout_;
  bool requiresGrad_;

  c10::Device device_;
};

} // namespace torch::nativert
