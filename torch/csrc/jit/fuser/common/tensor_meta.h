#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h> // TORCH_API
#include <c10/core/Device.h>
#include <c10/core/DeviceType.h>
#include <c10/util/Exception.h>

#include <torch/csrc/jit/fuser/common/utils.h>

#include <vector>
#include <cstdint>

namespace torch {
namespace jit {
namespace fuser {

struct TensorMeta {

TensorMeta(
  const c10::DeviceType _device_type)
: device_type_{_device_type} { }

TensorMeta(
  const c10::DeviceType _device_type
, std::vector<int64_t>&& _sizes
, std::vector<int64_t>&& _strides)
: device_type_{_device_type}
, sizes_{_sizes}
, strides_{_strides} { }

TensorMeta(
  const std::shared_ptr<c10::TensorType>& tensor
, const RankType expand_to = 0) {
  TORCH_CHECK(tensor->isComplete(), "Trying to create TensorMeta from incomplete tensor!");

  device_type_ = getDeviceType(tensor);

  sizes_ = extractSizes(tensor);
  strides_ = extractStrides(tensor);

  while (expand_to > sizes_.size()) {
    sizes_.insert(sizes_.begin(), 1);
    strides_.insert(strides_.begin(), 0);
  }
}

// Getters
RankType rank() const { return sizes_.size(); }

std::vector<int64_t>& sizes() { return sizes_; }
const std::vector<int64_t>& sizes() const { return sizes_; }

std::vector<int64_t>& strides() { return strides_; }
const std::vector<int64_t>& strides() const { return strides_; }

// Removes the specified dimension
void removeDim(const RankType dim) {
  TORCH_CHECK(dim < rank(), "Trying to remove dim greater than rank!");

  sizes_.erase(sizes_.begin() + dim);
  strides_.erase(strides_.begin() + dim);
}

bool canCollapse(const RankType dim) {
  TORCH_CHECK(dim < (rank() - 1), "Checking whether a dim greater than (rank - 1) is collapsible!");

  // Dimensions of size 1 are always collapisble and can always be collapsed into
  if (sizes_[dim] == 1 || sizes_[dim + 1] == 1) {
    return true;
  }

  // If neither dimension of size 1, then the outer dimension is collapsible
  // into the inner dimension if the dims are contiguous
  if (strides_[dim] == (sizes_[dim + 1] * strides_[dim + 1])) {
    return true;
  }

  return false;
}

// Merges the dim specified by dim with dim + 1
void collapse(const RankType dim) {
  TORCH_CHECK(dim < rank(), "Trying to collapse dim greater than rank!");

  if (sizes_[dim] == 1) {
    removeDim(dim);
    return;
  }

  TORCH_CHECK(canCollapse(dim), "Request to collapse non-collapsible dim!");

  sizes_[dim + 1] *= sizes_[dim];
  removeDim(dim);
}

c10::DeviceType device_type_;
std::vector<int64_t> sizes_;
std::vector<int64_t> strides_;

};

}}} // namespace torch::jit::fuser
