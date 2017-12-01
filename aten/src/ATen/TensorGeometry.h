#pragma once

#include <ATen/Type.h>

namespace at {

struct TensorGeometry {
  TensorGeometry() : storage_offset_(0) {}

  explicit TensorGeometry(const Tensor& t)
    : sizes_(t.sizes())
    , strides_(t.strides())
    , storage_offset_(t.storage_offset()) {}

  // true if the tensor is contiguous
  bool is_contiguous() const;

  // creates a new tensor with the sizes and strides of the source
  Tensor zeros_with_stride(const Type& type) const;

  int64_t dim() const { return sizes_.size(); }
  int64_t size(int64_t dim) const { return sizes_.at(static_cast<size_t>(dim)); }
  IntList sizes() const { return IntList{ sizes_ }; }
  IntList strides() const { return IntList{ strides_ }; }
  int64_t storage_offset() const { return storage_offset_; }
  int64_t numel() const {
    int64_t r = 1;
    for (auto s : sizes()) {
      r *= s;
    }
    return r;
  }

  std::vector<int64_t> sizes_;
  std::vector<int64_t> strides_;
  int64_t storage_offset_;
};

} // namespace at
