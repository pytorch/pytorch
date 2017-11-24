#pragma once

#include <ATen/ATen.h>

namespace torch {

struct TensorGeometry {
  TensorGeometry() : storage_offset(0) {}

  explicit TensorGeometry(const at::Tensor& t)
    : sizes(t.sizes())
    , strides(t.strides())
    , storage_offset(t.storage_offset()) {}

  // true if the tensor is contiguous
  bool is_contiguous() const;

  // creates a new tensor with the sizes and strides of the source
  at::Tensor zeros_with_stride(const at::Type& type) const;

  std::vector<int64_t> sizes;
  std::vector<int64_t> strides;
  int64_t storage_offset;
};

} // namespace torch
