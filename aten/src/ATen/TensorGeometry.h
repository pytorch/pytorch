#pragma once

#include <ATen/Type.h>
#include <ATen/WrapDimUtils.h>

namespace at {

struct AT_API TensorGeometry {
  TensorGeometry() : storage_offset_(0) {}

  explicit TensorGeometry(IntList sizes)
    : sizes_(sizes)
    , strides_(sizes.size())
    , storage_offset_(0) {
      int64_t dim = sizes.size();
      int64_t expected_stride = 1;
      for (int64_t i = dim - 1; i >= 0; i--) {
        strides_[i] = expected_stride;
        expected_stride *= sizes_[i];
      }
  }

  explicit TensorGeometry(const Tensor& t)
    : sizes_(t.sizes())
    , strides_(t.strides())
    , storage_offset_(t.storage_offset()) {}

  // true if the tensor is contiguous
  bool is_contiguous() const;

  // creates a new tensor with the sizes and strides of the source
  Tensor zeros_with_stride(const Type& type) const;

  int64_t dim() const { return sizes_.size(); }
  int64_t size(int64_t dim) const {
    dim = maybe_wrap_dim(dim, this->dim());
    return sizes_.at(static_cast<size_t>(dim));
  }
  IntList sizes() const { return IntList{ sizes_ }; }
  int64_t stride(int64_t dim) const {
    dim = maybe_wrap_dim(dim, this->dim());
    return strides_.at(static_cast<size_t>(dim));
  }
  IntList strides() const { return IntList{ strides_ }; }
  int64_t storage_offset() const { return storage_offset_; }
  int64_t numel() const {
    int64_t r = 1;
    for (auto s : sizes()) {
      r *= s;
    }
    return r;
  }

  TensorGeometry transpose(int64_t dim0, int64_t dim1) {
    TensorGeometry r = *this; // copy
    AT_ASSERT(dim0 < dim(), "transpose: dim0=%d out of range (dim=%d)", dim0, dim())
    AT_ASSERT(dim1 < dim(), "transpose: dim1=%d out of range (dim=%d)", dim1, dim())
    std::swap(r.sizes_[dim0], r.sizes_[dim1]);
    std::swap(r.strides_[dim0], r.strides_[dim1]);
    return r;
  }

  std::vector<int64_t> sizes_;
  std::vector<int64_t> strides_;
  int64_t storage_offset_;
};

} // namespace at
