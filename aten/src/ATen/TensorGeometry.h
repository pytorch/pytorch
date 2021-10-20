#pragma once

#include <c10/core/WrapDimMinimal.h>
#include <ATen/core/TensorBase.h>

namespace at {

// Return if the tensor geometry represented by `sizes` and `strides` is contiguous
// Although we cache is_contiguous in tensor now, this is till useful because it
// allows checking if a particular geometry is contiguous without explicitly
// constructing a tensor, e.g., when you want to choose a kernel strategy based
// on whether a subgeometry is contiguous.
TORCH_API bool geometry_is_contiguous(IntArrayRef sizes, IntArrayRef strides);

struct TORCH_API TensorGeometry {
  TensorGeometry() : storage_offset_(0) {}

  explicit TensorGeometry(IntArrayRef sizes)
    : sizes_(sizes.vec())
    , strides_(sizes.size())
    , storage_offset_(0) {
      int64_t dim = sizes.size();
      int64_t expected_stride = 1;
      for (int64_t i = dim - 1; i >= 0; i--) {
        strides_[i] = expected_stride;
        expected_stride *= sizes_[i];
      }
      numel_ = expected_stride;
  }

  explicit TensorGeometry(const TensorBase& t)
    : sizes_(t.sizes().vec())
    , strides_(t.strides().vec())
    , storage_offset_(t.storage_offset())
    , numel_(t.numel()) {}

  // true if the tensor is contiguous
  bool is_contiguous() const;

  int64_t dim() const { return sizes_.size(); }
  int64_t size(int64_t dim) const {
    dim = c10::maybe_wrap_dim(dim, this->dim());
    return sizes_.at(static_cast<size_t>(dim));
  }
  IntArrayRef sizes() const { return IntArrayRef{ sizes_ }; }
  int64_t stride(int64_t dim) const {
    dim = c10::maybe_wrap_dim(dim, this->dim());
    return strides_.at(static_cast<size_t>(dim));
  }
  IntArrayRef strides() const { return IntArrayRef{ strides_ }; }
  int64_t storage_offset() const { return storage_offset_; }
  int64_t numel() const { return numel_; }

  TensorGeometry transpose(int64_t dim0, int64_t dim1) {
    TensorGeometry r = *this; // copy
    TORCH_CHECK(dim0 < dim(), "transpose: dim0=", dim0, " out of range (dim=", dim(), ")")
    TORCH_CHECK(dim1 < dim(), "transpose: dim1=", dim1, " out of range (dim=", dim(), ")")
    std::swap(r.sizes_[dim0], r.sizes_[dim1]);
    std::swap(r.strides_[dim0], r.strides_[dim1]);
    return r;
  }

  std::vector<int64_t> sizes_;
  std::vector<int64_t> strides_;
  int64_t storage_offset_;
  int64_t numel_;
};

} // namespace at
