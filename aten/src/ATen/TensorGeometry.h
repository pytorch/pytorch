#pragma once

#include <ATen/core/TensorBase.h>
#include <c10/core/WrapDimMinimal.h>

namespace at {

// Return if the tensor geometry represented by `sizes` and `strides` is
// contiguous Although we cache is_contiguous in tensor now, this is till useful
// because it allows checking if a particular geometry is contiguous without
// explicitly constructing a tensor, e.g., when you want to choose a kernel
// strategy based on whether a subgeometry is contiguous.
TORCH_API bool geometry_is_contiguous(IntArrayRef sizes, IntArrayRef strides);

struct TORCH_API TensorGeometry {
  TensorGeometry() : storage_offset_(0) {}

  explicit TensorGeometry(c10::SymIntArrayRef sizes)
      : sizes_(sizes.vec()), strides_(sizes.size()), storage_offset_(0) {
    int64_t dim = sizes.size();
    c10::SymInt expected_stride = 1;
    for (int64_t i = dim - 1; i >= 0; i--) {
      strides_[i] = expected_stride;
      expected_stride *= sizes_[i];
    }
    numel_ = expected_stride;
  }

  explicit TensorGeometry(const TensorBase& t)
      : sizes_(t.sym_sizes().vec()),
        strides_(t.sym_strides().vec()),
        storage_offset_(t.sym_storage_offset()),
        numel_(t.sym_numel()) {}

  // true if the tensor is contiguous
  bool is_contiguous() const;

  int64_t dim() const {
    return sizes_.size();
  }
  c10::SymInt size(int64_t dim) const {
    dim = c10::maybe_wrap_dim(dim, this->dim());
    return sizes_.at(dim);
  }
  c10::SymIntArrayRef sizes() const {
    return c10::SymIntArrayRef{sizes_};
  }
  c10::SymInt stride(int64_t dim) const {
    dim = c10::maybe_wrap_dim(dim, this->dim());
    return strides_.at(dim);
  }
  c10::SymIntArrayRef strides() const {
    return c10::SymIntArrayRef{strides_};
  }
  c10::SymInt storage_offset() const {
    return storage_offset_;
  }
  c10::SymInt numel() const {
    return numel_;
  }

  TensorGeometry transpose(int64_t dim0, int64_t dim1) {
    TensorGeometry r = *this; // copy
    TORCH_CHECK(
        dim0 < dim(),
        "transpose: dim0=",
        dim0,
        " out of range (dim=",
        dim(),
        ")")
    TORCH_CHECK(
        dim1 < dim(),
        "transpose: dim1=",
        dim1,
        " out of range (dim=",
        dim(),
        ")")
    std::swap(r.sizes_[dim0], r.sizes_[dim1]);
    std::swap(r.strides_[dim0], r.strides_[dim1]);
    return r;
  }

  std::vector<c10::SymInt> sizes_;
  std::vector<c10::SymInt> strides_;
  c10::SymInt storage_offset_;
  c10::SymInt numel_;
};

} // namespace at
