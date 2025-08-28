#pragma once

#include <ATen/core/IListRef.h>
#include <ATen/core/Tensor.h>
#include <c10/core/TensorImpl.h>
#include <c10/core/WrapDimMinimal.h>
#include <c10/util/irange.h>

namespace at {

// if dim_post_expr is 0 and wrap_scalar is true, then dim must be in the
// range [-1, 0]. This is a special case for scalar tensors and manifests in
// e.g. torch.sum(scalar_tensor, 0) Otherwise, dim should be in the range
// [-dim_post_expr, dim_post_expr-1].
using c10::maybe_wrap_dim;

inline int64_t maybe_wrap_dim(int64_t dim, TensorImpl* tensor) {
  return maybe_wrap_dim(dim, tensor->dim());
}

inline int64_t maybe_wrap_dim(int64_t dim, TensorList tensors) {
  if (tensors.empty()) {
    // can't wrap empty TensorList; rely on underlying implementation to throw
    // error if necessary.
    return dim;
  }
  return maybe_wrap_dim(dim, tensors[0].dim());
}

inline int64_t maybe_wrap_dim(
    int64_t dim,
    const std::vector<std::vector<int64_t>>& tensor_sizes) {
  if (tensor_sizes.empty()) {
    // can't wrap empty list; rely on underlying implementation to throw error
    // if necessary
    return dim;
  }
  return maybe_wrap_dim(dim, static_cast<int64_t>(tensor_sizes[0].size()));
}

// Given an array of dimensions `dims` of length `ndims`, this function "Wraps"
// each dim in-place for a tensor of rank `dim_post_expr`, allowing dims to be
// specified using negative indices.
//
// Additionally, if `wrap_scalar` is true then scalar tensors with rank 0, will
// allow dimensions in the range [-1, 0]. Otherwise, an IndexError is raised for
// dimensions not in the range [-dim_post_expr, dim_post_expr).
inline void maybe_wrap_dims_n(
    int64_t* dims,
    int64_t ndims,
    int64_t dim_post_expr,
    bool wrap_scalars = true) {
  if (dim_post_expr <= 0) {
    if (wrap_scalars) {
      dim_post_expr = 1; // this will make range [-1, 0]
    } else {
      TORCH_CHECK_INDEX(
          ndims == 0,
          "Dimension specified as ",
          dims[0],
          " but tensor has no dimensions");
      return;
    }
  }
  int64_t min = -dim_post_expr;
  int64_t max = dim_post_expr - 1;
  for (const auto i : c10::irange(ndims)) {
    auto& dim = dims[i];
    if (dim < min || dim > max) {
      TORCH_CHECK_INDEX(
          false,
          "Dimension out of range (expected to be in range of [",
          min,
          ", ",
          max,
          "], but got ",
          dim,
          ")");
    }
    if (dim < 0)
      dim += dim_post_expr;
  }
}

// Given a contiguous container of dimensions `dims`, this function "Wraps"
// each dim in-place for a tensor of rank `dim_post_expr`, allowing dims to be
// specified using negative indices.
//
// Additionally, if `wrap_scalar` is true then scalar tensors with rank 0, will
// allow dimensions in the range [-1, 0]. Otherwise, an IndexError is raised for
// dimensions not in the range [-dim_post_expr, dim_post_expr).
template <typename Container>
inline void maybe_wrap_dims(
    Container& dims,
    int64_t dim_post_expr,
    bool wrap_scalars = true) {
  return maybe_wrap_dims_n(
      dims.data(), dims.size(), dim_post_expr, wrap_scalars);
}

// previously, size [0] tensors were the only possible empty tensors; thus, it
// wasn't possible to cat empty tensors unless all the other tensors were
// 1-dimensional, so we allowed these tensors to be "skipped" (both for wrap
// dimension behavior and dimension size checking). We maintain this behavior
// for backwards compatibility, but only for this specific size (i.e. other
// empty sizes are not skipped).
inline int64_t legacy_cat_wrap_dim(
    int64_t dim,
    const std::vector<std::vector<int64_t>>& tensor_sizes) {
  for (auto& sizes : tensor_sizes) {
    if (sizes.size() == 1 && sizes[0] == 0) {
      continue;
    }
    return maybe_wrap_dim(dim, static_cast<int64_t>(sizes.size()));
  }
  return dim;
}

inline int64_t legacy_cat_wrap_dim_symint(
    int64_t dim,
    const std::vector<std::vector<c10::SymInt>>& tensor_sizes) {
  for (auto& sizes : tensor_sizes) {
    if (sizes.size() == 1) {
      if (TORCH_GUARD_OR_FALSE(sizes[0].sym_eq(0))) {
        continue;
      }
    }
    return maybe_wrap_dim(dim, static_cast<int64_t>(sizes.size()));
  }
  return dim;
}

inline int64_t legacy_cat_wrap_dim(
    int64_t dim,
    const MaterializedITensorListRef& tensors) {
  for (const Tensor& tensor : tensors) {
    if (tensor.dim() == 1) {
      if (TORCH_GUARD_OR_FALSE(tensor.sym_sizes()[0].sym_eq(0))) {
        continue;
      }
    }
    return maybe_wrap_dim(dim, tensor.dim());
  }
  return dim;
}

// wrap negative dims in a vector
inline void wrap_all_dims(
    std::vector<int64_t>& dims_to_wrap,
    int64_t tensor_total_dims) {
  for (const auto i : c10::irange(dims_to_wrap.size())) {
    dims_to_wrap[i] = maybe_wrap_dim(dims_to_wrap[i], tensor_total_dims);
  }
}

} // namespace at
