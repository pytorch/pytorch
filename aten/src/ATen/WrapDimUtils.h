#pragma once

#include "ATen/TensorImpl.h"
#include <sstream>

namespace at {

static inline int64_t maybe_wrap_dim(int64_t dim, int64_t dim_post_expr, bool wrap_scalar=true) {
  if (dim_post_expr <= 0) {
    if (!wrap_scalar) {
      std::ostringstream oss;
      oss << "dimension specified as " << dim << " but tensor has no dimensions";
      throw std::runtime_error(oss.str());
    }
    dim_post_expr = 1; // this will make range [-1, 0]
  }

  int64_t min = -dim_post_expr;
  int64_t max = dim_post_expr - 1;
  if (dim < min || dim > max) {
    std::ostringstream oss;
    oss << "dimension out of range (expected to be in range of [" << min
        << ", " << max << "], but got " << dim << ")",
    throw std::runtime_error(oss.str());
  }
  if (dim < 0) dim += dim_post_expr;
  return dim;
}

static inline int64_t maybe_wrap_dim(int64_t dim, TensorImpl *tensor) {
  return maybe_wrap_dim(dim, tensor->dim());
}

static inline int64_t maybe_wrap_dim(int64_t dim, TensorList tensors) {
  if (tensors.size() == 0) {
    // can't wrap empty TensorList; rely on underlying implementation to throw error if necessary.
    return dim;
  }
  return maybe_wrap_dim(dim, tensors[0].dim());
}

static inline int64_t maybe_wrap_dim(int64_t dim, const std::vector<std::vector<int64_t>> & tensor_sizes) {
  if (tensor_sizes.size() == 0) {
    // can't wrap empty list; rely on underlying implementation to throw error if necessary
    return dim;
  }
  return maybe_wrap_dim(dim, tensor_sizes[0].size());
}

// FIXME: when empty tensors with arbitrary shapes is implemented
//
// One of the invariants for `cat` should be that all tensors being
// concatenated should have the same size in all dimensions except the
// specified cat dimension. When we have empty tensors with arbitrary
// shapes, they should also follow this invariant:
// - torch.cat([tensor of size (1, 0, 4), tensor of size (1, n, 4)]) should work
// - torch.cat([tensor of size (0,), tensor of size (1, 2)]) should fail.
//
// Right now, as a workaround, we support the following legacy behavior when
// a non-empty `tensor` and empty tensor `empty` are concatenated:
// 1) torch.cat([empty, tensor]) should ignore empty tensors
// 2) torch.cat([tensor, empty]) should ignore empty tensors
// 3) torch.cat([empty, empty, ..., empty]) should be empty
// The following functions wrap dim on the first non-empty shape
// to preserve legacy behavior.
static inline int64_t legacy_cat_wrap_dim(int64_t dim, const std::vector<std::vector<int64_t>>& tensor_sizes) {
  for (auto& sizes : tensor_sizes) {
    if (sizes == std::vector<int64_t>({0})) {
      continue;
    }
    return maybe_wrap_dim(dim, sizes.size());
  }
  return dim;
}

static inline int64_t legacy_cat_wrap_dim(int64_t dim, TensorList tensors) {
  for (auto& tensor : tensors) {
    if (tensor.numel() == 0) {
      continue;
    }
    return maybe_wrap_dim(dim, tensor.dim());
  }
  return dim;
}

}
