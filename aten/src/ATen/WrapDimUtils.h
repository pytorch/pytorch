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

}
