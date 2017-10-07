#pragma once

#include "ATen/TensorImpl.h"
#include "ATen/WrapDim.h"
#include <sstream>

namespace at {

static inline int64_t maybe_wrap_dim(int64_t dim, TensorImpl *tensor, int64_t to_add) {
  return maybe_wrap_dim(dim, tensor->dim() + to_add);
}

static inline int64_t maybe_wrap_dim(int64_t dim, TensorList tensors, int64_t to_add) {
  if (tensors.size() == 0) {
    // can't wrap empty TensorList; rely on underlying implementation to throw error if necessary.
    return dim;
  }
  return maybe_wrap_dim(dim, tensors[0].dim() + to_add);
}

}
