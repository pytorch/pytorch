#pragma once

#include "ATen/TensorImpl.h"

namespace at {
struct SparseTensorImpl : public TensorImpl {
  explicit SparseTensorImpl(Type * type)
  : TensorImpl(type) {}
};
} // namespace at
