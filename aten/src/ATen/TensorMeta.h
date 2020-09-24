#pragma once

#include <ATen/ATen.h>  // TODO: improve
// #include <ATen/NativeFunctions.h>

namespace at {

struct TensorMeta {
  DimVector sizes;
  // TODO: DimVector strides;
  ScalarType dtype;
};

inline Tensor tensor_from_meta(const TensorMeta& meta) {
  // TODO: eliminate indirection
  return at::empty(meta.sizes, dtype(meta.dtype));
}

} // namespace at
