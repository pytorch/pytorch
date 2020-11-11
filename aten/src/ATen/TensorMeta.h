#pragma once

#include <ATen/ATen.h>  // TODO: improve
// #include <ATen/NativeFunctions.h>

namespace at {

struct TensorMeta {
  DimVector sizes;
  // TODO: DimVector strides;
  TensorOptions options;
};

inline Tensor tensor_from_meta(const TensorMeta& meta) {
  // TODO: eliminate indirection
  return at::empty(meta.sizes, meta.options);
}

// Analogous to self.new_empty(sizes)
inline TensorMeta new_meta(const Tensor& self, IntArrayRef sizes) {
  TensorMeta m;
  m.sizes = sizes;
  m.options = self.options();
  return m;
}

} // namespace at
