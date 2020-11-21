#pragma once

#include <ATen/ATen.h>  // TODO: improve
// #include <ATen/NativeFunctions.h>

namespace at {

struct TensorMeta {
  DimVector sizes;
  // TODO: DimVector strides;
  TensorOptions options;

  TensorMeta(IntArrayRef _sizes, TensorOptions _options)
    : sizes(_sizes), options(_options) {}
};

inline Tensor tensor_from_meta(const TensorMeta& meta) {
  // TODO: eliminate indirection
  return at::empty(meta.sizes, meta.options);
}

// Analogous to self.new_empty(sizes)
inline TensorMeta new_meta(const Tensor& self, IntArrayRef sizes) {
  return TensorMeta(sizes, self.options());
}

} // namespace at
