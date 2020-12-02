#pragma once

#include <ATen/ATen.h>  // TODO: improve
// #include <ATen/NativeFunctions.h>

#include <ATen/DimVector.h>
#include <c10/core/TensorOptions.h>
#include <ATen/core/Dimname.h>

namespace at {

namespace impl {

struct MetaBase {
  virtual void set_output(int64_t output_idx, IntArrayRef sizes, IntArrayRef strides, TensorOptions options, DimnameList names) = 0;
  void set_output(IntArrayRef sizes, TensorOptions options) {
    set_output(0, sizes, {}, options, {});
  }
  virtual ~MetaBase() {}
};

} // namespace impl

struct TensorMeta {
  DimVector sizes;
  // TODO: DimVector strides;
  TensorOptions options;

  TensorMeta(IntArrayRef _sizes, TensorOptions _options)
    : sizes(_sizes), options(_options) {}
};

inline Tensor meta_tensor_from_meta(const TensorMeta& meta) {
  // TODO: eliminate indirection
  return at::empty_meta(meta.sizes, meta.options);
}

inline Tensor tensor_from_meta(const TensorMeta& meta) {
  // TODO: eliminate indirection
  return at::empty(meta.sizes, meta.options);
}

// Analogous to self.new_empty(sizes)
inline TensorMeta new_meta(const Tensor& self, IntArrayRef sizes) {
  return TensorMeta(sizes, self.options());
}

} // namespace at
