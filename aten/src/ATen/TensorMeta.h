#pragma once

#include <ATen/DimVector.h>
#include <c10/core/TensorOptions.h>
#include <ATen/core/Dimname.h>

namespace at {

class Tensor;

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

CAFFE2_API Tensor meta_tensor_from_meta(const TensorMeta& meta);
CAFFE2_API Tensor tensor_from_meta(const TensorMeta& meta);
// Analogous to self.new_empty(sizes)
CAFFE2_API TensorMeta new_meta(const Tensor& self, IntArrayRef sizes);

} // namespace at
