#pragma once

#include <ATen/DimVector.h>
#include <c10/core/TensorOptions.h>

namespace at {

class Tensor;

struct TensorMeta {
  DimVector sizes;
  // TODO: DimVector strides;
  TensorOptions options;

  TensorMeta() {}  // blegh default construction
  TensorMeta(IntArrayRef _sizes, TensorOptions _options)
    : sizes(_sizes), options(_options) {}
};

CAFFE2_API Tensor meta_tensor_from_meta(const TensorMeta& meta);

CAFFE2_API Tensor tensor_from_meta(const TensorMeta& meta);

// Analogous to self.new_empty(sizes)
CAFFE2_API TensorMeta new_meta(const Tensor& self, IntArrayRef sizes);

} // namespace at
