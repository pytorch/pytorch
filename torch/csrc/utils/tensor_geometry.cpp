#include "torch/csrc/utils/tensor_geometry.h"

using namespace at;

namespace torch {

bool TensorGeometry::is_contiguous() const {
  int64_t dim = sizes.size();
  int64_t expected_stride = 1;
  for (int64_t i = dim - 1; i >= 0; i--) {
    if (sizes[i] != 1 && strides[i] != expected_stride) {
      return false;
    }
    expected_stride *= sizes[i];
  }
  return true;
}

Tensor TensorGeometry::zeros_with_stride(const Type& type) const {
  return type.tensor(sizes, strides).zero_();
}


} // namespace torch
