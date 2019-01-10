#include <ATen/TensorGeometry.h>

#include <ATen/ATen.h>

namespace at {

bool TensorGeometry::is_contiguous() const {
  int64_t dim = sizes_.size();
  int64_t expected_stride = 1;
  for (int64_t i = dim - 1; i >= 0; i--) {
    if (sizes_[i] != 1 && strides_[i] != expected_stride) {
      return false;
    }
    expected_stride *= sizes_[i];
  }
  return true;
}

Tensor TensorGeometry::zeros_with_stride(const Type& type) const {
  return type.tensor(sizes_, strides_).zero_();
}

} // namespace at
