#include <ATen/TensorGeometry.h>
#include <ATen/TensorUtils.h>

#include <ATen/ATen.h>

namespace at {

bool TensorGeometry::is_contiguous() const {
  if (numel_ == 0) {
    return true;
  }
  return at::geometry_is_contiguous(sizes_, strides_);
}

} // namespace at
