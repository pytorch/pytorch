#include <ATen/TensorGeometry.h>
#include <c10/util/overflows.h>

namespace at {

// See TensorGeometry.h on why this is useful now that we cache is_contiguous.
template <typename T>
bool _geometry_is_contiguous(ArrayRef<T> sizes, ArrayRef<T> strides) {
  assert(!overflows<std::int64_t>(sizes.size()));
  auto dim = static_cast<std::int64_t>(sizes.size());
  T expected_stride = 1;
  bool contig_if_nonempty = true;
  for (int64_t i = dim - 1; i >= 0; i--) {
    if (sizes[i] == 0) {
      return true;
    }
    if (contig_if_nonempty) {
      if (sizes[i] != 1 && strides[i] != expected_stride) {
        contig_if_nonempty = false;
      }
      expected_stride *= sizes[i];
    }
  }
  return contig_if_nonempty;
}

bool geometry_is_contiguous(IntArrayRef sizes, IntArrayRef strides) {
  return _geometry_is_contiguous(sizes, strides);
}

bool TensorGeometry::is_contiguous() const {
  if (numel_ == 0) {
    return true;
  }
  return at::_geometry_is_contiguous<c10::SymInt>(sizes_, strides_);
}

} // namespace at
