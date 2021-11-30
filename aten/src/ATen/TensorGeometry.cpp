#include <ATen/TensorGeometry.h>

#include <limits>
#include <cstddef>

namespace at {

// See TensorGeometry.h on why this is useful now that we cache is_contiguous.
bool geometry_is_contiguous(IntArrayRef sizes, IntArrayRef strides) {
  assert(sizes.size() < std::size_t{std::numeric_limits<std::int64_t>::max()});
  auto dim = static_cast<std::int64_t>(sizes.size());
  int64_t expected_stride = 1;
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

bool TensorGeometry::is_contiguous() const {
  if (numel_ == 0) {
    return true;
  }
  return at::geometry_is_contiguous(sizes_, strides_);
}

} // namespace at
