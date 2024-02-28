#include <c10/util/Exception.h>
#include <utility>

namespace at {

/*
[collapse dims] Updates sizes, and strides to reflect a "collapse" of
the info, possibly excluding the optional excludeDim. A "collapsed" version
of the info is the fewest dims that order the tensor's elements in the same
way as the original info. If excludeDim is specified, the collapse is the
fewest dims that order the tensor's elements as the original and preserve the
excluded dimension, unless the tensor collapses to a point.

This function returns a pair of values.

1) The (new) index of the preserved dimension if excludeDim is
specified. 0 if the tensor is collapsed to a point. -1
otherwise.

2) The new number of dimensions.
*/
template <typename T>
inline std::pair<int64_t, int64_t> collapse_dims(
    T* sizes,
    T* strides,
    int64_t dims,
    const int excludeDim = -1) {
  TORCH_CHECK(
      excludeDim >= -1 && excludeDim < dims,
      "expected excluded dim between -1 and dims - 1");

  int64_t stopDim = (excludeDim == -1) ? dims : excludeDim;
  int64_t newIndex = -1;
  int64_t oldIndex = 0;
  int64_t remappedExcludedDim = -1;

  while (oldIndex < dims) {
    // Finds a dimension to collapse into
    for (; oldIndex < stopDim; ++oldIndex) {
      if (sizes[oldIndex] == 1) {
        continue;
      }

      ++newIndex;
      sizes[newIndex] = sizes[oldIndex];
      strides[newIndex] = strides[oldIndex];
      ++oldIndex;
      break;
    }

    // Collapses dims
    for (; oldIndex < stopDim; ++oldIndex) {
      if (sizes[oldIndex] == 1) {
        continue;
      }

      if (strides[newIndex] == sizes[oldIndex] * strides[oldIndex]) {
        sizes[newIndex] *= sizes[oldIndex];
        strides[newIndex] = strides[oldIndex];
      } else {
        ++newIndex;
        sizes[newIndex] = sizes[oldIndex];
        strides[newIndex] = strides[oldIndex];
      }
    }

    // Handles excludeDim being set (oldIndex == excludeDim)
    if (oldIndex != dims) {
      // Preserves excluded dimension
      ++newIndex;
      sizes[newIndex] = sizes[oldIndex];
      strides[newIndex] = strides[oldIndex];
      remappedExcludedDim = newIndex;

      // Restarts iteration after excludeDim
      ++oldIndex;
      stopDim = dims;
    }
  }

  // Handles special case of all dims size 1
  if (newIndex == -1 || (newIndex == 0 && sizes[0] == 1)) {
    dims = 1;
    sizes[0] = 1;
    strides[0] = 1;

    return std::pair<int64_t, int64_t>(0, 1);
  }

  dims = newIndex + 1;
  return std::pair<int64_t, int64_t>(remappedExcludedDim, dims);
}

} // namespace at
