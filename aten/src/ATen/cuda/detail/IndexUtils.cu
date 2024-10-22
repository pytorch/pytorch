#include <ATen/cuda/detail/IndexUtils.cuh>
#include <vector>

namespace at {
namespace cuda {
namespace detail {

struct SizeAndStride {
  int64_t size;
  int64_t stride;
};

/*
 A comparator that will sort SizeAndStride structs by stride,
 in ascending order.
 */
 int compareSizeAndStride(const void* a, const void* b) {
  const SizeAndStride* aS = (const SizeAndStride*) a;
  const SizeAndStride* bS = (const SizeAndStride*) b;

  if (aS->stride < bS->stride) return -1;
  if (aS->stride == bS->stride) return 0;
  return 1;
}

/*
Returns false if there is no possibility that the tensor
has "overlapping" indices and true otherwise.
"Overlapping" indices are two+ valid indices that specify
the same offset within the tensor.
The function does this by checking for a sufficient but not
necessary condition of no overlap. In particular, that
that there exists an ordering of the tensor's dimensions
that is nicely "nested," with each dimension contained
within the next one.
*/
bool maybeOverlappingIndices(const TensorBase& t) {
  /* Extract size/stride arrays; only consider size >1 dims. */
  std::vector<SizeAndStride> info(t.dim());
  int dims = t.dim();
  int nonSize1Dims = 0;
  for (int i = 0; i < dims; ++i) {
    int64_t size = t.size(i);
    if (size > 1) {
      info[nonSize1Dims].size = size;
      info[nonSize1Dims].stride = t.stride(i);

      if (info[nonSize1Dims].stride < 1) {
        return true;
      }

      ++nonSize1Dims;
    }
  }

  // Short-circuits if tensor is a single element.
  if (nonSize1Dims == 0) {
    return false;
  }

  /* Ascending order (innermost dimension in sorted view is at [0]) */
  qsort(info.data(), nonSize1Dims, sizeof(SizeAndStride), compareSizeAndStride);

  for (int i = 0; i < (nonSize1Dims - 1); ++i) {
    if (((info[i].size - 1) * info[i].stride) >= info[i + 1].stride) {
      return true;
    }
  }

  return false;
}

} // detail
} // cuda
} // at
