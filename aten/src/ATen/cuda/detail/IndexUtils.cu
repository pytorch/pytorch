#include "IndexUtils.cuh"

namespace at {
namespace cuda {
namespace detail {

struct SizeAndStride {
  int64_t size;
  int64_t stride;
};

int compareSizeAndStride(const void* a, const void* b) {
  const SizeAndStride* aS = (const SizeAndStride*) a;
  const SizeAndStride* bS = (const SizeAndStride*) b;

  return aS->stride < bS->stride;
}

bool overlappingIndices(const Tensor& t) {
  // In this function, we don't care about permutations of the
  // size/stride arrays (transpositions).
  // We order the size/stride arrays by stride, skipping dimensions
  // of size 1. Strides of dimensions of size 1 don't matter, since
  // there is only one addressing point in them.
  // In this reordered view, the tensor is contiguous if
  // stride[dim] == size[dim + 1] * stride[dim + 1] for all `dim`.
  // The tensor has holes if
  // stride[dim] > size[dim + 1] * stride[dim + 1] for one or more
  // `dim`.
  // The tensor has overlaps if
  // stride[dim] < size[dim + 1] * stride[dim + 1] for one or more
  // `dim`, or the innermost stride is 0.

  /* Extract size/stride arrays; only consider size >1 dims. */
  SizeAndStride *info = (SizeAndStride *)alloca(sizeof(SizeAndStride) * t.dim());
  int dims = t.dim();
  int nonSize1Dims = 0;
  for (int i = 0; i < dims; ++i) {
    int64_t size = t.size(i);
    if (size > 1) {
      info[nonSize1Dims].size = size;
      info[nonSize1Dims].stride = t.stride(i);
      ++nonSize1Dims;
    }
  }

  if (nonSize1Dims == 0) {
    /* no overlap */
    return false;
  }

  /* Ascending order (innermost dimension in sorted view is at [0]) */
  qsort(info, nonSize1Dims, sizeof(SizeAndStride), compareSizeAndStride);

  /* Base case: innermost dimension must have stride >= 1 */
  if (info[nonSize1Dims - 1].stride < 1) {
    return true;
  }

  /* Subsequent dimensions, if any */
  for (int i = nonSize1Dims - 2; i >= 0; --i) {
    if (info[i].stride < info[i + 1].size * info[i + 1].stride) {
      /* There are overlaps */
      return true;
    }
  }

  /* Tensor has holes or is contiguous */
  return false;
}

bool canUse32BitIndexMath(const Tensor& t, int64_t max_elem) {
  int64_t elements = t.numel();
  if (elements >= max_elem) {
    return false;
  }

  int64_t offset = 0;
  int64_t linearId = elements - 1;

  for (int i = t.dim() - 1; i >= 0; --i) {
    int64_t curDimIndex = linearId % t.size(i);
    int64_t curDimOffset = curDimIndex * t.stride(i);
    offset += curDimOffset;
    linearId /= t.size(i);
  }

  if (offset >= max_elem) {
    return false;
  }

  return true;
}

} // detail
} // cuda
} // at
