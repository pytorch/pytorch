#include <ATen/MemoryOverlap.h>
#include <c10/core/Layout.h>

namespace at {

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

MemOverlap has_internal_overlap(const Tensor& tensor) {
  return has_internal_overlap(tensor.unsafeGetTensorImpl());
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
MemOverlap has_internal_overlap(TensorImpl* t) {
  AT_ASSERT(t->layout() == kStrided);

  if (t->is_contiguous()) {
    return MemOverlap::NO;
  }

  std::vector<SizeAndStride> info(t->dim());

  int dims = t->dim();
  int nonSize1Dims = 0;

  for (int i = 0; i < dims; ++i) {
    int64_t size = t->sizes()[i];
    if (size > 1) {
      info[nonSize1Dims].size = size;
      info[nonSize1Dims].stride = t->strides()[i];

      if (info[nonSize1Dims].stride < 1) {
        return MemOverlap::YES;
      }

      ++nonSize1Dims;
    }
  }

  // Short-circuits if tensor is a single element.
  if (nonSize1Dims == 0) {
    return MemOverlap::NO;
  }

  /* Ascending order (innermost dimension in sorted view is at [0]) */
  qsort(info.data(), nonSize1Dims, sizeof(SizeAndStride), compareSizeAndStride);

  /* There's no easy solution to a linear Diophantine equation with restricted
   * coefficients
   */
  for (int i = 0; i < (nonSize1Dims - 1); ++i) {
    if (((info[i].size - 1) * info[i].stride) >= info[i + 1].stride) {
      return MemOverlap::TOO_HARD;
    }
  }

  return MemOverlap::NO;
}

void assert_no_internal_overlap(const Tensor& t) {
  assert_no_internal_overlap(t.unsafeGetTensorImpl());
}

void assert_no_internal_overlap(TensorImpl* t) {
  TORCH_CHECK(has_internal_overlap(t) != MemOverlap::YES,
    "unsupported operation: more than one element of the written-to tensor "
    "refers to a single memory location. Please clone() the tensor before "
    "performing the operation.");
}

MemOverlapStatus get_overlap_status(const Tensor& a, const Tensor& b) {
  return get_overlap_status(a.unsafeGetTensorImpl(), b.unsafeGetTensorImpl());
}

MemOverlapStatus get_overlap_status(TensorImpl* a, TensorImpl* b) {
  if (!a->is_contiguous() || !b->is_contiguous()) {
    return MemOverlapStatus::TOO_HARD;
  }
  if (a->storage().data() == b->storage().data()) {
    const auto a_begin = static_cast<char*>(a->data());
    const auto a_end = a_begin + a->numel() * a->itemsize();
    const auto b_begin = static_cast<char*>(b->data());
    const auto b_end = b_begin + b->numel() * b->itemsize();

    if (a_begin == b_begin && a_end == b_end) {
      return MemOverlapStatus::FULL;
    }
    if (a_begin < b_end && b_begin < a_end) {
      return MemOverlapStatus::PARTIAL;
    }
  }
  return MemOverlapStatus::NO;
}

void assert_no_partial_overlap(const Tensor& a, const Tensor& b) {
  assert_no_partial_overlap(a.unsafeGetTensorImpl(), b.unsafeGetTensorImpl());
}

void assert_no_partial_overlap(TensorImpl* a, TensorImpl* b) {
  TORCH_CHECK(get_overlap_status(a, b) != MemOverlapStatus::PARTIAL,
    "unsupported operation: some elements of the input tensor and "
    "the written-to tensor refer to a single memory location. "
    "Please clone() the tensor before performing the operation.");
}

}
