#include <ATen/MemoryOverlap.h>
#include <c10/core/Layout.h>

namespace at {

MemOverlap has_internal_overlap(const Tensor& tensor) {
  return has_internal_overlap(tensor.unsafeGetTensorImpl());
}

/*
"Overlapping" indices are two+ valid indices that specify
the same offset within the tensor.
The function does this by checking for a sufficient but not
necessary condition of no overlap. In particular, that
that there exists an ordering of the tensor's dimensions
that is nicely "nested," with each dimension contained
within the next one.

Due to the implementation, the function returns
1. MemOverlap::NO if there is no possibility of "overlapping" indices;
2. MemOverlap::YES if there is "overlapping" detected;
3. MemOverlap::TOO_HARD for cases where there's no easy way to determine.
*/
MemOverlap has_internal_overlap(TensorImpl* t) {
  AT_ASSERT(t->layout() == kStrided);

  // obviously contiguous tensor would not have overlapping indices
  if (t->is_contiguous()) {
    return MemOverlap::NO;
  }

  std::vector<std::int64_t> sizes, strides;

  int dims = t->dim();
  sizes.reserve(dims);
  strides.reserve(dims);

  // Step (0)~(1) for the algorithm in NOTE [ Detecting Memory Overlap Within A Strided Tensor ]
  for (int i = 0; i < dims; ++i) {
    size_t size = t->sizes()[i];
    if (size == 0) {
    } else if (size > 1) {
      size_t stride = t->strides()[i];
      if (stride < 1) {
        return MemOverlap::YES;
      }
      sizes.emplace_back(size);
      strides.emplace_back(stride);
    } // size == 1 is ignored here, see 
  }
 
  // Step (2)~(4) for the algorithm in NOTE [ Detecting Memory Overlap Within A Strided Tensor ]
  if (maybe_overlapping_memory(sizes, strides)) {
    return MemOverlap::TOO_HARD;
  } else {
    return MemOverlap::NO;
  }
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
