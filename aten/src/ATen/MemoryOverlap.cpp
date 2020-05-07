#include <ATen/MemoryOverlap.h>
#include <c10/core/Layout.h>

namespace at {

MemOverlap has_internal_overlap(const Tensor& tensor) {
  return has_internal_overlap(tensor.unsafeGetTensorImpl());
}

MemOverlap has_internal_overlap(TensorImpl* t) {
  AT_ASSERT(t->layout() == kStrided);

  if (t->is_contiguous()) {
    return MemOverlap::NO;
  }

  auto strides = t->strides();
  auto sizes = t->sizes();
  for (size_t i = 0; i < strides.size(); ++i) {
    if (strides[i] == 0 && sizes[i] > 1) {
      return MemOverlap::YES;
    }
  }

  return MemOverlap::TOO_HARD;
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
  if (a == b) return MemOverlapStatus::FULL;
  if (a->numel() == 0 || b->numel() == 0) {
    return MemOverlapStatus::NO;
  }
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
