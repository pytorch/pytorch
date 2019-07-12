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
  if (strides.end() != std::find_if(
        strides.begin(), strides.end(), [](int64_t s) { return s == 0; })) {
    return MemOverlap::YES;
  }

  return MemOverlap::TOO_HARD;
}

void assert_no_internal_overlap(const Tensor& t, const std::string& op) {
  assert_no_internal_overlap(t.unsafeGetTensorImpl(), op);
}

void assert_no_internal_overlap(TensorImpl* t, const std::string& op) {
  if (has_internal_overlap(t) == MemOverlap::YES) {
    AT_ERROR(
        op, ": unsupported operation: more than one element of the written-to "
        "tensor refers to a single memory location. Please clone() the tensor "
        "before calling ", op);
  }
}

}
