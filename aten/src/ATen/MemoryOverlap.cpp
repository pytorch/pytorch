#include <ATen/MemoryOverlap.h>
#include <c10/core/Layout.h>

namespace at {

MemOverlap has_internal_overlap(const Tensor& tensor) {
  auto* t = tensor.unsafeGetTensorImpl();

  AT_ASSERT(tensor.layout() == kStrided);

  if (t->is_contiguous()) {
    return MemOverlap::NO;
  }

  auto strides = t->strides();
  if (std::find_if(
        strides.begin(), strides.end(), [](int s) { return s == 0; })) {
    return MemOverlap::YES;
  }

  return MemOverlap::TOO_HARD;
}

void assert_no_internal_overlap(const Tensor& t, std::string op) {
  if (has_internal_overlap(t) == MemOverlap::YES) {
    AT_ERROR(
        op, ": unsupported operation: more than one element of the written-to "
        "tensor refers to a single memory location. Please clone() the tensor "
        "before calling ", op);
  }
}

}
