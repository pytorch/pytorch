#include <ATen/MemoryOverlap.h>
#include <c10/core/Layout.h>
#include <c10/core/Backend.h>

namespace at {

MemOverlap has_internal_overlap(const Tensor& tensor) {
  auto* t = tensor.unsafeGetTensorImpl();

  AT_ASSERT(layout_from_backend(tensorTypeIdToBackend(t->type_id())) == kStrided);

  if (t->is_contiguous()) {
    return MemOverlap::kNo;
  }

  auto strides = t->strides();
  if (std::find_if(
        strides.begin(), strides.end(), [](int s) { return s == 0; })) {
    return MemOverlap::kYes;
  }

  return MemOverlap::kTooHard;
}

void assert_no_internal_overlap(const Tensor& t, std::string op) {
  if (has_internal_overlap(t) == MemOverlap::kYes) {
    AT_ERROR(
        op, ": unsupported operation: more than one element of the written-to "
        "tensor refers to a single memory location. Please clone() the tensor "
        "before calling ", op);
  }
}

}
