#include <ATen/MemoryOverlap.h>
#include <c10/core/Layout.h>

namespace at {

MemOverlap has_internal_overlap(const Tensor& t) {
  if (t.is_contiguous()) {
    return MemOverlap::kNo;
  }

  auto strides = t.strides();
  if (std::find_if(
        strides.begin(), strides.end(), [](int s) { return s == 0; })) {
    return MemOverlap::kYes;
  }

  return MemOverlap::kTooHard;
}

}
