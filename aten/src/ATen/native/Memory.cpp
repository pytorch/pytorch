#include <ATen/ATen.h>
#include <ATen/MemoryOverlap.h>

namespace at {
namespace native {

// Exposes at::has_internal_overlap as an operator for testing purposes
int64_t _debug_has_internal_overlap(const Tensor& self) {
  return static_cast<int64_t>(at::has_internal_overlap(self));
}

}
}
