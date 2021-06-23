#include <ATen/ATen.h>
#include <ATen/MemoryOverlap.h>

namespace at {
namespace native {

// Exposes at::has_internal_overlap as an operator for testing purposes
int64_t _debug_has_internal_overlap(const Tensor& self) {
  return static_cast<int64_t>(at::has_internal_overlap(self));
}

// Technically, we could force backends to explicitly say "no, we don't support
// pinned memory, always return false", but this makes life a little easier when
// you haven't loaded the backend extension at all (which can happen, e.g., on a
// CPU build of PyTorch and you try to check if something is CUDA pinned)
bool is_pinned_default(const Tensor& self, c10::optional<Device> device) {
  return false;
}

}
}
