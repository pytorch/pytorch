#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/MemoryOverlap.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_debug_has_internal_overlap_native.h>
#include <ATen/ops/_pin_memory.h>
#include <ATen/ops/is_pinned_native.h>
#include <ATen/ops/pin_memory_native.h>
#endif

#include <ATen/native/nested/NestedTensorUtils.h>

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

Tensor pin_memory(const Tensor& self, c10::optional<Device> device) {
  // Kind of mad that I have to do two dynamic dispatches here, pretty
  // annoying
  if (self.is_pinned(device)) {
    return self;
  }
  // I'm even more annoyed at this additional check. We could change
  // _pin_memory to dispatch on dispatchkey of self since NestedTensorCUDA
  // is not a device. Or we can just assume that this additional bool field
  // check is cheap enough in comparison to the cost of pinning memory in the
  // first place.
  if (self.is_nested()) {
    auto* nt_input = get_nested_tensor_impl(self);
    TORCH_CHECK(nested_tensor_impl_is_contiguous(nt_input));
    const auto& input_buffer = nt_input->get_buffer();
    return wrap_buffer(_pin_memory(input_buffer, device), nt_input->get_nested_sizes());
  }
  return at::_pin_memory(self, device);
}

}
}
