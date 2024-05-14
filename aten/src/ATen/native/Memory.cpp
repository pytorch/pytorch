#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/Context.h>
#include <c10/core/Storage.h>
#include <ATen/EmptyTensor.h>
#include <ATen/NestedTensorImpl.h>
#include <ATen/native/nested/NestedTensorUtils.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_debug_has_internal_overlap_native.h>
#include <ATen/ops/is_pinned_native.h>
#include <ATen/ops/pin_memory_native.h>
#include <ATen/ops/empty_cpu_dispatch.h>
#endif

namespace at::native {

// Exposes at::has_internal_overlap as an operator for testing purposes
int64_t _debug_has_internal_overlap(const Tensor& self) {
  return static_cast<int64_t>(at::has_internal_overlap(self));
}

bool is_pinned(const Tensor& self) {
  // Only CPU tensors can be pinned
  if (!self.is_cpu()) {
    return false;
  }
  // Use getAcceleratorHooksInterface to make is_pinned device-agnostic
  return at::globalContext().isPinnedPtr(self.storage().data());
}

Tensor pin_memory(const Tensor& self) {
  if (self.is_pinned()) {
    return self;
  }
  TORCH_CHECK(self.device().is_cpu(), "cannot pin '", self.toString(), "' only dense CPU tensors can be pinned");
  if (self.is_nested()) {
    auto* nt_input = get_nested_tensor_impl(self);
    const auto& input_buffer = nt_input->get_unsafe_storage_as_tensor();
    return wrap_buffer(
        input_buffer.pin_memory(),
        nt_input->get_nested_sizes(),
        nt_input->get_nested_strides(),
        nt_input->get_storage_offsets());
  }
  // Use getAcceleratorHooksInterface to make pin_memory device-agnostic
  auto* allocator = at::globalContext().getPinnedMemoryAllocator();
  auto storage = Storage(
      Storage::use_byte_size_t(),
      detail::computeStorageNbytes(
          self.sizes(), self.strides(), self.dtype().itemsize()),
      allocator,
      /*resizable=*/false);
  auto tensor = at::cpu::empty({0}, self.options()).set_(storage, 0, self.sizes(), self.strides());
  tensor.copy_(self);
  return tensor;
}

} // namespace at::native
