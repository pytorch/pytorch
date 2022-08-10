#include <ATen/cuda/CUDAContext.h>
#include <type_traits>

#include <ATen/ATen.h>
#include <ATen/NestedTensorImpl.h>

#include <ATen/EmptyTensor.h>
#include <ATen/native/nested/NestedTensorMath.h>
#include <c10/core/Allocator.h>
#include <c10/core/DeviceType.h>

namespace at {
namespace native {

Tensor empty_like_nested_cuda(
    const Tensor& self,
    c10::optional<ScalarType> dtype,
    c10::optional<Layout> layout,
    c10::optional<Device> device,
    c10::optional<bool> pin_memory,
    c10::optional<c10::MemoryFormat> optional_memory_format) {
  TORCH_CHECK(
      device == self.device(),
      "Currently nested tensors doesn't support creating an empty_like tensor on a different device.")
  auto options = verify_empty_parameters(
      self, dtype, layout, device, pin_memory, optional_memory_format);

  auto* allocator = at::cuda::getCUDADeviceAllocator();
  auto self_impl = get_nested_tensor_impl(self);
  auto nested_size = self_impl->get_nested_size_tensor().clone();
  auto nested_strides = self_impl->get_nested_stride_tensor().clone();
  auto offsets = self_impl->get_offsets();

  return empty_nested_generic(
      self_impl->get_buffer_size(),
      allocator,
      options,
      self.key_set(),
      nested_size,
      nested_strides,
      offsets);
}

} // namespace native
} // namespace at
