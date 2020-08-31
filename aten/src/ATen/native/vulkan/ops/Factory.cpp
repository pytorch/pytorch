#include <ATen/native/vulkan/ops/Common.h>
#include <torch/library.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

Tensor empty_memory_format(
    const IntArrayRef sizes,
    const TensorOptions& options_,
    const optional<MemoryFormat> memory_format = c10::nullopt) {
  TORCH_CHECK(
      !(options_.has_memory_format() && memory_format.has_value()),
      "Cannot set memory_format both in TensorOptions and explicit argument!");

  const TensorOptions options = options_.merge_in(TensorOptions().memory_format(memory_format));
  verify(options);

  return at::detail::make_tensor<vTensorImpl>(
      DispatchKeySet(DispatchKey::Vulkan),
      options.dtype(),
      at::Device(at::kVulkan),
      vTensor(sizes, options),
      sizes,
      IntArrayRef{});
}

Tensor empty_strided(
    const IntArrayRef sizes,
    const IntArrayRef /* strides */,
    const optional<ScalarType> dtype,
    const optional<Layout> layout,
    const optional<Device> device,
    const optional<bool> pin_memory) {
  return empty_memory_format(
      sizes,
      TensorOptions().
          dtype(dtype).
          layout(layout).
          device(device).
          pinned_memory(pin_memory));
}

Tensor to(
    const Tensor& self,
    const c10::optional<ScalarType> dtype,
    const c10::optional<Layout> layout,
    const c10::optional<Device> device,
    const c10::optional<bool> pin_memory,
    const bool non_blocking,
    const bool copy,
    const c10::optional<MemoryFormat> memory_format) {
  const TensorOptions& from_options = self.options();
  verify(from_options);

  const TensorOptions to_options = TensorOptions().
      dtype(dtype).
      layout(layout).
      device(device).
      pinned_memory(pin_memory);
  verify(to_options);

  TORCH_INTERNAL_ASSERT(
      (kVulkan == from_options.device().type()) || (kVulkan == to_options.device().type()),
      "Incorrect dispatch!  Either the source or destination of `aten::to` must be Vulkan.");

  if ((from_options.dtype() == to_options.dtype()) &&
      (from_options.layout() == to_options.layout()) &&
      (from_options.device() == to_options.device()) &&
      (from_options.pinned_memory() == to_options.pinned_memory()) &&
      !copy) {
    return self;
  }

  // std::cout << "from " << self.options().dtype() << " to " << *dtype << std::endl;
  // std::cout << "from " << self.options().layout() << " to " << *layout << std::endl;
  // std::cout << "from " << self.options().device() << " to " << *device << std::endl;
  // std::cout << "from " << self.options().pinned_memory() << " to " << *pin_memory << std::endl;

  return empty_memory_format(std::vector<int64_t>{1, 2, 3, 3}, to_options);
}

#ifdef USE_VULKAN_API

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl_UNBOXED("empty.memory_format", at::native::vulkan::ops::empty_memory_format);
  m.impl("empty_strided", TORCH_FN(at::native::vulkan::ops::empty_strided));
  m.impl("to.dtype_layout", TORCH_FN(at::native::vulkan::ops::to));
}

#endif /* USE_VULKAN_API */

} // namespace
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
