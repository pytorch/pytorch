#include <ATen/Context.h>
#include <ATen/EmptyTensor.h>
#include <ATen/native/mtia/EmptyTensor.h>
#include <c10/core/Allocator.h>
#include <c10/core/DeviceGuard.h>

namespace at::detail {

at::Allocator* GetMTIAAllocator() {
  return GetAllocator(DeviceType::MTIA);
}

TensorBase empty_mtia(
    IntArrayRef size,
    ScalarType dtype,
    std::optional<Device> device_opt,
    std::optional<c10::MemoryFormat> memory_format_opt) {
  at::globalContext().lazyInitDevice(c10::DeviceType::MTIA);
  const auto device = device_or_default(device_opt);
  TORCH_INTERNAL_ASSERT(device.is_mtia());
  const DeviceGuard device_guard(device);
  auto* allocator = GetMTIAAllocator();
  constexpr c10::DispatchKeySet mtia_dks(c10::DispatchKey::MTIA);
  return at::detail::empty_generic(
      size, allocator, mtia_dks, dtype, memory_format_opt);
}

TensorBase empty_mtia(
    IntArrayRef size,
    std::optional<ScalarType> dtype_opt,
    std::optional<Layout> layout_opt,
    std::optional<Device> device_opt,
    std::optional<bool> pin_memory_opt,
    std::optional<c10::MemoryFormat> memory_format_opt) {
  const auto dtype = dtype_or_default(dtype_opt);
  return at::detail::empty_mtia(size, dtype, device_opt, memory_format_opt);
}

TensorBase empty_mtia(IntArrayRef size, const TensorOptions& options) {
  return at::detail::empty_mtia(
      size,
      optTypeMetaToScalarType(options.dtype_opt()),
      options.layout_opt(),
      options.device_opt(),
      options.pinned_memory_opt(),
      options.memory_format_opt());
}

TensorBase empty_strided_mtia(
    IntArrayRef size,
    IntArrayRef stride,
    ScalarType dtype,
    std::optional<Device> device_opt) {
  at::globalContext().lazyInitDevice(c10::DeviceType::MTIA);
  const auto device = device_or_default(device_opt);
  const DeviceGuard device_guard(device);
  auto* allocator = GetMTIAAllocator();
  constexpr c10::DispatchKeySet mtia_dks(c10::DispatchKey::MTIA);
  return at::detail::empty_strided_generic(
      size, stride, allocator, mtia_dks, dtype);
}

TensorBase empty_strided_mtia(
    IntArrayRef size,
    IntArrayRef stride,
    std::optional<ScalarType> dtype_opt,
    std::optional<Layout> layout_opt,
    std::optional<Device> device_opt,
    std::optional<bool> pin_memory_opt) {
  const auto dtype = dtype_or_default(dtype_opt);
  return at::detail::empty_strided_mtia(size, stride, dtype, device_opt);
}

TensorBase empty_strided_mtia(
    IntArrayRef size,
    IntArrayRef stride,
    const TensorOptions& options) {
  return at::detail::empty_strided_mtia(
      size,
      stride,
      optTypeMetaToScalarType(options.dtype_opt()),
      options.layout_opt(),
      options.device_opt(),
      options.pinned_memory_opt());
}
} // namespace at::detail
