#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/cuda/EmptyTensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/EmptyTensor.h>

namespace at {
namespace detail {

TensorBase empty_cuda(
    IntArrayRef size,
    ScalarType dtype,
    c10::optional<Device> device_opt,
    c10::optional<c10::MemoryFormat> memory_format_opt) {
  at::globalContext().lazyInitCUDA();
  const auto device = device_or_default(device_opt);
  TORCH_INTERNAL_ASSERT(device.is_cuda());
  const DeviceGuard device_guard(device);
  auto* allocator = at::cuda::getCUDADeviceAllocator();
  constexpr c10::DispatchKeySet cuda_dks(c10::DispatchKey::CUDA);
  return at::detail::empty_generic(
      size, allocator, cuda_dks, dtype, memory_format_opt);
}

TensorBase empty_cuda(
    IntArrayRef size,
    c10::optional<ScalarType> dtype_opt,
    c10::optional<Layout> layout_opt,
    c10::optional<Device> device_opt,
    c10::optional<bool> pin_memory_opt,
    c10::optional<c10::MemoryFormat> memory_format_opt) {
  TORCH_CHECK(!pin_memory_opt.has_value() || !*pin_memory_opt, "Only dense CPU tensors can be pinned");
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(layout_or_default(layout_opt) == Layout::Strided);

  const auto dtype = dtype_or_default(dtype_opt);
  return at::detail::empty_cuda(size, dtype, device_opt, memory_format_opt);
}

TensorBase empty_cuda(
    IntArrayRef size, const TensorOptions &options) {
  return at::detail::empty_cuda(
      size,
      optTypeMetaToScalarType(options.dtype_opt()),
      options.layout_opt(),
      options.device_opt(),
      options.pinned_memory_opt(),
      options.memory_format_opt());
}

TensorBase empty_strided_cuda(
    IntArrayRef size,
    IntArrayRef stride,
    ScalarType dtype,
    c10::optional<Device> device_opt) {
  at::globalContext().lazyInitCUDA();
  const auto device = device_or_default(device_opt);
  TORCH_INTERNAL_ASSERT(device.is_cuda());
  const DeviceGuard device_guard(device);
  auto* allocator = at::cuda::getCUDADeviceAllocator();
  constexpr c10::DispatchKeySet cuda_dks(c10::DispatchKey::CUDA);
  return at::detail::empty_strided_generic(
      size, stride, allocator, cuda_dks, dtype);
}

TensorBase empty_strided_cuda(
    IntArrayRef size,
    IntArrayRef stride,
    c10::optional<ScalarType> dtype_opt,
    c10::optional<Layout> layout_opt,
    c10::optional<Device> device_opt,
    c10::optional<bool> pin_memory_opt) {
  TORCH_CHECK(!pin_memory_opt.has_value() || !*pin_memory_opt, "Only dense CPU tensors can be pinned");
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(layout_or_default(layout_opt) == Layout::Strided);

  const auto dtype = dtype_or_default(dtype_opt);
  return at::detail::empty_strided_cuda(size, stride, dtype, device_opt);
}

TensorBase empty_strided_cuda(
    IntArrayRef size,
    IntArrayRef stride,
    const TensorOptions &options) {
  return at::detail::empty_strided_cuda(
      size,
      stride,
      optTypeMetaToScalarType(options.dtype_opt()),
      options.layout_opt(),
      options.device_opt(),
      options.pinned_memory_opt());
}

}}  // namespace at::detail
