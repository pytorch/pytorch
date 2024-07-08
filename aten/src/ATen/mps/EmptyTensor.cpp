//  Copyright © 2022 Apple Inc.

#include <ATen/ATen.h>
#include <ATen/Tensor.h>
#include <ATen/Utils.h>
#include <torch/library.h>
#include <ATen/mps/EmptyTensor.h>
#include <ATen/mps/MPSDevice.h>
#include <ATen/native/Resize.h>
#include <ATen/native/TensorFactories.h>
#include <ATen/native/mps/Copy.h>

#define MPS_ERROR_NOT_COMPILED "PyTorch code is not compiled with MPS enabled"
#define MPS_ERROR_RUNTIME_TOO_LOW \
  "The MPS backend is supported on MacOS 12.3+.", \
  "Current OS version can be queried using `sw_vers`"
#define MPS_ERROR_DOUBLE_NOT_SUPPORTED "Cannot convert a MPS Tensor to float64 dtype " \
  "as the MPS framework doesn't support float64. Please use float32 instead."

namespace at::detail {
TensorBase empty_mps(
    IntArrayRef size,
    std::optional<ScalarType> dtype_opt,
    std::optional<Layout> layout_opt,
    std::optional<Device> device_opt,
    std::optional<bool> pin_memory_opt,
    std::optional<c10::MemoryFormat> memory_format_opt) {
#if defined(__APPLE__)
#if __is_target_os(macOS)
  if (at::hasMPS()) {
    auto device = device_or_default(device_opt);
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(device.type() == DeviceType::MPS);

    TORCH_CHECK_NOT_IMPLEMENTED(
        layout_or_default(layout_opt) == Layout::Strided,
        "only strided tensors are supported on MPS");

    TORCH_CHECK(size.size() <= 16, "MPS supports tensors with dimensions <= 16, but got ", size.size(), ".");

    check_size_nonnegative(size);

    auto* allocator = at::mps::GetMPSAllocator();
    int64_t nelements = c10::multiply_integers(size);
    auto dtype = dtype_or_default(dtype_opt);
    TORCH_CHECK_TYPE(dtype != ScalarType::Double, MPS_ERROR_DOUBLE_NOT_SUPPORTED);
    TORCH_CHECK_TYPE(dtype != ScalarType::BFloat16 || is_macos_13_or_newer(mps::MacOSVersion::MACOS_VER_14_0_PLUS), "MPS BFloat16 is only supported on MacOS 14 or newer");


    auto dtype_meta = scalarTypeToTypeMeta(dtype);
    int64_t size_bytes = nelements * dtype_meta.itemsize();
    auto storage_impl = c10::make_intrusive<StorageImpl>(
        c10::StorageImpl::use_byte_size_t(),
        size_bytes,
        allocator->allocate(size_bytes),
        allocator,
        /*resizeable=*/true);

    auto tensor =
        detail::make_tensor<TensorImpl>(storage_impl, DispatchKey::MPS, dtype_meta);
    // Default TensorImpl has size [0]
    if (size.size() != 1 || size[0] != 0) {
      tensor.unsafeGetTensorImpl()->set_sizes_contiguous(size);
    }

    auto memory_format = memory_format_opt.value_or(MemoryFormat::Contiguous);
    tensor.unsafeGetTensorImpl()->empty_tensor_restride(memory_format);
    // See Note [Enabling Deterministic Operations]
    if (C10_UNLIKELY(at::globalContext().deterministicAlgorithms() && at::globalContext().deterministicFillUninitializedMemory())) {
      at::native::fill_empty_deterministic_(tensor);
    }
    return tensor;
  } else {
    TORCH_CHECK(false, MPS_ERROR_RUNTIME_TOO_LOW)
  }
#else
  TORCH_CHECK(false, MPS_ERROR_NOT_COMPILED)
#endif
#else
  TORCH_CHECK(false, MPS_ERROR_NOT_COMPILED)
#endif
}

TensorBase empty_mps(
    IntArrayRef size, const TensorOptions &options) {
  return at::detail::empty_mps(
      size,
      optTypeMetaToScalarType(options.dtype_opt()),
      options.layout_opt(),
      options.device_opt(),
      options.pinned_memory_opt(),
      options.memory_format_opt());
}

TensorBase empty_strided_mps(
    IntArrayRef size,
    IntArrayRef stride,
    ScalarType dtype,
    std::optional<Device> device_opt) {
#if defined(__APPLE__)
#if __is_target_os(macOS)
  if (at::hasMPS()) {
    auto device = device_or_default(device_opt);
    TORCH_INTERNAL_ASSERT(device.is_mps());
    TORCH_CHECK_TYPE(dtype != ScalarType::Double, MPS_ERROR_DOUBLE_NOT_SUPPORTED);
    const DeviceGuard device_guard(device);
    auto* allocator = at::mps::GetMPSAllocator();
    constexpr c10::DispatchKeySet mps_dks(c10::DispatchKey::MPS);
    Tensor result = at::detail::empty_strided_generic(
        size, stride, allocator, mps_dks, dtype);
    // See Note [Enabling Deterministic Operations]
    if (C10_UNLIKELY(at::globalContext().deterministicAlgorithms() && at::globalContext().deterministicFillUninitializedMemory())) {
      at::native::fill_empty_deterministic_(result);
    }
    return result;
  } else {
    TORCH_CHECK(false, MPS_ERROR_RUNTIME_TOO_LOW)
  }
#else
  TORCH_CHECK(false, MPS_ERROR_NOT_COMPILED)
#endif
#else
  TORCH_CHECK(false, MPS_ERROR_NOT_COMPILED)
#endif
}

TensorBase empty_strided_mps(
    IntArrayRef size,
    IntArrayRef stride,
    const TensorOptions &options) {
  return at::native::empty_strided_mps(
      size,
      stride,
      optTypeMetaToScalarType(options.dtype_opt()),
      options.layout_opt(),
      options.device_opt(),
      options.pinned_memory_opt());
}

} // namespace at::detail
