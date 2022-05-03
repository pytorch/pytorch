//  Copyright © 2022 Apple Inc.

#pragma once
#include <ATen/core/TensorBase.h>

namespace at {
namespace detail {

C10_EXPORT TensorBase empty_mps(
    IntArrayRef size,
    c10::optional<ScalarType> dtype_opt,
    c10::optional<Layout> layout_opt,
    c10::optional<Device> device_opt,
    c10::optional<bool> pin_memory_opt,
    c10::optional<c10::MemoryFormat> memory_format_opt);
C10_EXPORT TensorBase empty_mps(
    IntArrayRef size, const TensorOptions &options);

C10_EXPORT TensorBase empty_strided_mps(
    IntArrayRef size,
    IntArrayRef stride,
    ScalarType dtype,
    c10::optional<Device> device_opt);

C10_EXPORT TensorBase empty_strided_mps(
    IntArrayRef size,
    IntArrayRef stride,
    const TensorOptions &options);

} // namespace detail
} // namespace at

#define AT_DISPATCH_MPS_TYPES(TYPE, NAME, ...)                                \
  [&] {                                                                       \
    const auto& the_type = TYPE;                                              \
    at::ScalarType _st = ::detail::scalar_type(the_type);                     \
    RECORD_KERNEL_FUNCTION_DTYPE(NAME, _st);                                  \
    switch (_st) {                                                            \
      AT_PRIVATE_CASE_TYPE(NAME, at::ScalarType::Float, float, __VA_ARGS__)   \
      AT_PRIVATE_CASE_TYPE(NAME, at::ScalarType::Int, int32_t, __VA_ARGS__)   \
      AT_PRIVATE_CASE_TYPE(NAME, at::ScalarType::Long, int64_t, __VA_ARGS__)  \
      AT_PRIVATE_CASE_TYPE(NAME, at::ScalarType::Short, int16_t, __VA_ARGS__) \
      AT_PRIVATE_CASE_TYPE(NAME, at::ScalarType::Half, at::Half, __VA_ARGS__) \
      default:                                                                \
        AT_ERROR(#NAME, " not implemented for '", toString(_st), "'");        \
    }                                                                         \
  }()
