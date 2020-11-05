#pragma once

#ifdef USE_VULKAN_API

#include <ATen/ATen.h>
#include <ATen/native/vulkan/api/api.h>
#include <ATen/native/vulkan/ops/Tensor.h>

namespace at {
namespace native {
namespace vulkan {

template <typename To, typename From>
inline constexpr To safe_downcast_internal(const From v) {
  typedef std::common_type_t<From, To> Type;
  constexpr Type min{static_cast<Type>(std::numeric_limits<To>::lowest())};
  constexpr Type max{static_cast<Type>(std::numeric_limits<To>::max())};
  TORCH_CHECK(min <= v && v <= max, "Cast failed: out of range");
  return static_cast<To>(v);
}

template <typename To, typename From>
inline constexpr bool is_signed_to_unsigned() {
  return std::is_signed<From>::value && std::is_unsigned<To>::value;
}

template <
    typename To,
    typename From,
    std::enable_if_t<is_signed_to_unsigned<To, From>(), bool> = true>
inline constexpr To safe_downcast(const From v) {
  TORCH_CHECK(v >= From{}, "Cast failed: negative signed to unsigned");
  return safe_downcast_internal<To, From>(v);
}

template <
    typename To,
    typename From,
    std::enable_if_t<!is_signed_to_unsigned<To, From>(), bool> = true>
inline constexpr To safe_downcast(const From v) {
  return safe_downcast_internal<To, From>(v);
}

} // namespace vulkan
} // namespace native
} // namespace at
#endif /* USE_VULKAN_API */
