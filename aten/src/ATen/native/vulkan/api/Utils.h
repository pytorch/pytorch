#pragma once

#ifdef USE_VULKAN_API

namespace at {
namespace native {
namespace vulkan {
namespace api {
namespace utils {

inline int64_t align_down(
    const int64_t number,
    const int64_t multiple) {
  return (number / multiple) * multiple;
}

inline int64_t align_up(
    const int64_t number,
    const int64_t multiple) {
  return align_down(number + multiple - 1, multiple);
}

inline int64_t div_up(
    const int64_t numerator,
    const int64_t denominator) {
  return (numerator + denominator - 1) / denominator;
}

inline VkFormat convert(const caffe2::TypeMeta dtype) {
  switch (c10::typeMetaToScalarType(dtype)) {
    case kFloat:
#ifdef VULKAN_FP16_INFERENCE
      return VK_FORMAT_R16G16B16A16_SFLOAT;
#else
      return VK_FORMAT_R32G32B32A32_SFLOAT;
#endif /* VULKAN_FP16_INFERENCE */

    default:
      TORCH_CHECK(
        false,
        "Vulkan tensor format not supported!");
  }

  return VK_FORMAT_UNDEFINED;
}

namespace detail {

template <typename To, typename From>
inline constexpr To safe_downcast(const From v) {
  typedef std::common_type_t<From, To> Type;
  constexpr Type min{static_cast<Type>(std::numeric_limits<To>::lowest())};
  constexpr Type max{static_cast<Type>(std::numeric_limits<To>::max())};
  TORCH_CHECK(min <= v && v <= max, "Cast failed: out of range!");
  return static_cast<To>(v);
}

template <typename To, typename From>
inline constexpr bool is_signed_to_unsigned() {
  return std::is_signed<From>::value && std::is_unsigned<To>::value;
}

} // namespace detail

template <
    typename To,
    typename From,
    std::enable_if_t<detail::is_signed_to_unsigned<To, From>(), bool> = true>
inline constexpr To safe_downcast(const From v) {
  TORCH_CHECK(v >= From{}, "Cast failed: negative signed to unsigned!");
  return detail::safe_downcast<To, From>(v);
}

template <
    typename To,
    typename From,
    std::enable_if_t<!detail::is_signed_to_unsigned<To, From>(), bool> = true>
inline constexpr To safe_downcast(const From v) {
  return detail::safe_downcast<To, From>(v);
}

} // namespace utils
} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
