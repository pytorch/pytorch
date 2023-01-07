#pragma once

#include <c10/util/ArrayRef.h>
#include <c10/util/Half.h> // For c10::overflows

#include <ATen/native/vulkan/api/Common.h>

#ifdef USE_VULKAN_API

namespace at {
namespace native {
namespace vulkan {
namespace api {
namespace utils {

//
// Alignment
//

template <typename Type>
inline constexpr Type align_down(const Type number, const Type multiple) {
  return (number / multiple) * multiple;
}

template <typename Type>
inline constexpr Type align_up(const Type number, const Type multiple) {
  return align_down(number + multiple - 1, multiple);
}

template <typename Type>
inline constexpr Type div_up(const Type numerator, const Type denominator) {
  return (numerator + denominator - 1) / denominator;
}

//
// Cast
//

namespace detail {

template <typename To, typename From>
inline constexpr To safe_downcast(const From v) {
  TORCH_CHECK(!c10::overflows<To>(v), "Cast failed: out of range!");
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

//
// Vector
//

namespace detail {

template <typename Type, uint32_t N>
struct vec final {
  Type data[N];
};

} // namespace detail

template <uint32_t N>
using ivec = detail::vec<int32_t, N>;
using ivec2 = ivec<2u>;
using ivec3 = ivec<3u>;
using ivec4 = ivec<4u>;

template <uint32_t N>
using uvec = detail::vec<uint32_t, N>;
using uvec2 = uvec<2u>;
using uvec3 = uvec<3u>;
using uvec4 = uvec<4u>;

template <uint32_t N>
using vec = detail::vec<float, N>;
using vec2 = vec<2u>;
using vec3 = vec<3u>;
using vec4 = vec<4u>;

inline ivec2 make_ivec2(IntArrayRef ints, bool reverse = false) {
  TORCH_CHECK(ints.size() == 2);
  if (reverse) {
    return {safe_downcast<int32_t>(ints[1]), safe_downcast<int32_t>(ints[0])};
  } else {
    return {safe_downcast<int32_t>(ints[0]), safe_downcast<int32_t>(ints[1])};
  }
}

inline ivec4 make_ivec4(IntArrayRef ints, bool reverse = false) {
  TORCH_CHECK(ints.size() == 4);
  if (reverse) {
    return {
        safe_downcast<int32_t>(ints[3]),
        safe_downcast<int32_t>(ints[2]),
        safe_downcast<int32_t>(ints[1]),
        safe_downcast<int32_t>(ints[0]),
    };
  } else {
    return {
        safe_downcast<int32_t>(ints[0]),
        safe_downcast<int32_t>(ints[1]),
        safe_downcast<int32_t>(ints[2]),
        safe_downcast<int32_t>(ints[3]),
    };
  }
}

inline ivec3 make_ivec3(uvec3 ints) {
  return {
      safe_downcast<int32_t>(ints.data[0u]),
      safe_downcast<int32_t>(ints.data[1u]),
      safe_downcast<int32_t>(ints.data[2u])};
}

} // namespace utils

inline bool operator==(const utils::uvec3& _1, const utils::uvec3& _2) {
  return (
      _1.data[0u] == _2.data[0u] && _1.data[1u] == _2.data[1u] &&
      _1.data[2u] == _2.data[2u]);
}

inline VkOffset3D create_offset3d(const utils::uvec3& offsets) {
  return VkOffset3D{
      static_cast<int32_t>(offsets.data[0u]),
      static_cast<int32_t>(offsets.data[1u]),
      static_cast<int32_t>(offsets.data[2u])};
}

inline VkExtent3D create_extent3d(const utils::uvec3& extents) {
  return VkExtent3D{extents.data[0u], extents.data[1u], extents.data[2u]};
}

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
