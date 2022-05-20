#pragma once

#ifdef USE_VULKAN_API

namespace at {
namespace native {
namespace vulkan {
namespace api {
namespace utils {

//
// Alignment
//

template<typename Type>
inline constexpr Type align_down(
    const Type number,
    const Type multiple) {
  return (number / multiple) * multiple;
}

template<typename Type>
inline constexpr Type align_up(
    const Type number,
    const Type multiple) {
  return align_down(number + multiple - 1, multiple);
}

template<typename Type>
inline constexpr Type div_up(
    const Type numerator,
    const Type denominator) {
  return (numerator + denominator - 1) / denominator;
}

//
// Cast
//

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

//
// Vector
//

namespace detail {

template<typename Type, uint32_t N>
struct vec final {
  Type data[N];
};

} // namespace detail

template<uint32_t N>
using ivec = detail::vec<int32_t, N>;
using ivec2 = ivec<2u>;
using ivec3 = ivec<3u>;
using ivec4 = ivec<4u>;

template<uint32_t N>
using uvec = detail::vec<uint32_t, N>;
using uvec2 = uvec<2u>;
using uvec3 = uvec<3u>;
using uvec4 = uvec<4u>;

template<uint32_t N>
using vec = detail::vec<float, N>;
using vec2 = vec<2u>;
using vec3 = vec<3u>;
using vec4 = vec<4u>;

} // namespace utils
} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
