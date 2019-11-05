#pragma once

#include <c10/util/C++17.h>
#include <c10/util/ConstexprCrc.h>
#include <c10/util/IdWrapper.h>
#include <cinttypes>
#include <functional>

namespace c10 {
namespace util {

struct type_index final : IdWrapper<type_index, uint64_t> {
  constexpr explicit type_index(uint64_t checksum) : IdWrapper(checksum) {}

  // Allow usage in std::map / std::set
  // TODO Disallow this and rather use std::unordered_map/set everywhere
  friend constexpr bool operator<(type_index lhs, type_index rhs) noexcept {
    return lhs.underlyingId() < rhs.underlyingId();
  }

  friend std::ostream& operator<<(std::ostream& stream, type_index typeId) {
    return stream << typeId.underlyingId();
  }
};

namespace detail {

#if !defined(__clang__) && !defined(_MSC_VER) && defined(__GNUC__) && \
    __GNUC__ < 5
// Getting __PRETTY_FUNCTION__ at compile time only works with GCC >= 5
#error "You're running a too old version of GCC. We need GCC 5 or later."
#endif

template <typename T>
inline C10_HOST_CONSTEXPR uint64_t type_index_impl() noexcept {
// Idea: __PRETTY_FUNCTION__ (or __FUNCSIG__ on msvc) contains a qualified name
// of this function, including its template parameter, i.e. including the
// type we want an id for. We use this name and run crc64 on it to get a type
// id.
#if defined(_MSC_VER)
  return crc64(__FUNCSIG__, sizeof(__FUNCSIG__)).checksum();
#else
  return crc64(__PRETTY_FUNCTION__, sizeof(__PRETTY_FUNCTION__)).checksum();
#endif
}

} // namespace detail

template <typename T>
inline C10_HOST_CONSTEXPR type_index get_type_index() noexcept {
#if !defined(__CUDA_ARCH__)
  // To enforce that this is really computed at compile time, we pass the crc
  // checksum through std::integral_constant.
  return type_index{std::integral_constant<
      uint64_t,
      detail::type_index_impl<guts::remove_cv_t<guts::decay_t<T>>>()>::value};
#else
  // nvcc unfortunately doesn't like this being constexpr in device code
  return type_index{
      detail::type_index_impl<guts::remove_cv_t<guts::decay_t<T>>>()};
#endif
}

} // namespace util
} // namespace c10

C10_DEFINE_HASH_FOR_IDWRAPPER(c10::util::type_index);
