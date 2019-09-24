#pragma once

#include <c10/util/C++17.h>
#include <c10/util/ConstexprCrc.h>
#include <c10/util/IdWrapper.h>
#include <c10/util/string_view.h>
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

inline constexpr string_view extract(
    string_view prefix,
    string_view suffix,
    string_view str) {
#if !defined(__CUDA_ARCH__) // CUDA doesn't like std::logic_error in device code
  return (!str.starts_with(prefix) || !str.ends_with(suffix))
      ? (throw std::logic_error("Invalid pattern"), string_view())
      : str.substr(prefix.size(), str.size() - prefix.size() - suffix.size());
#else
  return str.substr(prefix.size(), str.size() - prefix.size() - suffix.size());
#endif
}

template <typename T>
inline constexpr string_view fully_qualified_type_name_impl() noexcept {
#if defined(_MSC_VER)
  return extract(
      "class c10::string_view __cdecl c10::util::detail::fully_qualified_type_name_impl<",
      ">(void)",
      __FUNCSIG__);
#elif defined(__clang__)
  return extract(
      "c10::string_view c10::util::detail::fully_qualified_type_name_impl() [T = ",
      "]",
      __PRETTY_FUNCTION__);
#elif defined(__GNUC__)
  return extract(
      "constexpr c10::string_view c10::util::detail::fully_qualified_type_name_impl() [with T = ",
      "]",
      __PRETTY_FUNCTION__);
#endif
}

template <typename T>
inline constexpr uint64_t type_index_impl() {
#if !defined(__CUDA_ARCH__)
// Idea: __PRETTY_FUNCTION__ (or __FUNCSIG__ on msvc) contains a qualified name
// of this function, including its template parameter, i.e. including the
// type we want an id for. We use this name and run crc64 on it to get a type
// id.
#if defined(_MSC_VER)
  return crc64(__FUNCSIG__, sizeof(__FUNCSIG__)).checksum();
#elif defined(__clang__)
  return crc64(__PRETTY_FUNCTION__, sizeof(__PRETTY_FUNCTION__)).checksum();
#elif defined(__GNUC__)
  return crc64(__PRETTY_FUNCTION__, sizeof(__PRETTY_FUNCTION__)).checksum();
#endif
#else
  throw std::logic_error("This should not be called on device code");
#endif
}

} // namespace detail

template <typename T>
inline constexpr type_index get_type_index() noexcept {
#if !defined(__CUDA_ARCH__)
  // To enforce that this is really computed at compile time, we pass the
  // type index through std::integral_constant.
  return type_index{std::integral_constant<
      uint64_t,
      detail::type_index_impl<guts::remove_cv_t<guts::decay_t<T>>>()>::value};
#else
  // There's nothing in theory preventing us from running this on device code
  // except for nvcc throwing a compiler error if we enable it.
  return (
      throw std::logic_error("This should not be called on device code"),
      type_index(0));
#endif
}

template <typename T>
inline constexpr string_view get_fully_qualified_type_name() noexcept {
  return detail::fully_qualified_type_name_impl<T>();
}
} // namespace util
} // namespace c10

C10_DEFINE_HASH_FOR_IDWRAPPER(c10::util::type_index);
