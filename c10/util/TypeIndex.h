#pragma once

#include <c10/util/C++17.h>
#include <c10/util/ConstexprCrc.h>
#include <c10/util/IdWrapper.h>
#include <c10/util/string_view.h>
#include <cinttypes>
#include <functional>

namespace c10 {
namespace util {

// TODO Make it work for more compilers

// Clang works
#if defined(__clang__)

// except for NVCC
#if defined(__CUDACC__)
#define C10_TYPENAME_SUPPORTS_CONSTEXPR 0
#define C10_TYPENAME_CONSTEXPR
#else
#define C10_TYPENAME_SUPPORTS_CONSTEXPR 1
#define C10_TYPENAME_CONSTEXPR constexpr
#endif

// Windows works
#elif defined(_MSC_VER)

// except for NVCC
#if defined(__CUDACC__)
#define C10_TYPENAME_SUPPORTS_CONSTEXPR 0
#define C10_TYPENAME_CONSTEXPR
#else
#define C10_TYPENAME_SUPPORTS_CONSTEXPR 1
#define C10_TYPENAME_CONSTEXPR constexpr
#endif

// GCC works
#elif defined(__GNUC__)

// except when gcc < 9
#if (__GNUC__ < 9) || defined(__CUDACC__)
#define C10_TYPENAME_SUPPORTS_CONSTEXPR 0
#define C10_TYPENAME_CONSTEXPR
#else
#define C10_TYPENAME_SUPPORTS_CONSTEXPR 1
#define C10_TYPENAME_CONSTEXPR constexpr
#endif

// some other compiler we don't know about
#else
#define C10_TYPENAME_SUPPORTS_CONSTEXPR 1
#define C10_TYPENAME_CONSTEXPR constexpr
#endif

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

#if defined(__clang__) && __clang_major__ < 4
// Getting __PRETTY_FUNCTION__ at compile time only works with Clang >= 4
#error "You're running a too old version of Clang. We need Clang 4 or later."
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
inline C10_TYPENAME_CONSTEXPR c10::string_view fully_qualified_type_name_impl() {
#if defined(_MSC_VER) && !defined(__clang__)
#if defined(__NVCC__)
  return extract(
      "c10::basic_string_view<char> c10::util::detail::fully_qualified_type_name_impl<",
      ">()",
      __FUNCSIG__);
#else
  return extract(
      "class c10::basic_string_view<char> __cdecl c10::util::detail::fully_qualified_type_name_impl<",
      ">(void)",
      __FUNCSIG__);
#endif
#elif defined(__clang__)
  return extract(
      "c10::string_view c10::util::detail::fully_qualified_type_name_impl() [T = ",
      "]",
      __PRETTY_FUNCTION__);
#elif defined(__GNUC__)
  return extract(
#if C10_TYPENAME_SUPPORTS_CONSTEXPR
      "constexpr c10::string_view c10::util::detail::fully_qualified_type_name_impl() [with T = ",
#else
      "c10::string_view c10::util::detail::fully_qualified_type_name_impl() [with T = ",
#endif
      "; c10::string_view = c10::basic_string_view<char>]",
      __PRETTY_FUNCTION__);
#endif
}

#if !defined(__CUDA_ARCH__)
template <typename T>
inline constexpr uint64_t type_index_impl() {
// Idea: __PRETTY_FUNCTION__ (or __FUNCSIG__ on msvc) contains a qualified name
// of this function, including its template parameter, i.e. including the
// type we want an id for. We use this name and run crc64 on it to get a type
// id.
#if defined(_MSC_VER) && !defined(__clang__)
  return crc64(__FUNCSIG__, sizeof(__FUNCSIG__)).checksum();
#elif defined(__clang__)
  return crc64(__PRETTY_FUNCTION__, sizeof(__PRETTY_FUNCTION__)).checksum();
#elif defined(__GNUC__)
  return crc64(__PRETTY_FUNCTION__, sizeof(__PRETTY_FUNCTION__)).checksum();
#endif
}
#endif

} // namespace detail

template <typename T>
inline constexpr type_index get_type_index() {
#if !defined(__CUDA_ARCH__)
  // To enforce that this is really computed at compile time, we pass the
  // type index through std::integral_constant.
  return type_index{std::integral_constant<
      uint64_t,
      detail::type_index_impl<std::decay_t<T>>()>::value};
#else
  // There's nothing in theory preventing us from running this on device code
  // except for nvcc throwing a compiler error if we enable it.
  return (abort(), type_index(0));
#endif
}

#if !defined(TORCH_PEDANTIC)
// Use precomputed hashsum for std::string
// Needed to workaround ambiguity in class name resolution
// into __PRETTY_FUNCION__ when abovementioned class is defined in inlined
// namespace. In multi-ABI C++ library, `std::string` is an alias to
// `std::__cxx11::basic_string<char>` which depending on compiler flags can be
// resolved to `basic_string<char>` either in `std` namespace or in
// `std::__cxx11` one (`__cxx11` is an inline namespace)
template <>
inline constexpr type_index get_type_index<std::string>() {
  // hashsum for std::basic_string<char>
  return type_index{4193213214807308375ULL};
}
#endif

template <typename T>
inline C10_TYPENAME_CONSTEXPR string_view
get_fully_qualified_type_name() noexcept {
#if C10_TYPENAME_SUPPORTS_CONSTEXPR
  constexpr
#else
  static
#endif
      string_view name = detail::fully_qualified_type_name_impl<T>();
  return name;
}
} // namespace util
} // namespace c10

C10_DEFINE_HASH_FOR_IDWRAPPER(c10::util::type_index);
