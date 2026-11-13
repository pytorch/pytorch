#pragma once

#include <c10/util/ConstexprCrc.h>
#include <c10/util/IdWrapper.h>
#include <cstdint>
#include <ostream>
#include <string>
#include <string_view>
#include <type_traits>

namespace c10::util {

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

template <typename T>
inline constexpr std::string_view fully_qualified_type_name_impl() {
#if defined(_MSC_VER) && !defined(__clang__)
  constexpr std::string_view fun_sig = __FUNCSIG__;
  constexpr std::string_view prefix =
      "c10::util::detail::fully_qualified_type_name_impl<";
#if defined(__NVCC__)
  constexpr std::string_view suffix = ">()";
#else
  constexpr std::string_view suffix = ">(void)";
#endif
#elif defined(__clang__) || defined(__GNUC__)
  constexpr std::string_view fun_sig = __PRETTY_FUNCTION__;
  // Clang uses [T = ...], while GCC and NVCC use [with T = ...].
  constexpr std::string_view prefix = "T = ";
  constexpr std::string_view suffix = "]";
#endif
  constexpr auto prefix_pos = fun_sig.find(prefix);
  constexpr auto start = prefix_pos + prefix.size();
  constexpr auto end = fun_sig.size() - suffix.size();
#if !defined(__CUDA_ARCH__) && !defined(__CUDA_ARCH_LIST__)
  static_assert(prefix_pos != std::string_view::npos);
  static_assert(fun_sig.ends_with(suffix));
#endif
#if defined(__GNUC__) && !defined(__clang__)
  // GCC appends the expansion of the std::string_view return type alias.
  constexpr auto alias_pos = fun_sig.rfind("; std::string_view = ");
#if !defined(__CUDA_ARCH__) && !defined(__CUDA_ARCH_LIST__)
  static_assert(alias_pos != std::string_view::npos);
#endif
  constexpr auto type_end =
      alias_pos == std::string_view::npos ? end : alias_pos;
#else
  constexpr auto type_end = end;
#endif
  return std::string_view(fun_sig.data() + start, type_end - start);
}

#if !defined(__CUDA_ARCH__) && !defined(__CUDA_ARCH_LIST__)
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
#if !defined(__CUDA_ARCH__) && !defined(__CUDA_ARCH_LIST__)
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
// into __PRETTY_FUNCTION__ when abovementioned class is defined in inlined
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
inline constexpr std::string_view get_fully_qualified_type_name() noexcept {
  return detail::fully_qualified_type_name_impl<T>();
}
} // namespace c10::util

C10_DEFINE_HASH_FOR_IDWRAPPER(c10::util::type_index)
