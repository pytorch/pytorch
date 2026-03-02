#pragma once
#ifndef C10_UTIL_CPP17_H_
#define C10_UTIL_CPP17_H_

#include <c10/macros/Macros.h>
#include <functional>
#include <memory>
#include <type_traits>
#include <utility>

#if !defined(__clang__) && !defined(_MSC_VER) && defined(__GNUC__) && \
    __GNUC__ < 9
#error \
    "You're trying to build PyTorch with a too old version of GCC. We need GCC 9 or later."
#endif

#if defined(__clang__) && __clang_major__ < 9
#error \
    "You're trying to build PyTorch with a too old version of Clang. We need Clang 9 or later."
#endif

#if (defined(_MSC_VER) && (!defined(_MSVC_LANG) || _MSVC_LANG < 201703L)) || \
    (!defined(_MSC_VER) && __cplusplus < 201703L)
#error You need C++17 to compile PyTorch
#endif

#if defined(_WIN32) && (defined(min) || defined(max))
#error Macro clash with min and max -- define NOMINMAX when compiling your program on Windows
#endif

/*
 * This header adds some polyfills with C++17 functionality
 */

namespace c10::guts {

#if defined(__HIP__)

// Implementation from http://en.cppreference.com/w/cpp/utility/apply (but
// modified)
// TODO This is an incomplete implementation of std::apply, not working for
// member functions.
namespace detail {
template <class F, class Tuple, std::size_t... INDEX>
C10_HOST_DEVICE constexpr auto apply_impl(
    F&& f,
    Tuple&& t,
    std::index_sequence<INDEX...>) {
  return std::forward<F>(f)(std::get<INDEX>(std::forward<Tuple>(t))...);
}
} // namespace detail

template <class F, class Tuple>
C10_HOST_DEVICE constexpr auto apply(F&& f, Tuple&& t) {
  return detail::apply_impl(
      std::forward<F>(f),
      std::forward<Tuple>(t),
      std::make_index_sequence<
          std::tuple_size<std::remove_reference_t<Tuple>>::value>{});
}

#endif

} // namespace c10::guts

#endif // C10_UTIL_CPP17_H_
