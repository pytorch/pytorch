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

namespace c10 {

// std::is_pod is deprecated in C++20, std::is_standard_layout and
// std::is_trivial are introduced in C++11, std::conjunction has been introduced
// in C++17.
template <typename T>
using is_pod = std::conjunction<std::is_standard_layout<T>, std::is_trivial<T>>;

template <typename T>
constexpr bool is_pod_v = is_pod<T>::value;

namespace guts {

template <class F, class Tuple>
C10_HOST_DEVICE inline constexpr decltype(auto) apply(F&& f, Tuple&& t) {
  return std::apply(std::forward<F>(f), std::forward<Tuple>(t));
}

} // namespace guts

} // namespace c10

#endif // C10_UTIL_CPP17_H_
