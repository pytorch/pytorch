#ifndef C10_UTIL_OPTIONAL_H_
#define C10_UTIL_OPTIONAL_H_

#include <optional>
#include <type_traits>

// Macros.h is not needed, but it does namespace shenanigans that lots
// of downstream code seems to rely on. Feel free to remove it and fix
// up builds.

namespace c10 {
// NOLINTNEXTLINE(misc-unused-using-decls)
using std::bad_optional_access;
// NOLINTNEXTLINE(misc-unused-using-decls)
using std::make_optional;
// NOLINTNEXTLINE(misc-unused-using-decls)
using std::nullopt;
// NOLINTNEXTLINE(misc-unused-using-decls)
using std::nullopt_t;
// NOLINTNEXTLINE(misc-unused-using-decls)
using std::optional;

#if !defined(FBCODE_CAFFE2) && !defined(C10_NODEPRECATED)

namespace detail_ {
// the call to convert<A>(b) has return type A and converts b to type A iff b
// decltype(b) is implicitly convertible to A
template <class U>
constexpr U convert(U v) {
  return v;
}
} // namespace detail_
template <class T, class F>
[[deprecated(
    "Please use std::optional::value_or instead of c10::value_or_else")]] constexpr T
value_or_else(const std::optional<T>& v, F&& func) {
  static_assert(
      std::is_convertible_v<typename std::invoke_result_t<F>, T>,
      "func parameters must be a callable that returns a type convertible to the value stored in the optional");
  return v.has_value() ? *v : detail_::convert<T>(std::forward<F>(func)());
}

template <class T, class F>
[[deprecated(
    "Please use std::optional::value_or instead of c10::value_or_else")]] constexpr T
value_or_else(std::optional<T>&& v, F&& func) {
  static_assert(
      std::is_convertible_v<typename std::invoke_result_t<F>, T>,
      "func parameters must be a callable that returns a type convertible to the value stored in the optional");
  return v.has_value() ? constexpr_move(std::move(v).contained_val())
                       : detail_::convert<T>(std::forward<F>(func)());
}

#endif

} // namespace c10
#endif // C10_UTIL_OPTIONAL_H_
