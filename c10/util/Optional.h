#ifndef C10_UTIL_OPTIONAL_H_
#define C10_UTIL_OPTIONAL_H_

#include <optional>
#include <type_traits>

#include <c10/util/Metaprogramming.h>

namespace c10 {
using std::bad_optional_access;
using std::make_optional;
using std::nullopt;
using std::nullopt_t;
using std::optional;

namespace detail_ {
// the call to convert<A>(b) has return type A and converts b to type A iff b
// decltype(b) is implicitly convertible to A
template <class U>
constexpr U convert(U v) {
  return v;
}
} // namespace detail_
template <class T, class F>
constexpr T value_or_else(const optional<T>& v, F&& func) {
  static_assert(
      std::is_convertible_v<
          typename guts::infer_function_traits_t<F>::return_type,
          T>,
      "func parameters must be a callable that returns a type convertible to the value stored in the optional");
  return v.has_value() ? *v : detail_::convert<T>(std::forward<F>(func)());
}

template <class T, class F>
constexpr T value_or_else(optional<T>&& v, F&& func) {
  static_assert(
      std::is_convertible_v<
          typename guts::infer_function_traits_t<F>::return_type,
          T>,
      "func parameters must be a callable that returns a type convertible to the value stored in the optional");
  return v.has_value() ? constexpr_move(std::move(v).contained_val())
                       : detail_::convert<T>(std::forward<F>(func)());
}
} // namespace c10
#endif // C10_UTIL_OPTIONAL_H_
