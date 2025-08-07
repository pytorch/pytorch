#pragma once

#include <torch/csrc/utils/variadic.h>
#include <torch/types.h>

#include <cstdint>
#include <type_traits>

namespace torch::nn {
class Module;
} // namespace torch::nn

namespace torch::detail {
/// Detects if a type T has a forward() method.
template <typename T>
struct has_forward {
  // Declare two types with differing size.
  using yes = int8_t;
  using no = int16_t;

  // Here we declare two functions. The first is only enabled if `&U::forward`
  // is well-formed and returns the `yes` type. In C++, the ellipsis parameter
  // type (`...`) always puts the function at the bottom of overload resolution.
  // This is specified in the standard as: 1) A standard conversion sequence is
  // always better than a user-defined conversion sequence or an ellipsis
  // conversion sequence. 2) A user-defined conversion sequence is always better
  // than an ellipsis conversion sequence This means that if the first overload
  // is viable, it will be preferred over the second as long as we pass any
  // convertible type. The type of `&U::forward` is a pointer type, so we can
  // pass e.g. 0.
  template <typename U>
  static yes test(decltype(&U::forward));
  template <typename U>
  static no test(...);

  // Finally we test statically whether the size of the type returned by the
  // selected overload is the size of the `yes` type.
  static constexpr bool value = (sizeof(test<T>(nullptr)) == sizeof(yes));
};

template <typename Head = void, typename... Tail>
constexpr bool check_not_lvalue_references() {
  return (!std::is_lvalue_reference_v<Head> ||
          std::is_const_v<std::remove_reference_t<Head>>) &&
      check_not_lvalue_references<Tail...>();
}

template <>
inline constexpr bool check_not_lvalue_references<void>() {
  return true;
}

/// A type trait whose `value` member is true if `M` derives from `Module`.
template <typename M>
using is_module = std::is_base_of<torch::nn::Module, std::decay_t<M>>;

template <typename M, typename T = void>
using enable_if_module_t = std::enable_if_t<is_module<M>::value, T>;
} // namespace torch::detail
