#pragma once

namespace torch {
namespace jit {
namespace tensorexpr {

constexpr auto cpp_intrinsics_definition = R"(
namespace std {

template <typename T,
          typename std::enable_if<std::is_floating_point<T>::value, int>::type = 0>
T rsqrt(T v) {
  return 1.0f / std::sqrt(v);
}

template <typename T,
          typename std::enable_if<std::is_floating_point<T>::value, int>::type = 0>
T frac(T v) {
  T intpart;
  return std::modf(v, &intpart);
}

template <typename From, typename To>
To bitcast(const From& v) {
  assert(sizeof(To) == sizeof(From));
  To res;
  std::memcpy(&res, &v, sizeof(From));
  return res;
}

} // namespace std
)";

} // namespace tensorexpr
} // namespace jit
} // namespace torch
