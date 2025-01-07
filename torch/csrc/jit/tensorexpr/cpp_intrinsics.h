#pragma once

namespace torch::jit::tensorexpr {

constexpr auto cpp_intrinsics_definition = R"(
namespace std {

template <typename T,
          std::enable_if_t<std::is_floating_point_v<T>, int> = 0>
T rsqrt(T v) {
  return 1.0f / std::sqrt(v);
}

template <typename T,
          std::enable_if_t<std::is_floating_point_v<T>, int> = 0>
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

} // namespace torch::jit::tensorexpr
