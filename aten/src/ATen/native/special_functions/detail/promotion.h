#pragma once

namespace at {
namespace native {
namespace special_functions {
namespace detail {
template<typename T1, bool = std::is_integral<T1>::value>
struct promotion {
  using type = double;
}; // struct promotion

template<typename T1>
struct promotion<T1, false> {
}; // struct promotion<T1, false>

template<>
struct promotion<float> {
  using type = float;
}; // struct promotion<float>

template<>
struct promotion<double> {
  using type = double;
}; // struct promotion<double>

template<>
struct promotion<long double> {
  using type = long double;
}; // struct promotion<long double>
} // namespace detail
} // namespace special_functions
} // namespace native
} // namespace at
