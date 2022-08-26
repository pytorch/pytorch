#pragma once

namespace at {
namespace native {
namespace special_functions {
namespace detail {
template<typename T1, bool = std::is_integral<T1>::value>
struct promotion {
  using type = double;
};

template<typename T1>
struct promotion<T1, false> {};

template<>
struct promotion<float> {
  using type = float;
};

template<>
struct promotion<double> {
  using type = double;
};

template<>
struct promotion<long double> {
  using type = long double;
};
}
}
}
}
