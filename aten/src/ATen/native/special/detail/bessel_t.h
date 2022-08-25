#pragma once

namespace at {
namespace native {
namespace special {
namespace detail {
template<typename T1, typename T2, typename T3>
struct bessel_t {
  T1 n;
  T2 x;

  T3 j;
  T3 j_derivative;

  T3 y;
  T3 y_derivative;
};
}
}
}
}
