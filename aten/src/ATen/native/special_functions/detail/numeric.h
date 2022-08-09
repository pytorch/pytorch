#pragma once

#include <complex>

namespace at {
namespace native {
namespace special_functions {
namespace detail {
template<typename T1>
struct numeric {
  using value_type = T1;
};

template<typename T1>
struct numeric<std::complex<T1>> {
  using value_type = typename std::complex<T1>::value_type;
};
}
}
}
}
