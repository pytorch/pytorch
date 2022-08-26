#pragma once

#include <c10/util/complex.h>

namespace at {
namespace native {
namespace special_functions {
namespace detail {
template<typename T1>
struct numeric {
  using value_type = T1;
};

template<typename T1>
struct numeric<c10::complex<T1>> {
  using value_type = typename c10::complex<T1>::value_type;
};
}
}
}
}
