#pragma once

#include <ATen/native/special/detail/is_complex.h>

namespace at {
namespace native {
namespace special {
namespace detail {
template<typename T1>
constexpr bool is_complex_v = is_complex<T1>::value;
}
}
}
}
