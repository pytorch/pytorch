#pragma once

#include <ATen/native/special_functions/detail/is_complex.h>

namespace at {
namespace native {
namespace special_functions {
namespace detail {
template<typename T1>
using is_complex_t = typename is_complex<T1>::type;
}
}
}
}
