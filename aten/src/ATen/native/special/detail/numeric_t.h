#pragma once

#include <ATen/native/special/detail/numeric.h>

namespace at {
namespace native {
namespace special {
namespace detail {
template<typename T1>
using numeric_t = typename numeric<T1>::value_type;
}
}
}
}
