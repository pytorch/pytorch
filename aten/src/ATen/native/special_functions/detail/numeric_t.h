#pragma once

#include <ATen/native/special_functions/detail/numeric.h>


#pragma once

#include <ATen/native/special_functions/detail/numeric.h>

namespace at {
namespace native {
namespace special_functions {
namespace detail {
template<typename T1>
using numeric_t = typename numeric<T1>::value_type;
}
}
}
}
