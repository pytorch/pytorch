#pragma once

#include <ATen/native/math/numeric.h>

namespace at {
namespace native {
namespace special_functions {
namespace detail {
template<typename T1>
using numeric_t = typename numeric<T1>::value_type;
} // namespace detail
} // namespace special_functions
} // namespace native
} // namespace at
