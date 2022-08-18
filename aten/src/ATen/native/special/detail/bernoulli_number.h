#pragma once

#include <ATen/native/special/detail/bernoulli_series.h>

namespace at {
namespace native {
namespace special {
namespace detail {
template<typename T1>
constexpr
T1
bernoulli_number(unsigned int n) {
  return bernoulli_series<T1>(n);
}
} // namespace detail
} // namespace special
} // namespace native
} // namespace at
