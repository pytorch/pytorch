#pragma once

#include <complex>

#include <ATen/native/special_functions/detail/promotion.h>

namespace at {
namespace native {
namespace special_functions {
namespace detail {
template<typename... T>
using promotion_t = typename promotion<T...>::type;

template<typename T1>
struct promotion<std::complex<T1>, false> {
 private:
  using value_type = typename std::complex<T1>::value_type;
 public:
  using type = decltype(std::complex<promotion_t<value_type>>{});
};
}
}
}
}
