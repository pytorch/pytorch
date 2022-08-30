#pragma once

#include <ATen/native/special_functions/detail/promotion.h>

namespace at {
namespace native {
namespace special_functions {
namespace detail {
template<typename... T>
using promotion_t = typename promotion<T...>::type;

template<typename T1>
struct promotion<c10::complex<T1>, false> {
 private:
  using value_type = typename c10::complex<T1>::value_type;
 public:
  using type = decltype(c10::complex<promotion_t<value_type>>{});
};
}
}
}
}
