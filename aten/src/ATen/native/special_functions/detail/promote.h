#pragma once

#include <ATen/native/math/promotion_t.h>

namespace at {
namespace native {
namespace special_functions {
namespace detail {
template<typename T1, typename... T>
struct promote {
  using type = decltype(promotion_t<std::decay_t<T1>>{} + typename promote<T...>::type{});
}; // struct promote

template<typename T1>
struct promote<T1> {
  using type = decltype(promotion_t<std::decay_t<T1>>{});
}; // struct promote<T1>
} // namespace detail
} // namespace special_functions
} // namespace native
} // namespace at
