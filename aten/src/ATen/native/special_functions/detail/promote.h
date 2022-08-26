#pragma once

#include <ATen/native/special_functions/detail/promote.h>
#include <ATen/native/special_functions/detail/promotion_t.h>

namespace at {
namespace native {
namespace special_functions {
namespace detail {
template<typename T1, typename... T>
struct promote {
  using type = decltype(promotion_t<std::decay_t<T1>>{} + typename promote<T...>::type{});
};

template<typename T1>
struct promote<T1> {
  using type = decltype(promotion_t<std::decay_t<T1>>{});
};
}
}
}
}
